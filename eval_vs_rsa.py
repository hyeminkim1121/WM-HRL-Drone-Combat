"""eval_vs_rsa.py — 학습된 모델 vs Mirror RSA 평가 (sampling 기반).

WM+HRL 또는 PPO+HRL ckpt를 로드해서 N episode 동안 평가.
새로 대칭화된 Mirror RSA 환경에서 실제 성능 측정.

실행:
  # WM+HRL
  python paper_wm/eval_vs_rsa.py --model wm --ckpt paper_results/wm_v6_s1/wm_v2_final.pt --n_episodes 100

  # PPO+HRL
  python paper_wm/eval_vs_rsa.py --model ppo --ckpt paper_results/ppo_hrl_s1/ppo_hrl_final.pt --n_episodes 100
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

# 단일 config (학습과 동일 setup)
from config import ENV_CFG
import wm_upper_v2
wm_upper_v2.ENV_CFG = ENV_CFG

from env.drone_env import DroneEnv
from env.units import UnitType
from lower_agent import rulebased_actions
from wm_upper_v2 import set_discrete_subgoals
from config import K, WM_CFG
from models.networks import STATE_DIM


def evaluate_wm(ckpt_path: str, n_episodes: int = 100, base_seed: int = 10**8,
                device: str = "cuda", greedy: bool = False) -> dict:
    """WM+HRL 모델 평가 (Categorical sampling)."""
    from wm_upper_v2 import WMUpperTrainerV2

    if greedy:
        ENV_CFG["mirror_greedy"] = True

    trainer = WMUpperTrainerV2(cfg=WM_CFG, device=device)
    trainer.load(ckpt_path)
    trainer.tssm.eval()
    trainer.actor.eval()
    trainer.critic.eval()

    results = []
    for i in range(n_episodes):
        seed = base_seed + i
        ep, won, lost, rsa = trainer._collect_episode(seed=seed, eval_mode=True)
        # ep["rewards"]는 K-step 누적 보상 (discounted) — 우리는 raw sum 다시 계산
        # 직접 env 돌려서 raw reward 측정
        ep_total_reward, env_steps_actual = _replay_for_reward(trainer, seed, "wm")
        results.append({
            "won": won, "lost": lost, "rsa": rsa,
            "meta": ep["meta"],
            "raw_reward": ep_total_reward,
            "env_steps_replayed": env_steps_actual,
        })
        if (i + 1) % 10 == 0:
            wins = sum(1 for r in results if r["won"])
            print(f"  [{i+1}/{n_episodes}] wins so far: {wins} ({100*wins/(i+1):.1f}%)")

    return _summarize(results, n_episodes, "WM+HRL")


def evaluate_ppo(ckpt_path: str, n_episodes: int = 100, base_seed: int = 10**8,
                 device: str = "cuda", greedy: bool = False) -> dict:
    """PPO+HRL 모델 평가 (Categorical sampling)."""
    from ppo_hrl_upper import PPOHRLTrainer, PPO_CFG

    if greedy:
        ENV_CFG["mirror_greedy"] = True

    trainer = PPOHRLTrainer(cfg=PPO_CFG, device=device)
    trainer.load(ckpt_path)
    trainer.encoder.eval()
    trainer.actor.eval()
    trainer.critic.eval()

    results = []
    for i in range(n_episodes):
        seed = base_seed + i
        ep, won, lost, rsa = trainer._collect_episode(seed=seed, eval_mode=True)
        # PPO 코드는 ep_reward를 meta에 포함
        results.append({
            "won": won, "lost": lost, "rsa": rsa,
            "meta": ep["meta"],
            "raw_reward": float(ep["meta"].get("ep_reward", 0.0)),
            "env_steps_replayed": int(ep["meta"].get("env_steps", 0)),
        })
        if (i + 1) % 10 == 0:
            wins = sum(1 for r in results if r["won"])
            print(f"  [{i+1}/{n_episodes}] wins so far: {wins} ({100*wins/(i+1):.1f}%)")

    return _summarize(results, n_episodes, "PPO+HRL")


def _replay_for_reward(trainer, seed: int, model_type: str) -> tuple:
    """동일 seed로 다시 돌려 reward 누적치 계산 (v6는 meta에 ep_reward 없음)."""
    # 단순화: 학습된 정책으로 한 번 더 돌리고 step_rew 누적
    # (deterministic: 같은 seed로 같은 시작, sampling은 randomness 있어 약간 변동 가능)
    env = DroneEnv(config=ENV_CFG, render_mode=None)
    env.reset(seed=seed)
    dev = trainer.device
    n_agents = trainer.n_agents
    n_targets = trainer.n_targets

    last_infos = {}
    cur_indices = np.zeros(n_agents, dtype=np.int64)
    step_in_K = 0
    env_steps = 0
    total_reward = 0.0

    state = trainer.tssm.encode_obs(torch.zeros(1, 205, device=dev))
    cur_onehot = np.zeros(n_agents * n_targets, dtype=np.float32)

    while env.agents:
        obs = env.get_global_state()
        if step_in_K == 0:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                state = trainer.tssm.encode_obs(obs_t, state.h)
                feat = torch.cat([state.h, state.z], dim=-1)
                logits = trainer.actor.net(feat).reshape(1, n_agents, n_targets)
                logits = type(trainer.actor)._apply_defender_mask(logits)
                dist = torch.distributions.Categorical(logits=logits)
                cur_indices = dist.sample().squeeze(0).cpu().numpy()
            set_discrete_subgoals(env, cur_indices)
            cur_onehot = np.zeros((n_agents, n_targets), dtype=np.float32)
            for j, idx in enumerate(cur_indices):
                cur_onehot[j, idx] = 1.0
            cur_onehot = cur_onehot.flatten()
            with torch.no_grad():
                act_t = torch.tensor(cur_onehot, dtype=torch.float32, device=dev).unsqueeze(0)
                state = trainer.tssm.step_dynamics(state, act_t)

        # 서브골 없는 에이전트 즉시 재할당
        agents_no_subgoal = [a for a in env.agents if env._subgoals.get(a) is None]
        if agents_no_subgoal:
            with torch.no_grad():
                obs_new = env.get_global_state()
                obs_t = torch.tensor(obs_new, dtype=torch.float32, device=dev).unsqueeze(0)
                state = trainer.tssm.encode_obs(obs_t, state.h)
                feat = torch.cat([state.h, state.z], dim=-1)
                logits = trainer.actor.net(feat).reshape(1, n_agents, n_targets)
                logits = type(trainer.actor)._apply_defender_mask(logits)
                dist = torch.distributions.Categorical(logits=logits)
                new_indices = dist.sample().squeeze(0).cpu().numpy()
            partial = {}
            enemy_bases = env._enemy_bases
            ally_bases  = env._ally_bases
            enemy_defs  = [u for u in env._enemy_units
                           if u.unit_type == UnitType.DEFEND_DRONE]
            for a in agents_no_subgoal:
                unit = env._units.get(a)
                if unit is None or not unit.alive:
                    continue
                idx = env.possible_agents.index(a)
                target_idx = int(new_indices[idx])
                if unit.unit_type == UnitType.ATTACK_DRONE:
                    if target_idx < 5 and enemy_bases[target_idx].alive:
                        partial[a] = (enemy_bases[target_idx].x, enemy_bases[target_idx].y)
                    elif target_idx >= 5:
                        def_idx = target_idx - 5
                        if def_idx < len(enemy_defs) and enemy_defs[def_idx].alive:
                            d = enemy_defs[def_idx]
                            partial[a] = (d.x, d.y)
                else:
                    zone_idx = target_idx if target_idx < 5 else target_idx - 5
                    base = ally_bases[zone_idx]
                    partial[a] = (base.x, base.y)
            if partial:
                env.set_subgoals(partial)

        actions = rulebased_actions(env)
        _, rewards, terminations, truncations, infos = env.step(actions)
        env_steps += 1
        if infos:
            last_infos = next(iter(infos.values()))
        if rewards:
            total_reward += float(np.mean(list(rewards.values())))
        step_in_K = (step_in_K + 1) % K

        ep_done = (
            all(terminations.values()) or all(truncations.values())
            or bool(last_infos.get("won")) or bool(last_infos.get("lost"))
        )
        if ep_done:
            break

    env.close()
    return total_reward, env_steps


def _summarize(results, n_episodes, label):
    wins   = sum(1 for r in results if r["won"])
    losses = sum(1 for r in results if r["lost"])
    draws  = n_episodes - wins - losses
    mean_reward = np.mean([r["raw_reward"] for r in results])
    std_reward  = np.std([r["raw_reward"] for r in results])
    mean_env_len = np.mean([r["meta"].get("env_steps", 0) for r in results])
    mean_survived = np.mean([r["meta"].get("drones_survived", 0) for r in results])
    mean_enemy_bd = np.mean([r["meta"].get("enemy_bases_destroyed", 0) for r in results])
    mean_ally_bd  = np.mean([r["meta"].get("ally_bases_destroyed", 0) for r in results])

    print(f"\n{'='*55}")
    print(f"  {label} vs Mirror RSA  ({n_episodes} episodes)")
    print(f"{'='*55}")
    print(f"  win_rate:       {wins/n_episodes:.3f}  ({wins}W / {losses}L / {draws}D)")
    print(f"  mean_reward:    {mean_reward:+.2f}  (std {std_reward:.2f})")
    print(f"  mean_env_len:   {mean_env_len:.1f}  steps")
    print(f"  drones_survived:{mean_survived:.2f} / 8")
    print(f"  enemy_bd:       {mean_enemy_bd:.2f} / 5")
    print(f"  ally_bd:        {mean_ally_bd:.2f} / 5")
    print(f"{'='*55}\n")
    return {
        "win_rate": wins/n_episodes, "wins": wins, "losses": losses, "draws": draws,
        "mean_reward": mean_reward, "std_reward": std_reward,
        "mean_env_len": mean_env_len, "mean_survived": mean_survived,
        "mean_enemy_bd": mean_enemy_bd, "mean_ally_bd": mean_ally_bd,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wm", "ppo"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--base_seed", type=int, default=10**8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--greedy", action="store_true",
                   help="Mirror RSA greedy mode (nearest target instead of random)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        print(f"ERROR: ckpt not found: {args.ckpt}")
        sys.exit(1)
    rsa_label = "GREEDY Mirror RSA" if args.greedy else "RANDOM Mirror RSA"
    print(f"Loading {args.model} ckpt: {args.ckpt}")
    print(f"Evaluating {args.n_episodes} episodes (sampling) vs {rsa_label}\n")

    if args.model == "wm":
        evaluate_wm(args.ckpt, args.n_episodes, args.base_seed, args.device, args.greedy)
    else:
        evaluate_ppo(args.ckpt, args.n_episodes, args.base_seed, args.device, args.greedy)


if __name__ == "__main__":
    main()
