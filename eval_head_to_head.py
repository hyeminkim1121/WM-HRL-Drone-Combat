"""eval_head_to_head.py — 학습된 PPO/WM+HRL 모델끼리 직접 맞붙임.

한 쪽은 아군(ally) upper agent, 다른 쪽은 적군(enemy) upper agent.
환경이 좌우 대칭(x-mirror, 5atk+3def 양쪽 동일)이라 obs 미러링으로 동일 모델
인터페이스 재사용.

실행:
  # WM+HRL ally vs PPO+HRL enemy
  python paper_wm/eval_head_to_head.py \
      --ally wm  --ally_ckpt  paper_results/wm_v6_s1/wm_v2_final.pt \
      --enemy ppo --enemy_ckpt paper_results/ppo_hrl_s1/ppo_hrl_final.pt \
      --n_episodes 100

  # 반대도 테스트 (PPO ally vs WM enemy)
"""
from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import ENV_CFG
import wm_upper_v2
wm_upper_v2.ENV_CFG = ENV_CFG

from env.drone_env import DroneEnv
from env.units import UnitType
from lower_agent import rulebased_actions
from wm_upper_v2 import set_discrete_subgoals
from config import K, WM_CFG
from models.networks import STATE_DIM


# ---------------------------------------------------------------------------
# Obs mirror (ally↔enemy + x-flip)
# ---------------------------------------------------------------------------
def mirror_obs_from_env(env: DroneEnv) -> np.ndarray:
    """env의 현재 상태에서 적군 시점 obs(120-dim)를 만든다.

    대칭 obs 레이아웃 (env.get_global_state):
      [ally drones 8×5=40] [ally bases 5×4=20]
      [enemy drones 8×5=40] [enemy bases 5×4=20]
    적군 시점 = ally↔enemy swap + x좌표 (G - x) 반전.
    """
    G  = float(env.grid_size)
    ZM = float(env.z_max)
    buf: list[float] = []

    # "ally drones" (8 slots = 5 atk + 3 def) ← enemy drones (5 atk + 3 def)
    enemy_atk = [u for u in env._enemy_units if u.unit_type == UnitType.ATTACK_DRONE]
    enemy_def = [u for u in env._enemy_units if u.unit_type == UnitType.DEFEND_DRONE]
    for i in range(5):
        if i < len(enemy_atk) and enemy_atk[i].alive:
            u = enemy_atk[i]
            buf.extend([(G - u.x)/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
        else:
            buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(3):
        if i < len(enemy_def) and enemy_def[i].alive:
            u = enemy_def[i]
            buf.extend([(G - u.x)/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
        else:
            buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # "ally bases" (5) ← enemy bases (x mirrored)
    for b in env._enemy_bases:
        buf.extend([(G - b.x)/G, b.y/G, b.hp/b.max_hp, float(b.alive)])

    # "enemy drones" (8 slots) ← ally drones (5 atk + 3 def)
    for i in range(5):
        u = env._units.get(f"ally_attack_{i}")
        if u and u.alive:
            buf.extend([(G - u.x)/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
        else:
            buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(3):
        u = env._units.get(f"ally_defend_{i}")
        if u and u.alive:
            buf.extend([(G - u.x)/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
        else:
            buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # "enemy bases" (5) ← ally bases (x mirrored)
    for b in env._ally_bases:
        buf.extend([(G - b.x)/G, b.y/G, b.hp/b.max_hp, float(b.alive)])

    arr = np.asarray(buf, dtype=np.float32)
    assert arr.shape[0] == STATE_DIM, f"mirror obs dim mismatch: {arr.shape}"
    return arr


# ---------------------------------------------------------------------------
# Enemy subgoal application from model output
# ---------------------------------------------------------------------------
def set_enemy_subgoals_from_indices(env: DroneEnv, indices: np.ndarray) -> None:
    """모델이 enemy 시점에서 출력한 (8 agents × target_idx)를 enemy 유닛에 적용.

    매핑(ally의 set_discrete_subgoals + env.set_subgoals 미러):
      - enemy_attack_i (i=0..4): target_idx 0..4 → ally_base[t], 5..7 → ally_def[t-5]
      - enemy_defend_i (i=0..2): target_idx [0..4] → enemy_base[zone] 기준 nearest ally attacker
    """
    alive_ally_bases = [b for b in env._ally_bases if b.alive]
    alive_ally_defs  = [env._units[f"ally_defend_{i}"]
                        for i in range(3)
                        if env._units.get(f"ally_defend_{i}")
                        and env._units[f"ally_defend_{i}"].alive]
    alive_ally_atk   = [env._units[f"ally_attack_{i}"]
                        for i in range(5)
                        if env._units.get(f"ally_attack_{i}")
                        and env._units[f"ally_attack_{i}"].alive]

    # Enemy attackers (model agents 0-4)
    for i in range(5):
        eid = f"enemy_attack_{i}"
        unit = env._units.get(eid)
        if unit is None or not unit.alive:
            continue
        target_idx = int(indices[i])
        tgt = None
        if target_idx < 5:
            if target_idx < len(env._ally_bases) and env._ally_bases[target_idx].alive:
                tgt = env._ally_bases[target_idx]
            elif alive_ally_bases:
                tgt = min(alive_ally_bases,
                          key=lambda b: abs(b.x - unit.x) + abs(b.y - unit.y))
        else:
            def_idx = target_idx - 5
            if def_idx < len(alive_ally_defs):
                tgt = alive_ally_defs[def_idx]
            elif alive_ally_defs:
                tgt = min(alive_ally_defs,
                          key=lambda d: abs(d.x - unit.x) + abs(d.y - unit.y))
            elif alive_ally_bases:
                tgt = min(alive_ally_bases,
                          key=lambda b: abs(b.x - unit.x) + abs(b.y - unit.y))
        if tgt is not None:
            env._enemy_subgoals[eid] = (tgt.x, tgt.y)

    # Enemy defenders (model agents 5-7) — zone-based nearest ally attacker
    if alive_ally_atk:
        for i in range(3):
            eid = f"enemy_defend_{i}"
            unit = env._units.get(eid)
            if unit is None or not unit.alive:
                continue
            target_idx = int(indices[5 + i])
            zone_idx = target_idx if target_idx < 5 else target_idx - 5
            zone_base = env._enemy_bases[zone_idx]
            tx, ty = zone_base.x, zone_base.y
            tgt = min(alive_ally_atk,
                      key=lambda e: abs(e.x - tx) + abs(e.y - ty))
            env._enemy_atk_targets[eid] = tgt.unit_id
            env._enemy_subgoals[eid] = (tgt.x, tgt.y)


# ---------------------------------------------------------------------------
# Model wrappers (output target indices given obs)
# ---------------------------------------------------------------------------
class WMModelWrapper:
    """WM+HRL 추론 wrapper. Categorical sampling으로 target indices 출력."""
    def __init__(self, ckpt_path: str, device: str):
        from wm_upper_v2 import WMUpperTrainerV2
        self.trainer = WMUpperTrainerV2(cfg=WM_CFG, device=device)
        self.trainer.load(ckpt_path)
        self.trainer.tssm.eval()
        self.trainer.actor.eval()
        self.device = device
        self.n_agents  = self.trainer.n_agents
        self.n_targets = self.trainer.n_targets
        self.state = None  # latent (h, z)

    def reset(self) -> None:
        # init zero latent
        z0 = self.trainer.tssm.encode_obs(torch.zeros(1, STATE_DIM, device=self.device))
        self.state = z0
        self.last_act_onehot = np.zeros(self.n_agents * self.n_targets, dtype=np.float32)

    @torch.no_grad()
    def select(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.state = self.trainer.tssm.encode_obs(obs_t, self.state.h)
        feat = torch.cat([self.state.h, self.state.z], dim=-1)
        logits = self.trainer.actor.net(feat).reshape(1, self.n_agents, self.n_targets)
        logits = type(self.trainer.actor)._apply_defender_mask(logits)
        dist = torch.distributions.Categorical(logits=logits)
        indices = dist.sample().squeeze(0).cpu().numpy()
        # advance dynamics (so latent state reflects this action for next K)
        cur_onehot = np.zeros((self.n_agents, self.n_targets), dtype=np.float32)
        for j, idx in enumerate(indices):
            cur_onehot[j, idx] = 1.0
        cur_onehot = cur_onehot.flatten()
        act_t = torch.tensor(cur_onehot, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.state = self.trainer.tssm.step_dynamics(self.state, act_t)
        return indices


class PPOModelWrapper:
    """PPO+HRL 추론 wrapper. Categorical sampling으로 target indices 출력."""
    def __init__(self, ckpt_path: str, device: str):
        from ppo_hrl_upper import PPOHRLTrainer, PPO_CFG
        self.trainer = PPOHRLTrainer(cfg=PPO_CFG, device=device)
        self.trainer.load(ckpt_path)
        self.trainer.encoder.eval()
        self.trainer.actor.eval()
        self.device = device
        self.n_agents  = self.trainer.n_agents
        self.n_targets = self.trainer.n_targets

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feat = self.trainer.encoder(obs_t)
        logits = self.trainer.actor.net(feat).reshape(1, self.n_agents, self.n_targets)
        logits = type(self.trainer.actor)._apply_defender_mask(logits)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().squeeze(0).cpu().numpy()


def make_model(model_type: str, ckpt: str, device: str):
    if model_type == "wm":
        return WMModelWrapper(ckpt, device)
    elif model_type == "ppo":
        return PPOModelWrapper(ckpt, device)
    raise ValueError(model_type)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(env: DroneEnv, ally_model, enemy_model, seed: int) -> dict:
    """enemy_model=None이면 env Mirror RSA(random) 사용."""
    env.reset(seed=seed)
    ally_model.reset()
    if enemy_model is not None:
        enemy_model.reset()
        # 1) Enemy subgoals: env auto-init을 model 출력으로 즉시 덮어쓰기
        enemy_obs = mirror_obs_from_env(env)
        enemy_indices = enemy_model.select(enemy_obs)
        env._enemy_subgoals = {}
        env._enemy_atk_targets = {}
        set_enemy_subgoals_from_indices(env, enemy_indices)
        # 2) env가 K마다 자동 재할당하지 않게 (외부 제어)
        env._mirror_subgoal_timer = -10**9
    # else: env's Mirror RSA가 그대로 작동 (random subgoal, K 주기 자동 재할당)

    step_in_K = 0
    env_steps = 0
    total_reward = 0.0
    last_infos = {}
    won = False
    lost = False
    tiebreak = False  # 공격드론 전멸 → 베이스 수 비교로 결정

    while env.agents:
        if step_in_K == 0:
            # Ally upper: K마다 sampling
            ally_obs = env.get_global_state()
            ally_indices = ally_model.select(ally_obs)
            set_discrete_subgoals(env, ally_indices)
            # Enemy upper: K마다 sampling (mirrored obs) — 모델일 때만
            if enemy_model is not None:
                enemy_obs = mirror_obs_from_env(env)
                enemy_indices = enemy_model.select(enemy_obs)
                set_enemy_subgoals_from_indices(env, enemy_indices)

        # Ally lower: rulebased
        actions = rulebased_actions(env)
        _, rewards, terminations, truncations, infos = env.step(actions)
        env_steps += 1
        if infos:
            last_infos = next(iter(infos.values()))
        if rewards:
            total_reward += float(np.mean(list(rewards.values())))
        step_in_K = (step_in_K + 1) % K

        # env가 tiebreak 자동 처리 → infos에서 그대로 읽기
        if last_infos.get("won"):     won = True
        if last_infos.get("lost"):    lost = True
        if last_infos.get("tiebreak"): tiebreak = True

        ep_done = (
            all(terminations.values()) or all(truncations.values()) or won or lost
            or tiebreak
        )
        if ep_done:
            break

    drones_survived = sum(1 for a in env.possible_agents
                          if env._units.get(a) and env._units[a].alive)
    enemy_bd = sum(1 for b in env._enemy_bases if not b.alive)
    ally_bd  = sum(1 for b in env._ally_bases  if not b.alive)
    env.close()
    return {
        "won": won, "lost": lost, "tiebreak": tiebreak,
        "total_reward": total_reward,
        "env_steps": env_steps,
        "drones_survived": drones_survived,
        "enemy_bd": enemy_bd,
        "ally_bd":  ally_bd,
    }


def run_one_seed(ally_model, enemy_model, n_episodes: int, base_seed: int,
                 verbose: bool = True) -> list:
    env = DroneEnv(config=ENV_CFG, render_mode=None)
    results = []
    for i in range(n_episodes):
        r = run_episode(env, ally_model, enemy_model, seed=base_seed + i)
        results.append(r)
        if verbose and (i + 1) % 25 == 0:
            wins = sum(1 for x in results if x["won"])
            print(f"    [{i+1}/{n_episodes}] wins so far: {wins} "
                  f"({100*wins/(i+1):.1f}%)")
    return results


def aggregate(results: list, n_episodes: int) -> dict:
    wins   = sum(1 for x in results if x["won"])
    losses = sum(1 for x in results if x["lost"])
    return {
        "win_rate":  wins   / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": (n_episodes - wins - losses) / n_episodes,
        "tiebreak":  sum(1 for x in results if x.get("tiebreak")) / n_episodes,
        "mean_reward":     float(np.mean([x["total_reward"]    for x in results])),
        "mean_env_len":    float(np.mean([x["env_steps"]       for x in results])),
        "drones_survived": float(np.mean([x["drones_survived"] for x in results])),
        "enemy_bd":        float(np.mean([x["enemy_bd"]        for x in results])),
        "ally_bd":         float(np.mean([x["ally_bd"]         for x in results])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ally",       choices=["wm", "ppo"], required=True)
    p.add_argument("--ally_ckpt",  type=str, required=True)
    p.add_argument("--enemy_mode", choices=["rsa", "model"], default="model",
                   help="rsa: env Mirror RSA(random) / model: 학습된 모델")
    p.add_argument("--enemy",      choices=["wm", "ppo"], default=None)
    p.add_argument("--enemy_ckpt", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--seeds",      type=int, nargs="+",
                   default=[10**8, 2*10**8, 3*10**8])
    p.add_argument("--device",     type=str, default="cuda")
    args = p.parse_args()

    if args.enemy_mode == "model":
        if not args.enemy or not args.enemy_ckpt:
            raise SystemExit("--enemy_mode model 사용 시 --enemy/--enemy_ckpt 필수")
        enemy_label = f"{args.enemy} ({args.enemy_ckpt})"
    else:
        enemy_label = "Mirror RSA (random)"

    print(f"[h2h]  ALLY = {args.ally} ({args.ally_ckpt})")
    print(f"       ENEMY= {enemy_label}")
    print(f"  Episodes/seed: {args.n_episodes}, Seeds: {args.seeds}\n")

    ally_model  = make_model(args.ally,  args.ally_ckpt,  args.device)
    enemy_model = (make_model(args.enemy, args.enemy_ckpt, args.device)
                   if args.enemy_mode == "model" else None)

    per_seed = []
    for s_idx, base_seed in enumerate(args.seeds, 1):
        print(f"  seed[{s_idx}/{len(args.seeds)}] = {base_seed}")
        results = run_one_seed(ally_model, enemy_model,
                               args.n_episodes, base_seed, verbose=True)
        agg = aggregate(results, args.n_episodes)
        print(f"    → win {agg['win_rate']:.3f}  loss {agg['loss_rate']:.3f}  "
              f"draw {agg['draw_rate']:.3f}  enemy_bd {agg['enemy_bd']:.2f}  "
              f"ally_bd {agg['ally_bd']:.2f}")
        per_seed.append(agg)

    keys = ["win_rate","loss_rate","draw_rate","tiebreak",
            "mean_reward","mean_env_len","drones_survived","enemy_bd","ally_bd"]
    means = {k: float(np.mean([s[k] for s in per_seed])) for k in keys}
    stds  = {k: float(np.std ([s[k] for s in per_seed])) for k in keys}

    print(f"\n{'='*64}")
    print(f"  ALLY={args.ally}  vs  ENEMY={enemy_label}")
    print(f"  {len(args.seeds)} seeds × {args.n_episodes} ep  "
          f"({len(args.seeds)*args.n_episodes} total)")
    print(f"  Tie-break: 양측 공격드론 전멸 시 base 수로 결정")
    print(f"{'='*64}")
    for k in keys:
        print(f"  {k:18s}: {means[k]:+.4f}  ± {stds[k]:.4f}")
    print(f"{'='*64}")
    print(f"  per-seed win_rate: {[round(s['win_rate'],3) for s in per_seed]}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
