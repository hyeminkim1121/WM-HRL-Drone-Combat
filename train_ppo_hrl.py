"""train_ppo_hrl.py — PPO+HRL baseline 학습 엔트리포인트.

v6 (WM+HRL)와 같은 환경/액션공간/Lower agent. WM 제거, on-policy PPO.

실행:
  python paper_wm/train_ppo_hrl.py --seed 43 --name ppo_hrl_s1 --wandb
  python paper_wm/train_ppo_hrl.py --seed 44 --name ppo_hrl_s2 --wandb

config_v3 ENV_CFG (대칭 + 보상 수정 + reinforce OFF) monkey-patch — v6와 동일.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# WM+HRL과 동일한 환경/액션공간을 N·M 파라메트릭으로 derive
import config as C
import wm_upper_v2

from ppo_hrl_upper import PPOHRLTrainer, PPO_CFG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",     type=int, default=43)
    p.add_argument("--name",     type=str, default=None)
    p.add_argument("--out_dir",  type=str, default="paper_results")
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=None,
                   help="총 학습 episode 오버라이드 (default 100K)")
    p.add_argument("--group",    type=str, default="ppo-sweep", help="wandb group")
    p.add_argument("--wandb",    action="store_true")
    p.add_argument("--resume",   type=str, default=None)
    # ── N·M 스케일 노브 (WM 스윕과 동일) ──
    p.add_argument("--n_ally_attack",  type=int, default=5)
    p.add_argument("--n_ally_defend",  type=int, default=3)
    p.add_argument("--n_ally_base",    type=int, default=5)
    p.add_argument("--n_enemy_attack", type=int, default=5)
    p.add_argument("--n_enemy_defend", type=int, default=3)
    p.add_argument("--n_enemy_base",   type=int, default=5)
    p.add_argument("--focus_fire_k", type=int, default=1)
    p.add_argument("--enemy_power",  type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # 유닛 수 → (ENV_CFG, WM_CFG) derive; PPO는 WM과 동일 환경/액션공간을 공유
    env_cfg, wm_cfg = C.make_configs(
        n_ally_attack=args.n_ally_attack, n_ally_defend=args.n_ally_defend, n_ally_base=args.n_ally_base,
        n_enemy_attack=args.n_enemy_attack, n_enemy_defend=args.n_enemy_defend, n_enemy_base=args.n_enemy_base,
    )
    env_cfg["focus_fire_k"] = args.focus_fire_k
    if args.enemy_power is not None:
        env_cfg["enemy_atk_power"] = args.enemy_power
    wm_upper_v2.ENV_CFG = env_cfg   # ppo _collect_episode가 모듈 ENV_CFG 사용

    cfg = dict(PPO_CFG)
    # N·M 파생 차원을 PPO cfg에 주입 (네트워크 크기/마스크)
    cfg["state_dim"]            = wm_cfg["tssm"]["state_dim"]
    cfg["n_action_agents"]      = wm_cfg["n_action_agents"]
    cfg["n_action_targets"]     = wm_cfg["n_action_targets"]
    cfg["n_action_attackers"]   = wm_cfg["n_action_attackers"]
    cfg["n_action_ally_zones"]  = wm_cfg["n_action_ally_zones"]
    if args.episodes is not None:
        cfg["total_episodes"] = args.episodes

    N, M    = cfg["n_action_agents"], cfg["n_action_targets"]
    ff_tag  = f"_ff{args.focus_fire_k}" if args.focus_fire_k > 1 else ""
    name    = args.name or f"ppo{ff_tag}_N{N}_M{M}_s{args.seed}"
    out_dir = os.path.join(args.out_dir, name)

    if args.wandb:
        try:
            import wandb
            _orig = wandb.init
            def _init(**kwargs):
                kwargs["group"] = args.group
                return _orig(**kwargs)
            wandb.init = _init
        except ImportError:
            pass

    print(f"[ppo_hrl] N={N} M={M} (state_dim={cfg['state_dim']}, action_dim={N*M}) "
          f"| 행동공간 ≈ M^N = {M}^{N}")

    trainer = PPOHRLTrainer(cfg=cfg, device=args.device)
    if args.resume:
        trainer.load(args.resume)
        print(f"  재개: {args.resume}  (ep {trainer.episode_count})")

    print(f"[ppo_hrl] total_episodes = {cfg['total_episodes']}")
    print(f"[ppo_hrl] rollout_macro_steps = {cfg['rollout_macro_steps']}")

    trainer.train(out_dir=out_dir, seed=args.seed,
                  use_wandb=args.wandb, run_name=name)


if __name__ == "__main__":
    main()
