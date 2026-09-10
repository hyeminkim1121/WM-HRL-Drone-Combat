"""run_revision_experiments.py — 리비전 실험 일괄 실행 스크립트.

실험 목록:
  0. WM+HRL baseline 재학습 (3 seeds, K=10, H=15)
  1. PPO+HRL 재학습 (3 seeds)
  2. Ablation: K 변화 (K=1, 5, 20)  — baseline K=10은 wm_baseline 사용
  3. Ablation: H 변화 (H=5, 10, 25) — baseline H=15는 wm_baseline 사용
  4. Ablation: Entity Encoder 제거 (use_entity_encoder=False)

실행:
  python run_revision_experiments.py --exp wm_baseline  # WM+HRL baseline만
  python run_revision_experiments.py --exp ppo          # PPO 재학습만
  python run_revision_experiments.py --exp ablation_k   # K 변화만
  python run_revision_experiments.py --exp ablation_h   # H 변화만
  python run_revision_experiments.py --exp ablation_ee  # Entity Encoder 제거만
  python run_revision_experiments.py --exp all          # 전부
  python run_revision_experiments.py --exp all --wandb  # wandb 연동
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from config import WM_CFG, ENV_CFG
from ppo_hrl_upper import PPO_CFG


SEEDS = [42, 43, 44]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_results")


def run_wm_baseline(seeds, use_wandb=False, device="cuda"):
    """WM+HRL baseline 재학습 (K=10, H=15, 3 seeds)."""
    from wm_upper_v2 import WMUpperTrainerV2
    import wm_upper_v2

    wm_upper_v2.ENV_CFG = dict(ENV_CFG)
    wm_upper_v2.K = 10

    for seed in seeds:
        name = f"wm_s{seed}"
        out_dir = os.path.join(RESULTS_DIR, "wm_baseline", name)
        if os.path.exists(os.path.join(out_dir, "checkpoint_final.pt")):
            print(f"[SKIP] {name} — 이미 완료됨")
            continue

        print(f"\n{'='*60}")
        print(f"[WM baseline] seed={seed}, out={out_dir}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(WM_CFG)
        trainer = WMUpperTrainerV2(cfg=cfg, device=device)
        trainer.train(out_dir=out_dir, seed=seed,
                      use_wandb=use_wandb, run_name=name)
        print(f"[WM baseline] {name} 완료!")


def run_ppo(seeds, use_wandb=False, device="cuda"):
    """PPO+HRL 재학습 (3 seeds)."""
    import wm_upper_v2
    from ppo_hrl_upper import PPOHRLTrainer

    wm_upper_v2.ENV_CFG = dict(ENV_CFG)
    wm_upper_v2.K = 10  # 기본값 보장

    for seed in seeds:
        name = f"ppo_hrl_s{seed}"
        out_dir = os.path.join(RESULTS_DIR, "ppo", name)
        if os.path.exists(os.path.join(out_dir, "ppo_hrl_final.pt")):
            print(f"[SKIP] {name} — 이미 완료됨")
            continue

        print(f"\n{'='*60}")
        print(f"[PPO] seed={seed}, out={out_dir}")
        print(f"{'='*60}")

        cfg = dict(PPO_CFG)
        cfg["state_dim"]           = WM_CFG["tssm"]["state_dim"]
        cfg["n_action_agents"]     = WM_CFG["n_action_agents"]
        cfg["n_action_targets"]    = WM_CFG["n_action_targets"]
        cfg["n_action_attackers"]  = WM_CFG["n_action_attackers"]
        cfg["n_action_ally_zones"] = WM_CFG["n_action_ally_zones"]

        trainer = PPOHRLTrainer(cfg=cfg, device=device)
        trainer.train(out_dir=out_dir, seed=seed,
                      use_wandb=use_wandb, run_name=name)
        print(f"[PPO] {name} 완료!")


def run_ablation_k(seeds, use_wandb=False, device="cuda"):
    """Ablation: K 변화 (K=1, 5, 20). K=10은 기존 baseline."""
    from wm_upper_v2 import WMUpperTrainerV2
    import config as C_mod

    for k_val in [1, 5, 20]:
        for seed in seeds:
            name = f"wm_K{k_val}_s{seed}"
            out_dir = os.path.join(RESULTS_DIR, "ablation_k", name)
            if os.path.exists(os.path.join(out_dir, "checkpoint_final.pt")):
                print(f"[SKIP] {name} — 이미 완료됨")
                continue

            print(f"\n{'='*60}")
            print(f"[Ablation-K] K={k_val}, seed={seed}")
            print(f"{'='*60}")

            # K 오버라이드
            C_mod.K = k_val
            cfg = copy.deepcopy(WM_CFG)
            env_cfg = copy.deepcopy(ENV_CFG)
            env_cfg["high_level_interval"] = k_val

            # K가 작으면 상상 지평 실질적 범위가 넓어지므로 설정은 유지
            # K=1이면 H=15가 15 env steps, K=20이면 300 env steps

            import wm_upper_v2
            wm_upper_v2.ENV_CFG = env_cfg
            wm_upper_v2.K = k_val

            trainer = WMUpperTrainerV2(cfg=cfg, device=device)
            trainer.train(out_dir=out_dir, seed=seed,
                          use_wandb=use_wandb, run_name=name)
            print(f"[Ablation-K] {name} 완료!")

    # K 복원
    C_mod.K = 10
    wm_upper_v2.K = 10


def run_ablation_h(seeds, use_wandb=False, device="cuda"):
    """Ablation: H 변화 (H=5, 10, 25). H=15는 기존 baseline."""
    from wm_upper_v2 import WMUpperTrainerV2

    for h_val in [5, 10, 25]:
        for seed in seeds:
            name = f"wm_H{h_val}_s{seed}"
            out_dir = os.path.join(RESULTS_DIR, "ablation_h", name)
            if os.path.exists(os.path.join(out_dir, "checkpoint_final.pt")):
                print(f"[SKIP] {name} — 이미 완료됨")
                continue

            print(f"\n{'='*60}")
            print(f"[Ablation-H] H={h_val}, seed={seed}")
            print(f"{'='*60}")

            cfg = copy.deepcopy(WM_CFG)
            cfg["tssm"]["imagination_horizon"] = h_val

            import wm_upper_v2
            wm_upper_v2.ENV_CFG = dict(ENV_CFG)

            trainer = WMUpperTrainerV2(cfg=cfg, device=device)
            trainer.train(out_dir=out_dir, seed=seed,
                          use_wandb=use_wandb, run_name=name)
            print(f"[Ablation-H] {name} 완료!")


def run_ablation_entity_encoder(seeds, use_wandb=False, device="cuda"):
    """Ablation: Entity Encoder 제거 (use_entity_encoder=False)."""
    from wm_upper_v2 import WMUpperTrainerV2

    for seed in seeds:
        name = f"wm_noEE_s{seed}"
        out_dir = os.path.join(RESULTS_DIR, "ablation_ee", name)
        if os.path.exists(os.path.join(out_dir, "checkpoint_final.pt")):
            print(f"[SKIP] {name} — 이미 완료됨")
            continue

        print(f"\n{'='*60}")
        print(f"[Ablation-EE] Entity Encoder OFF, seed={seed}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(WM_CFG)
        cfg["tssm"]["use_entity_encoder"] = False

        import wm_upper_v2
        wm_upper_v2.ENV_CFG = dict(ENV_CFG)

        trainer = WMUpperTrainerV2(cfg=cfg, device=device)
        trainer.train(out_dir=out_dir, seed=seed,
                      use_wandb=use_wandb, run_name=name)
        print(f"[Ablation-EE] {name} 완료!")


def parse_args():
    p = argparse.ArgumentParser(description="리비전 실험 일괄 실행")
    p.add_argument("--exp", type=str, default="all",
                   choices=["wm_baseline", "ppo", "ablation_k", "ablation_h", "ablation_ee", "all"],
                   help="실행할 실험 종류")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="사용할 random seeds (기본: 42 43 44)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()
    exp = args.exp

    # 실험 순서: baseline 먼저, 그다음 ablation
    if exp in ("all", "wm_baseline"):
        run_wm_baseline(args.seeds, args.wandb, args.device)

    if exp in ("all", "ppo"):
        run_ppo(args.seeds, args.wandb, args.device)

    if exp in ("all", "ablation_k"):
        run_ablation_k(args.seeds, args.wandb, args.device)

    if exp in ("all", "ablation_h"):
        run_ablation_h(args.seeds, args.wandb, args.device)

    if exp in ("all", "ablation_ee"):
        run_ablation_entity_encoder(args.seeds, args.wandb, args.device)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"전체 실험 완료! 총 소요시간: {elapsed/3600:.1f}시간")
    print(f"결과 위치: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
