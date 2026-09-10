"""train_wm.py — WM+HRL 학습 엔트리포인트 (한국시뮬레이션학회 논문용).

실행:
  python train_wm.py --seed 42 --name wm_s0 --wandb
  python train_wm.py --seed 43 --name wm_s1 --wandb
  python train_wm.py --seed 44 --name wm_s2 --wandb
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WM_CFG
from wm_upper_v2 import WMUpperTrainerV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--name",    type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--device",  type=str, default="cuda")
    p.add_argument("--wandb",   action="store_true")
    p.add_argument("--resume",  type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.name or f"wm_s{args.seed}"
    out = os.path.join(args.out_dir, run_name)

    trainer = WMUpperTrainerV2(cfg=WM_CFG, device=args.device)
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} (step={trainer.step})")

    trainer.train(
        out_dir=out,
        seed=args.seed,
        use_wandb=args.wandb,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
