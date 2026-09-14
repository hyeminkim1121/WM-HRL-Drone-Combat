"""run_ablation_fast.py — 30K 에피소드 ablation (resume 지원).

완료 판정: wm_v2_final.pt 존재.
중간 체크포인트가 있으면 가장 최근 것에서 resume.
"""
import copy
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WM_CFG, ENV_CFG
import wm_upper_v2
from wm_upper_v2 import WMUpperTrainerV2
import config as C_mod

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_results")
TOTAL_STEPS = 30_000
WARMUP_STEPS = 5_000
SEEDS = [42, 43, 44]


def is_done(out_dir):
    return os.path.exists(os.path.join(out_dir, "wm_v2_final.pt"))


def find_latest_checkpoint(out_dir):
    """가장 최근 체크포인트 .pt 파일 경로 반환. 없으면 None."""
    pts = sorted(glob.glob(os.path.join(out_dir, "wm_v2_step*.pt")))
    return pts[-1] if pts else None


def run_one(name, cfg_override, k_override=None, out_subdir="ablation", seed=42):
    out_dir = os.path.join(RESULTS_DIR, out_subdir, name)
    if is_done(out_dir):
        print(f"[SKIP] {name} — 이미 완료")
        return

    cfg = copy.deepcopy(WM_CFG)
    cfg["total_steps"] = TOTAL_STEPS
    cfg["wm_warmup_steps"] = WARMUP_STEPS
    cfg["save_every"] = 5_000
    for k, v in cfg_override.items():
        if k == "tssm":
            cfg["tssm"].update(v)
        else:
            cfg[k] = v

    env_cfg = copy.deepcopy(ENV_CFG)
    if k_override is not None:
        C_mod.K = k_override
        wm_upper_v2.K = k_override
        env_cfg["high_level_interval"] = k_override
    else:
        C_mod.K = 10
        wm_upper_v2.K = 10

    wm_upper_v2.ENV_CFG = env_cfg

    # Resume 체크
    resume_path = find_latest_checkpoint(out_dir)

    print(f"\n{'='*60}")
    print(f"[ABLATION] {name} (total={TOTAL_STEPS}, seed={seed})")
    if resume_path:
        print(f"[RESUME] {os.path.basename(resume_path)}")
    print(f"{'='*60}")

    trainer = WMUpperTrainerV2(cfg=cfg, device="cuda")
    if resume_path:
        trainer.load(resume_path)
        print(f"  Resumed from step {trainer.step}")

    trainer.train(out_dir=out_dir, seed=seed, use_wandb=False, run_name=name)
    print(f"[DONE] {name}")

    C_mod.K = 10
    wm_upper_v2.K = 10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # K ablation (3 seeds)
    for k_val in [10, 1, 5, 20]:
        for seed in SEEDS:
            run_one(f"wm_K{k_val}_s{seed}", {}, k_override=k_val,
                    out_subdir="ablation_k", seed=seed)

    # H ablation (3 seeds)
    for h_val in [15, 5, 10, 25]:
        for seed in SEEDS:
            run_one(f"wm_H{h_val}_s{seed}", {"tssm": {"imagination_horizon": h_val}},
                    out_subdir="ablation_h", seed=seed)

    # EE ablation (seed 42 only)
    run_one("wm_noEE_s42", {"tssm": {"use_entity_encoder": False}},
            out_subdir="ablation_ee", seed=42)
    run_one("wm_withEE_s42", {},
            out_subdir="ablation_ee", seed=42)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"전체 ablation 완료! 소요: {elapsed/3600:.1f}시간")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
