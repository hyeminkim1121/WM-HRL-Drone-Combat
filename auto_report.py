"""auto_report.py — 실험 완료 시 자동 분석 보고서 생성.

pipeline.log에서 호출되거나, 독립 실행 가능:
  python auto_report.py          # 현재 완료된 실험 전체 보고
  python auto_report.py --watch  # 5분마다 체크하며 새 완료 감지 시 보고 추가
"""
import argparse
import os
import time
import glob
import numpy as np
import pandas as pd
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_results")
REPORT_PATH = os.path.join(RESULTS_DIR, "REPORT.md")


def analyze_ppo():
    """PPO 재학습 분석."""
    lines = []
    lines.append("## PPO+HRL 재학습 결과\n")

    seeds_done = []
    for seed in [42, 43, 44]:
        ep_path = os.path.join(RESULTS_DIR, "ppo", f"ppo_hrl_s{seed}", "episodes.csv")
        log_path = os.path.join(RESULTS_DIR, "ppo", f"ppo_hrl_s{seed}", "log.csv")
        if not os.path.exists(ep_path):
            continue
        ep = pd.read_csv(ep_path)
        if len(ep) < 100000:
            lines.append(f"- seed {seed}: **진행 중** ({len(ep):,} / 100K ep)\n")
            continue

        seeds_done.append(seed)
        log = pd.read_csv(log_path)
        final_wr = log["win_rate"].iloc[-10:].mean()
        final_rw = log["ep_reward"].iloc[-10:].mean()

        # 수렴 시점
        wr_series = (ep["result"] == "win").astype(float).rolling(100, min_periods=1).mean()
        first_95 = wr_series[wr_series >= 0.95].index[0] if (wr_series >= 0.95).any() else "N/A"

        lines.append(f"### seed {seed}")
        lines.append(f"- 최종 Win Rate: **{final_wr:.3f}**")
        lines.append(f"- 최종 Reward: **{final_rw:.1f}**")
        lines.append(f"- 처음 wr≥0.95 도달: ep {first_95}")
        lines.append(f"- 총 에피소드: {len(ep):,}")
        lines.append("")

    if len(seeds_done) >= 2:
        final_wrs = []
        for seed in seeds_done:
            log = pd.read_csv(os.path.join(RESULTS_DIR, "ppo", f"ppo_hrl_s{seed}", "log.csv"))
            final_wrs.append(log["win_rate"].iloc[-10:].mean())
        lines.append(f"### 종합 (완료된 {len(seeds_done)} seeds)")
        lines.append(f"- Win Rate: mean={np.mean(final_wrs):.3f} ± {np.std(final_wrs):.3f}")
        lines.append(f"- Seeds: {[f'{w:.3f}' for w in final_wrs]}")
        lines.append("")

    return "\n".join(lines)


def analyze_wm_baseline():
    """WM+HRL baseline (WandB 데이터) 분석."""
    lines = []
    lines.append("## WM+HRL Baseline (WandB 데이터)\n")

    final_wrs, final_rws = [], []
    for seed in [43, 44, 45]:
        path = os.path.join(RESULTS_DIR, "wm_baseline", f"wm_s{seed}", "wandb_history.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        wr = df["win_rate"].dropna()
        rw = df["episode/reward"].dropna()
        final_wrs.append(wr.iloc[-100:].mean())
        final_rws.append(rw.iloc[-100:].mean())

        converge_step = df.loc[wr[wr >= 0.95].index[0], "_step"] if (wr >= 0.95).any() else "N/A"
        lines.append(f"### seed {seed}")
        lines.append(f"- 최종 Win Rate: **{wr.iloc[-100:].mean():.3f}**")
        lines.append(f"- 최종 Reward: **{rw.iloc[-100:].mean():.1f}**")
        lines.append(f"- 처음 wr≥0.95 도달: step {converge_step}")
        lines.append("")

    if final_wrs:
        lines.append(f"### 종합 ({len(final_wrs)} seeds)")
        lines.append(f"- Win Rate: mean={np.mean(final_wrs):.3f} ± {np.std(final_wrs):.3f}")
        lines.append(f"- Reward: mean={np.mean(final_rws):.1f} ± {np.std(final_rws):.1f}")
        lines.append("")

    return "\n".join(lines)


def analyze_ablation(exp_name, dir_name, param_name, values, baseline_label=None):
    """Ablation 실험 분석."""
    lines = []
    lines.append(f"## Ablation: {exp_name}\n")

    results = []
    for val in values:
        if dir_name == "ablation_ee":
            pattern = os.path.join(RESULTS_DIR, dir_name, f"wm_noEE_s*", "episodes.csv")
        else:
            pattern = os.path.join(RESULTS_DIR, dir_name, f"wm_{param_name}{val}_s*", "episodes.csv")

        files = sorted(glob.glob(pattern))
        if not files:
            lines.append(f"- {param_name}={val}: 아직 완료되지 않음")
            continue

        for f in files:
            ep = pd.read_csv(f)
            seed = f.split("_s")[-1].split("/")[0].split("\\")[0]
            if len(ep) < 50000:  # 절반도 안 됐으면 진행 중
                lines.append(f"- {param_name}={val} (seed {seed}): **진행 중** ({len(ep):,} ep)")
                continue

            wr_series = (ep["result"] == "win").astype(float).rolling(100, min_periods=1).mean()
            final_wr = wr_series.iloc[-100:].mean()
            final_rw = ep["reward"].rolling(100, min_periods=1).mean().iloc[-100:].mean()
            first_90 = wr_series[wr_series >= 0.90].index[0] if (wr_series >= 0.90).any() else "N/A"

            label = f"{param_name}={val}" if dir_name != "ablation_ee" else f"noEE"
            results.append({"label": label, "seed": seed, "wr": final_wr, "rw": final_rw, "first_90": first_90, "total_ep": len(ep)})
            lines.append(f"### {label} (seed {seed})")
            lines.append(f"- 최종 Win Rate: **{final_wr:.3f}**")
            lines.append(f"- 최종 Reward: **{final_rw:.1f}**")
            lines.append(f"- 처음 wr≥0.90 도달: ep {first_90}")
            lines.append(f"- 총 에피소드: {len(ep):,}")
            lines.append("")

    if results:
        lines.append("### 요약 Table")
        lines.append(f"| {param_name} | Win Rate | Reward | 첫 wr≥0.90 |")
        lines.append("|---|---|---|---|")
        for r in results:
            lines.append(f"| {r['label']} | {r['wr']:.3f} | {r['rw']:.1f} | {r['first_90']} |")
        lines.append("")

    return "\n".join(lines)


def generate_report():
    """전체 보고서 생성."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = []
    report.append(f"# 리비전 실험 분석 보고서")
    report.append(f"*최종 업데이트: {now}*\n")
    report.append("---\n")

    # WM baseline
    report.append(analyze_wm_baseline())
    report.append("---\n")

    # PPO
    report.append(analyze_ppo())
    report.append("---\n")

    # Ablation K
    report.append(analyze_ablation("Macro-Action Interval K", "ablation_k", "K", [1, 5, 20]))
    report.append("---\n")

    # Ablation H
    report.append(analyze_ablation("Imagination Horizon H", "ablation_h", "H", [5, 10, 25]))
    report.append("---\n")

    # Ablation EE
    report.append(analyze_ablation("Entity Encoder 제거", "ablation_ee", "noEE", ["noEE"]))

    text = "\n".join(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[{now}] 보고서 저장: {REPORT_PATH}")
    return text


def watch_mode(interval=300):
    """interval초마다 보고서 갱신."""
    print(f"Watch 모드 시작 (매 {interval}초)")
    last_hash = ""
    while True:
        text = generate_report()
        curr_hash = str(hash(text))
        if curr_hash != last_hash:
            print(f"  -> 변경 감지, 보고서 갱신됨")
            last_hash = curr_hash
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="5분마다 자동 갱신")
    parser.add_argument("--interval", type=int, default=300, help="갱신 간격 (초)")
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.interval)
    else:
        print(generate_report())
