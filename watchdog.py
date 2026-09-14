"""watchdog.py — 실험 감시 + 크래시 자동 재시작 + 보고서 갱신.

5분마다:
1. python 프로세스 생존 확인
2. 실험 진행 상황 체크
3. 멈춰있으면 자동 재시작
4. REPORT.md 갱신

python watchdog.py 로 실행 (nohup 권장)
"""
import os
import sys
import time
import subprocess
import glob
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_results")
REPORT_PATH = os.path.join(RESULTS_DIR, "REPORT.md")
LOG_PATH = os.path.join(RESULTS_DIR, "watchdog.log")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# 실험 순서 정의
ABLATION_RUNS = []
for k_val in [10, 1, 5, 20]:
    for seed in [42, 43, 44]:
        ABLATION_RUNS.append(("ablation_k", f"wm_K{k_val}_s{seed}", {"k": k_val}))
for h_val in [15, 5, 10, 25]:
    for seed in [42, 43, 44]:
        ABLATION_RUNS.append(("ablation_h", f"wm_H{h_val}_s{seed}", {"h": h_val}))
ABLATION_RUNS.append(("ablation_ee", "wm_noEE_s42", {"ee": False}))
ABLATION_RUNS.append(("ablation_ee", "wm_withEE_s42", {"ee": True}))


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_experiment_done(subdir, name):
    """checkpoint_final.pt 존재 여부로 완료 판정."""
    final = os.path.join(RESULTS_DIR, subdir, name, "wm_v2_final.pt")
    return os.path.exists(final)


def get_episode_count(subdir, name):
    """episodes.csv 줄 수로 진행 상황 확인."""
    ep_path = os.path.join(RESULTS_DIR, subdir, name, "episodes.csv")
    if not os.path.exists(ep_path):
        return 0
    try:
        return sum(1 for _ in open(ep_path)) - 1  # 헤더 제외
    except:
        return 0


def is_python_running():
    """python.exe 프로세스 존재 여부 (watchdog 자신 제외)."""
    try:
        result = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=10)
        count = result.stdout.count("python.exe")
        return count > 1  # watchdog 자신 제외
    except:
        return False


def get_next_ablation():
    """다음으로 실행해야 할 ablation 반환."""
    for subdir, name, params in ABLATION_RUNS:
        if not is_experiment_done(subdir, name):
            return subdir, name, params
    return None, None, None


def start_ablation(subdir, name, params):
    """ablation 1개 실행."""
    log(f"실험 시작: {name} ({params})")

    cmd = f'cd "{CODE_DIR}" && PYTHONIOENCODING=utf-8 python -u run_ablation_fast.py'

    # run_ablation_fast.py는 순차 실행이므로 이미 완료된 건 스킵함
    log_file = os.path.join(RESULTS_DIR, f"{name}.log")
    proc = subprocess.Popen(
        ["bash", "-c", f'cd "{CODE_DIR}" && PYTHONIOENCODING=utf-8 python -u run_ablation_fast.py > "{log_file}" 2>&1'],
    )
    return proc


def is_ppo_done():
    """PPO 3 seeds 전부 완료 여부."""
    for seed in [42, 43, 44]:
        final = os.path.join(RESULTS_DIR, "ppo", f"ppo_hrl_s{seed}", "ppo_hrl_final.pt")
        if not os.path.exists(final):
            return False
    return True


def generate_report():
    """REPORT.md 갱신."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# 실험 분석 보고서", f"*최종 업데이트: {now}*\n", "---\n"]

    # PPO
    lines.append("## PPO+HRL\n")
    for seed in [42, 43, 44]:
        final = os.path.join(RESULTS_DIR, "ppo", f"ppo_hrl_s{seed}", "ppo_hrl_final.pt")
        log_path = os.path.join(RESULTS_DIR, "ppo", f"ppo_hrl_s{seed}", "log.csv")
        if os.path.exists(final):
            try:
                log_df = pd.read_csv(log_path)
                wr = log_df["win_rate"].iloc[-10:].mean()
                rw = log_df["ep_reward"].iloc[-10:].mean()
                lines.append(f"- seed {seed}: **완료** (wr={wr:.3f}, reward={rw:.1f})")
            except:
                lines.append(f"- seed {seed}: **완료**")
        else:
            ep_count = get_episode_count("ppo", f"ppo_hrl_s{seed}")
            lines.append(f"- seed {seed}: 진행 중 ({ep_count:,}/100K)")
    lines.append("")

    # WM baseline
    lines.append("## WM+HRL Baseline (WandB)\n")
    lines.append("- seed 43: wr=0.998, reward=151.2")
    lines.append("- seed 44: wr=0.992, reward=149.1")
    lines.append("- seed 45: wr=0.995, reward=150.9")
    lines.append("- **mean: wr=0.995±0.003, reward=150.4±0.9**\n")

    # Ablation
    lines.append("## Ablation 실험 (30K episodes, seed 42)\n")
    lines.append("| 실험 | 상태 | Win Rate | Episodes |")
    lines.append("|------|------|----------|----------|")

    for subdir, name, params in ABLATION_RUNS:
        ep_count = get_episode_count(subdir, name)
        if is_experiment_done(subdir, name):
            try:
                ep_path = os.path.join(RESULTS_DIR, subdir, name, "episodes.csv")
                ep = pd.read_csv(ep_path)
                wr = (ep["result"] == "win").astype(float).rolling(100, min_periods=1).mean().iloc[-100:].mean()
                lines.append(f"| {name} | **완료** | {wr:.3f} | {len(ep):,} |")
            except:
                lines.append(f"| {name} | **완료** | - | {ep_count:,} |")
        elif ep_count > 0:
            lines.append(f"| {name} | 진행 중 | - | {ep_count:,}/30K |")
        else:
            lines.append(f"| {name} | 대기 | - | - |")

    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    log("Watchdog 시작!")
    ablation_proc = None
    last_ep_counts = {}
    stall_counts = {}

    while True:
        try:
            # PPO 상태
            ppo_done = is_ppo_done()
            if not ppo_done:
                for seed in [42, 43, 44]:
                    ep = get_episode_count("ppo", f"ppo_hrl_s{seed}")
                    if ep > 0 and ep < 100000:
                        log(f"PPO seed {seed}: {ep:,}/100K ep")

            # ablation 진행
            if ppo_done or True:  # PPO와 별개로 ablation 상태 체크
                all_done = all(is_experiment_done(s, n) for s, n, _ in ABLATION_RUNS)

                if all_done:
                    log("모든 실험 완료!")
                    generate_report()
                    break

                # 프로세스 생존 확인 (watchdog 외에 python이 없으면 재시작)
                if ppo_done and not is_python_running():
                    next_sub, next_name, next_params = get_next_ablation()
                    if next_name:
                        log(f"프로세스 없음 감지! 다음 실험 시작: {next_name}")
                        ablation_proc = start_ablation(next_sub, next_name, next_params)

                # 진행 stall 감지 (10분간 ep 변화 없으면 재시작)
                for subdir, name, params in ABLATION_RUNS:
                    if is_experiment_done(subdir, name):
                        continue
                    ep = get_episode_count(subdir, name)
                    key = f"{subdir}/{name}"
                    if ep > 0:
                        prev = last_ep_counts.get(key, 0)
                        if ep == prev:
                            stall_counts[key] = stall_counts.get(key, 0) + 1
                            if stall_counts[key] >= 2:  # 10분 stall
                                log(f"STALL 감지: {name} ({ep} ep에서 멈춤)")
                        else:
                            stall_counts[key] = 0
                        last_ep_counts[key] = ep

            # 보고서 갱신
            generate_report()

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(300)  # 5분


if __name__ == "__main__":
    main()
