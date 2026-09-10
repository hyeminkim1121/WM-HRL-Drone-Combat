"""plot_revision.py — 실험 결과 그래프 생성.

사용법:
  python plot_revision.py                    # 모든 그래프 생성
  python plot_revision.py --only learning    # 학습 곡선만
  python plot_revision.py --only ablation_k  # K ablation만
  python plot_revision.py --only ablation_h  # H ablation만
  python plot_revision.py --only ablation_ee # Entity Encoder ablation만
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 스타일 설정
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_results")
FIGURE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_figures")

COLORS = {
    "WM+HRL":  "#2196F3",   # blue
    "PPO+HRL": "#FF5722",   # red-orange
}
ABLATION_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


def load_log_csvs(pattern: str) -> list[pd.DataFrame]:
    """glob 패턴에 매칭되는 log.csv 파일들을 로드."""
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    return dfs


def load_episode_csvs(pattern: str) -> list[pd.DataFrame]:
    """glob 패턴에 매칭되는 episodes.csv 파일들을 로드."""
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    return dfs


def smooth(y, window=100):
    """이동 평균 스무딩."""
    if len(y) < window:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().values


def aggregate_seeds(dfs: list[pd.DataFrame], x_col: str, y_col: str, smooth_window=100):
    """여러 시드의 데이터를 공통 x축으로 보간 후 평균/표준편차 계산."""
    if not dfs:
        return None, None, None

    # 모든 시드에서 공통 x 범위 결정
    x_min = max(df[x_col].min() for df in dfs)
    x_max = min(df[x_col].max() for df in dfs)
    x_common = np.linspace(x_min, x_max, 500)

    interpolated = []
    for df in dfs:
        y_smooth = smooth(df[y_col].values, smooth_window)
        y_interp = np.interp(x_common, df[x_col].values, y_smooth)
        interpolated.append(y_interp)

    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    std  = interpolated.std(axis=0)
    return x_common, mean, std


def plot_learning_curves():
    """Fig: WM+HRL vs PPO+HRL 학습 곡선 (win_rate, reward)."""
    print("[Plot] Learning Curves (WM+HRL vs PPO+HRL)")

    # WM+HRL episodes.csv 로드
    wm_dfs = load_episode_csvs(os.path.join(RESULTS_DIR, "wm_baseline", "wm_*", "episodes.csv"))
    # PPO+HRL episodes.csv 로드
    ppo_dfs = load_episode_csvs(os.path.join(RESULTS_DIR, "ppo", "ppo_*", "episodes.csv"))

    if not wm_dfs and not ppo_dfs:
        print("  [SKIP] 데이터 없음")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, dfs, color in [("WM+HRL", wm_dfs, COLORS["WM+HRL"]),
                                ("PPO+HRL", ppo_dfs, COLORS["PPO+HRL"])]:
        if not dfs:
            print(f"  [SKIP] {label} 데이터 없음")
            continue

        # 에피소드별 win (1/0) → 롤링 win_rate
        for ax_idx, (y_col_raw, ylabel) in enumerate([("win_rate", "Win Rate"), ("reward", "Episode Reward")]):
            processed_dfs = []
            for df in dfs:
                df_proc = pd.DataFrame()
                df_proc["episode"] = df["episode"]
                if y_col_raw == "win_rate":
                    df_proc["y"] = (df["result"] == "win").astype(float).rolling(100, min_periods=1).mean()
                else:
                    df_proc["y"] = df["reward"].rolling(100, min_periods=1).mean()
                processed_dfs.append(df_proc)

            x, mean, std = aggregate_seeds(
                [pd.DataFrame({"episode": d["episode"], "y": d["y"]}).rename(columns={"y": "y", "episode": "episode"})
                 for d in processed_dfs],
                "episode", "y", smooth_window=1  # 이미 rolling
            )
            if x is None:
                continue

            ax = axes[ax_idx]
            ax.plot(x, mean, color=color, label=label, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))

    axes[0].set_title("Win Rate vs Mirror RSA")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("Episode Reward")

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "fig_learning_curves.png")
    plt.savefig(out)
    plt.savefig(out.replace(".png", ".pdf"))
    print(f"  -> {out}")
    plt.close()


def plot_ppo_training_details():
    """Fig: PPO 학습 상세 곡선 (actor_loss, critic_loss, entropy, clip_frac)."""
    print("[Plot] PPO Training Details")

    ppo_logs = load_log_csvs(os.path.join(RESULTS_DIR, "ppo", "ppo_*", "log.csv"))
    if not ppo_logs:
        print("  [SKIP] PPO log 데이터 없음")
        return

    metrics = [
        ("win_rate",    "Win Rate"),
        ("ep_reward",   "Episode Reward"),
        ("actor_loss",  "Actor Loss"),
        ("critic_loss", "Critic Loss"),
        ("entropy",     "Entropy"),
        ("clip_frac",   "Clip Fraction"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        for j, df in enumerate(ppo_logs):
            if col not in df.columns:
                continue
            ax.plot(df["episode"], df[col], alpha=0.7, label=f"seed {j}", linewidth=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))

    plt.suptitle("PPO+HRL Training Details", fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "fig_ppo_details.png")
    plt.savefig(out)
    plt.savefig(out.replace(".png", ".pdf"))
    print(f"  -> {out}")
    plt.close()


def _plot_ablation(title, dir_pattern, labels, fig_name, x_col="episode"):
    """Ablation 공통 플롯 함수. episodes.csv 기반."""
    print(f"[Plot] {title}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    has_data = False

    for i, (pattern, label, color) in enumerate(zip(dir_pattern, labels, ABLATION_COLORS)):
        dfs = load_episode_csvs(pattern)
        if not dfs:
            print(f"  [SKIP] {label} 데이터 없음")
            continue
        has_data = True

        processed_dfs = []
        for df in dfs:
            df_proc = pd.DataFrame()
            df_proc["episode"] = df["episode"]
            df_proc["win_rate"] = (df["result"] == "win").astype(float).rolling(100, min_periods=1).mean()
            df_proc["reward"]   = df["reward"].rolling(100, min_periods=1).mean()
            processed_dfs.append(df_proc)

        for ax_idx, (y_name, ylabel) in enumerate([("win_rate", "Win Rate"), ("reward", "Episode Reward")]):
            cols = [pd.DataFrame({"episode": d["episode"], "y": d[y_name]}) for d in processed_dfs]
            x, mean, std = aggregate_seeds(cols, "episode", "y", smooth_window=1)
            if x is None:
                continue
            ax = axes[ax_idx]
            ax.plot(x, mean, color=color, label=label, linewidth=1.5)
            if std is not None and len(dfs) > 1:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    if not has_data:
        plt.close()
        return

    for ax_idx, (ylabel, ylim) in enumerate([("Win Rate", (-0.05, 1.05)), ("Episode Reward", None)]):
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel(ylabel)
        axes[ax_idx].legend()
        if ylim:
            axes[ax_idx].set_ylim(ylim)
        axes[ax_idx].xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, fig_name)
    plt.savefig(out)
    plt.savefig(out.replace(".png", ".pdf"))
    print(f"  -> {out}")
    plt.close()


def plot_ablation_k():
    """K 변화 ablation 그래프."""
    patterns = [
        os.path.join(RESULTS_DIR, "ablation_k", "wm_K1_*", "episodes.csv"),
        os.path.join(RESULTS_DIR, "ablation_k", "wm_K5_*", "episodes.csv"),
        os.path.join(RESULTS_DIR, "wm_baseline", "wm_*", "episodes.csv"),  # K=10 baseline
        os.path.join(RESULTS_DIR, "ablation_k", "wm_K20_*", "episodes.csv"),
    ]
    labels = ["K=1", "K=5", "K=10 (baseline)", "K=20"]
    _plot_ablation("Ablation: Macro-Action Interval K", patterns, labels, "fig_ablation_k.png")


def plot_ablation_h():
    """H 변화 ablation 그래프."""
    patterns = [
        os.path.join(RESULTS_DIR, "ablation_h", "wm_H5_*", "episodes.csv"),
        os.path.join(RESULTS_DIR, "ablation_h", "wm_H10_*", "episodes.csv"),
        os.path.join(RESULTS_DIR, "wm_baseline", "wm_*", "episodes.csv"),  # H=15 baseline
        os.path.join(RESULTS_DIR, "ablation_h", "wm_H25_*", "episodes.csv"),
    ]
    labels = ["H=5", "H=10", "H=15 (baseline)", "H=25"]
    _plot_ablation("Ablation: Imagination Horizon H", patterns, labels, "fig_ablation_h.png")


def plot_ablation_ee():
    """Entity Encoder 제거 ablation 그래프."""
    patterns = [
        os.path.join(RESULTS_DIR, "wm_baseline", "wm_*", "episodes.csv"),  # with EE
        os.path.join(RESULTS_DIR, "ablation_ee", "wm_noEE_*", "episodes.csv"),
    ]
    labels = ["With Entity Encoder", "Without Entity Encoder (MLP)"]
    _plot_ablation("Ablation: Entity Encoder", patterns, labels, "fig_ablation_ee.png")


def plot_sample_efficiency():
    """Fig: 환경 step 기준 샘플 효율성 비교 (WM+HRL vs PPO+HRL)."""
    print("[Plot] Sample Efficiency (env_step basis)")

    wm_dfs  = load_episode_csvs(os.path.join(RESULTS_DIR, "wm_baseline", "wm_*", "episodes.csv"))
    ppo_dfs = load_episode_csvs(os.path.join(RESULTS_DIR, "ppo", "ppo_*", "episodes.csv"))

    if not wm_dfs and not ppo_dfs:
        print("  [SKIP] 데이터 없음")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, dfs, color in [("WM+HRL", wm_dfs, COLORS["WM+HRL"]),
                                ("PPO+HRL", ppo_dfs, COLORS["PPO+HRL"])]:
        if not dfs:
            continue

        processed = []
        for df in dfs:
            df_proc = pd.DataFrame()
            # env_step 누적 (env_length의 cumsum)
            if "env_step" in df.columns:
                df_proc["env_step"] = df["env_step"]
            else:
                df_proc["env_step"] = df["env_length"].cumsum()
            df_proc["win_rate"] = (df["result"] == "win").astype(float).rolling(100, min_periods=1).mean()
            df_proc["reward"]   = df["reward"].rolling(100, min_periods=1).mean()
            processed.append(df_proc)

        for ax_idx, (y_col, ylabel) in enumerate([("win_rate", "Win Rate"), ("reward", "Episode Reward")]):
            cols = [pd.DataFrame({"env_step": d["env_step"], "y": d[y_col]}) for d in processed]
            x, mean, std = aggregate_seeds(cols, "env_step", "y", smooth_window=1)
            if x is None:
                continue
            ax = axes[ax_idx]
            ax.plot(x, mean, color=color, label=label, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    axes[0].set_title("Win Rate (by Env Steps)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("Reward (by Env Steps)")

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "fig_sample_efficiency.png")
    plt.savefig(out)
    plt.savefig(out.replace(".png", ".pdf"))
    print(f"  -> {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        choices=["learning", "ppo_details", "ablation_k", "ablation_h",
                                 "ablation_ee", "sample_eff"])
    args = parser.parse_args()

    os.makedirs(FIGURE_DIR, exist_ok=True)

    if args.only is None or args.only == "learning":
        plot_learning_curves()
    if args.only is None or args.only == "ppo_details":
        plot_ppo_training_details()
    if args.only is None or args.only == "sample_eff":
        plot_sample_efficiency()
    if args.only is None or args.only == "ablation_k":
        plot_ablation_k()
    if args.only is None or args.only == "ablation_h":
        plot_ablation_h()
    if args.only is None or args.only == "ablation_ee":
        plot_ablation_ee()

    print(f"\n그래프 저장 위치: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
