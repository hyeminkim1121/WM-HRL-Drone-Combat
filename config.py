"""config.py — 한국시뮬레이션학회 논문용 설정.

논문: "계층적 구조 기반 World Model 강화학습을 통한 다중 드론 전투 시뮬레이션에 관한 연구"
스펙: h ∈ R^256, z ∈ R^16 (4 categoricals × 4 classes), feat ∈ R^272
      8 agents × 8 targets = 64-dim action, K=10, H=15
      Phase 1: 10K episodes (WM warmup), Total: 100K episodes

NOTE: 0622/ (SCI용) 코드와 독립. 이 파일은 한국시뮬레이션학회 리비전 대응 전용.
"""

K = 10
WANDB_PROJECT = "KSIM-WM-HRL"
WANDB_ENTITY  = None

# 논문 Table A1 / Table 2 에 기재된 하이퍼파라미터와 1:1 대응
ENV_CFG = {
    "max_episode_steps":   500,
    "high_level_interval": K,
    "n_ally_attack":  5,
    "n_ally_defend":  3,
    "n_ally_base":    5,
    "n_enemy_attack": 5,
    "n_enemy_defend": 3,
    "n_enemy_base":   5,
    "ally_base_hp":   130,
    "enemy_atk_hp":   50,
    "enemy_atk_range": 3,
    "enemy_atk_power": 15,
    "enemy_def_hp":   40,
    "enemy_def_range": 2,
    "enemy_def_power": 10,
    "reinforce_enabled": False,
    # 보상 (Table 1)
    "R_STEP":         -0.001,
    "R_KILL_DRONE":    0.0,
    "R_KILL_BASE":    20.0,
    "R_BASE_SHARED":   5.0,
    "R_DRONE_KILLED":  0.0,
    "R_BASE_LOST":    -5.0,
    "R_WIN":          50.0,
    "R_LOSE":        -10.0,
    "R_DAMAGE_DRONE":  0.0,
    "R_DAMAGE_BASE":   0.0,
    "enemy_strategy": "mirror",
}

# 논문 스펙 상수
N_AGENTS   = 8    # 공격 5 + 방어 3
N_TARGETS  = 8    # 적 기지 5 + 적 방어드론 3
STATE_DIM  = 120  # (8+8)×5 + (5+5)×4
ACTION_DIM = 64   # 8 × 8

WM_CFG = {
    "tssm": {
        "state_dim":      STATE_DIM,
        "action_dim":     ACTION_DIM,
        "entity_layout": [
            ("ally_drone",  8, 5),
            ("ally_base",   5, 4),
            ("enemy_drone", 8, 5),
            ("enemy_base",  5, 4),
        ],
        # 논문 3.2절: h ∈ R^256, z ∈ R^16
        "n_categoricals": 4,
        "n_classes":      4,
        "deter_size":     256,
        # Entity encoder
        "entity_embed_dim":  32,
        "entity_pool_heads": 4,
        "use_entity_encoder": True,
        # Dynamics Transformer
        "n_heads":   4,
        "n_layers":  2,
        "dropout":   0.0,
        "mlp_hidden": 256,
        # 손실 가중치
        "kl_free":      0.1,       # Table A1
        "kl_scale":     1.0,
        "kl_balance":   0.8,
        "recon_scale":  1.0,
        "reward_scale": 1.0,
        "cont_scale":   1.0,
        # AEC 비활성 (한국시뮬레이션학회 논문에서 미사용)
        "aec_beta":      0.0,
        "aec_embed_dim": 64,
        # RL
        "gamma":               0.99,   # Table A1
        "lambda_":             0.95,   # Table A1
        "imagination_horizon": 15,     # Table A1
        "high_level_interval": 1,      # imagination 내 K (macro 단위이므로 1)
    },
    "n_action_agents":    N_AGENTS,
    "n_action_targets":   N_TARGETS,
    "n_action_attackers": 5,
    "n_action_ally_zones": 5,
    # 학습 스케줄 (논문 3.4절, Table A1)
    "total_steps":       100_000,      # 100K episodes
    "wm_warmup_steps":   10_000,       # Phase 1: 10K episodes (심사용 원본 기준)
    "seq_len":           25,
    "batch_size":        64,           # Table A1
    "wm_lr":             3e-4,         # Table A1
    "ac_lr":             1e-4,         # Table A1
    "entropy_coef":      1e-3,         # Table A1: β_ent = 1×10^-3
    "slow_critic_tau":   0.02,         # Table A1
    "grad_clip":         10.0,
    "eval_every":        5_000,
    "save_every":        5_000,
    "log_every":         100,
    "buffer_max_size":   50_000,       # Table A1: 5×10^4 macro steps
}

# ── make_configs 호환 함수 (train_ppo_hrl.py, head2head.py 등에서 호출) ──
# 한국시뮬레이션학회 논문은 8×8 고정이므로 인자를 무시하고 고정 설정 반환
def make_configs(n_ally_attack=5, n_ally_defend=3, n_ally_base=5,
                 n_enemy_attack=5, n_enemy_defend=3, n_enemy_base=5,
                 base_hp=130, model_size="small"):
    """0622/ make_configs 호환. 논문 고정 설정 반환."""
    return dict(ENV_CFG), dict(WM_CFG)


# PPO baseline 설정 (논문 5.1절: "PPO 알고리즘의 표준 하이퍼파라미터")
PPO_CFG = {
    "total_episodes":  100_000,
    "lr":              3e-4,
    "gamma":           0.99,
    "lambda_gae":      0.95,
    "clip_eps":        0.2,
    "entropy_coef":    0.01,
    "value_coef":      0.5,
    "max_grad_norm":   0.5,
    "n_epochs":        4,
    "batch_size":      64,
    "hidden_dim":      256,
}
