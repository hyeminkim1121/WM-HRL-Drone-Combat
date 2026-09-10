# WM+HRL: World Model-based Hierarchical Reinforcement Learning for Multi-Drone Combat Simulation

계층적 구조 기반 World Model 강화학습을 통한 다중 드론 전투 시뮬레이션에 관한 연구

## Requirements

- Python 3.11+
- PyTorch 2.6.0+ (CUDA 12.4)
- NumPy, Pandas, Matplotlib

```bash
pip install torch numpy pandas matplotlib wandb
```

## Project Structure

```
code/
├── config.py              # Hyperparameters (Table A1)
├── train_wm.py            # WM+HRL training entry point
├── train_ppo_hrl.py       # PPO+HRL baseline training entry point
├── wm_upper_v2.py         # WM+HRL trainer (core)
├── ppo_hrl_upper.py       # PPO+HRL trainer
├── lower_agent.py         # Rule-based lower controller
├── eval_vs_rsa.py         # vs Mirror RSA evaluation
├── eval_head_to_head.py   # Head-to-Head evaluation
├── head2head.py           # H2H utilities
├── run_revision_experiments.py  # Ablation experiment runner
├── plot_revision.py       # Figure generation
├── models/
│   ├── tssm.py            # TSSM World Model
│   └── networks.py        # MLP, EntityEncoder, CausalTransformer
└── env/
    ├── drone_env.py        # Multi-drone combat environment
    ├── map.py              # Battlefield map (obstacles)
    ├── units.py            # Drone/Base unit definitions
    └── reinforcement.py    # Enemy reinforcement (optional)
```

## Training

### WM+HRL (Proposed Method)
```bash
python train_wm.py --seed 42 --name wm_s42
python train_wm.py --seed 43 --name wm_s43
python train_wm.py --seed 44 --name wm_s44
```

### PPO+HRL (Baseline)
```bash
python train_ppo_hrl.py --seed 42 --name ppo_s42
python train_ppo_hrl.py --seed 43 --name ppo_s43
python train_ppo_hrl.py --seed 44 --name ppo_s44
```

### Ablation Experiments
```bash
# K variation (macro-action interval)
python run_revision_experiments.py --exp ablation_k --seeds 42

# H variation (imagination horizon)
python run_revision_experiments.py --exp ablation_h --seeds 42

# Entity Encoder removal
python run_revision_experiments.py --exp ablation_ee --seeds 42
```

## Evaluation

### vs Mirror RSA
```bash
python eval_vs_rsa.py --checkpoint results/wm_s42/checkpoint_final.pt
```

### Head-to-Head (WM vs PPO)
```bash
python eval_head_to_head.py \
    --wm_checkpoint results/wm_s42/checkpoint_final.pt \
    --ppo_checkpoint results/ppo_s42/ppo_hrl_final.pt
```

## Key Hyperparameters (Table A1)

| Component | Value |
|-----------|-------|
| World Model lr | 3×10⁻⁴ |
| Actor/Critic lr | 1×10⁻⁴ |
| Discount γ | 0.99 |
| λ-return weight | 0.95 |
| Entropy coef βₑₙₜ | 1×10⁻³ |
| KL free bits | 0.1 |
| Slow critic EMA τ | 0.02 |
| Imagination horizon H | 15 macro steps |
| K-step macro-action | 10 |
| Batch size | 64 sequences |
| Replay buffer | 5×10⁴ macro steps |
| WM warmup (Phase 1) | 15K episodes |
| Total training | 100K episodes |
| Hardware | NVIDIA RTX 4060 Ti 16GB |

## Environment

- Grid: 100×100, altitude z ∈ [0, 20]
- Each side: 5 attack drones + 3 defense drones + 5 bases
- Mirror symmetric setup
- Action: 8 agents × 8 targets = 64-dim discrete macro-action
- Observation: 120-dim global state
- Reward: event-driven (base destruction) + time penalty

## Citation

```
김혜민, "계층적 구조 기반 World Model 강화학습을 통한 다중 드론 전투 시뮬레이션에 관한 연구",
한국시뮬레이션학회논문지, 2026.
```
