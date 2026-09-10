"""
models/tssm.py
TransDreamer 기반 TSSM (TransDreamer Stochastic State Model).

멀티에이전트 전투 최적화 수정 사항:
  수정 1 — Entity-Structured Input Encoding
      120-dim 플랫 벡터 → 26개 엔티티 임베딩 (EntityEncoder)
      엔티티 유형별 인코더 + 타입 임베딩 + attention pool

  수정 2 — Alive-Mask
      dead 엔티티 (alive=0) 를 attention 에서 자동으로 마스킹
      동적으로 변하는 에이전트 수 처리

  수정 3 — K-Step Consistent Imagination
      imagine() 에서 K=20 스텝마다 정책 재실행 (서브골 일관성)
      하위 에이전트 실행 주기와 일치

  수정 4 — Decomposed Reward Head
      향후 성분별 보상 헤드 확장 가능한 구조
      현재: 통합 보상 예측 (단일 헤드)

  수정 5 — RSA Latent Strategy Variable  [Phase 3 예정]
      적군 전략 잠재 변수 추가 예정 위치 표시

RSSM 호환 인터페이스 (교체 가능):
    model.forward(obs_seq, action_seq, done_seq, reward_seq) -> TSSMLoss
    model.imagine(start_h, start_z, policy, horizon)         -> ImagineOutput
    model.get_state_features(h, z)                           -> Tensor
    model.encode_obs(obs, h)                                 -> (h, z)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import (
    MLP, EntityEncoder, EntityAttentionPool,
    CausalTransformer, N_ENTITIES,
)


# ── 설정 ─────────────────────────────────────────────────────────────────

@dataclass
class TSSMConfig:
    """TSSM 하이퍼파라미터."""

    # 입출력 공간
    state_dim:  int = 120   # DroneEnv 글로벌 상태 차원 (대칭, padding 없음)
    action_dim: int = 16    # 8 agents × 2 coords (서브골)

    # 확률적 상태 (이산 범주형, straight-through)
    n_categoricals: int = 8   # 범주형 변수 수
    n_classes:      int = 4   # 각 변수의 클래스 수
    # → stoch_dim = n_categoricals × n_classes = 32

    # 결정론적 상태 (Transformer 출력 차원)
    deter_size: int = 512

    # Entity Encoder
    entity_embed_dim: int = 64    # 엔티티당 임베딩 차원
    entity_pool_heads: int = 4    # EntityAttentionPool 헤드 수
    entity_layout: Optional[list] = None   # None=networks 기본(8×8). N·M 스윕 시 유닛수 기반 레이아웃 주입
    use_entity_encoder: bool = True  # False=Pure TSSM (obs_mlp 사용)

    # Dynamics Transformer
    n_heads:  int   = 8
    n_layers: int   = 4
    dropout:  float = 0.0

    # MLP hidden (서브모듈 공용)
    mlp_hidden: int = 256

    # 학습 손실 가중치
    kl_free:      float = 1.0    # KL free bits (nat)
    kl_scale:     float = 1.0    # KL 손실 가중치
    kl_balance:   Optional[float] = 0.8   # DreamerV2 KL balancing (None=원본 대칭 KL, ablation용)
    recon_scale:  float = 1.0    # 재구성 손실 가중치
    reward_scale: float = 1.0    # 보상 예측 손실 가중치
    cont_scale:   float = 1.0    # continue 예측 손실 가중치

    # SymAEC (Symmetric Action-Effect Contrastive)
    aec_beta:      float = 0.0    # AEC 손실 가중치 (0=비활성, >0 활성)
    aec_embed_dim: int   = 64     # φ/ψ 임베딩 차원
    aec_mode:      str   = "sym"  # "sym"=대칭, "a2dh"=action→Δh만, "dh2a"=Δh→action만

    # Imagination / RL
    gamma:               float = 0.99
    lambda_:             float = 0.95
    imagination_horizon: int   = 15
    high_level_interval: int   = 20   # K: 상위 에이전트 실행 주기

    @property
    def stoch_dim(self) -> int:
        return self.n_categoricals * self.n_classes   # 32

    @property
    def feat_dim(self) -> int:
        """정책/가치 네트워크 입력 차원 = h + z."""
        return self.deter_size + self.stoch_dim       # 544


# ── 내부 상태 컨테이너 ────────────────────────────────────────────────────

class TSSMState(NamedTuple):
    """TSSM 잠재 상태 쌍."""
    h: torch.Tensor   # (B, deter_size) — 결정론적 상태
    z: torch.Tensor   # (B, stoch_dim)  — 확률적 상태 (one-hot 펼침)


# ── 학습 출력 컨테이너 ────────────────────────────────────────────────────

@dataclass
class TSSMLoss:
    total:     torch.Tensor
    recon:     torch.Tensor    # 관측 재구성 (MSE)
    reward:    torch.Tensor    # 보상 예측 (MSE)
    continue_: torch.Tensor    # continue 예측 (BCE)
    kl:        torch.Tensor    # KL(posterior || prior)
    aec:       torch.Tensor    # SymAEC 대조손실 (0 if disabled)
    metrics:   Dict[str, float] = field(default_factory=dict)


# ── Imagination 출력 컨테이너 ─────────────────────────────────────────────

@dataclass
class ImagineOutput:
    states:    torch.Tensor   # (B, H, feat_dim)   — 잠재 특성 시퀀스
    rewards:   torch.Tensor   # (B, H)              — 예측 보상
    values:    torch.Tensor   # (B, H)              — 예측 가치
    continues: torch.Tensor   # (B, H)              — continue 확률
    actions:   torch.Tensor   # (B, H, action_dim)  — 실행 서브골


# ── 서브모듈 ──────────────────────────────────────────────────────────────

class RepresentationModel(nn.Module):
    """Posterior q(z_t | o_t, h_t).

    수정 1+2: 엔티티 인코딩 + alive-masked attention pool.
    """

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        E = cfg.entity_embed_dim
        self.use_entity = cfg.use_entity_encoder

        if self.use_entity:
            self.entity_encoder = EntityEncoder(embed_dim=E, entity_layout=cfg.entity_layout)
            self.pool = EntityAttentionPool(E, n_heads=cfg.entity_pool_heads)
            proj_in = E + cfg.deter_size
        else:
            # Pure TSSM: obs → MLP → embed
            self.obs_mlp = MLP(cfg.state_dim, E, hidden_dim=cfg.mlp_hidden, n_layers=2)
            proj_in = E + cfg.deter_size

        # context + h → posterior logits
        self.proj = MLP(
            proj_in,
            cfg.n_categoricals * cfg.n_classes,
            hidden_dim=cfg.mlp_hidden,
            n_layers=2,
        )
        self.cfg = cfg

    def forward(
        self,
        obs: torch.Tensor,   # (B, state_dim)
        h:   torch.Tensor,   # (B, deter_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:     (B, n_categoricals, n_classes)
            alive_mask: (B, N_entities) or None
        """
        B = obs.shape[0]
        C = self.cfg

        if self.use_entity:
            embeds, alive_mask = self.entity_encoder(obs)
            ctx = self.pool(embeds, alive_mask)
        else:
            ctx = self.obs_mlp(obs)
            alive_mask = None

        logits = self.proj(torch.cat([ctx, h], dim=-1))
        logits = logits.reshape(B, C.n_categoricals, C.n_classes)
        return logits, alive_mask


class TransitionModel(nn.Module):
    """Prior p(z_t | h_t)."""

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = MLP(
            cfg.deter_size,
            cfg.n_categoricals * cfg.n_classes,
            hidden_dim=cfg.mlp_hidden,
            n_layers=2,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, deter_size)
        Returns:
            logits: (B, n_categoricals, n_classes)
        """
        B = h.shape[0]
        C = self.cfg
        return self.proj(h).reshape(B, C.n_categoricals, C.n_classes)


class DynamicsModel(nn.Module):
    """결정론적 상태 h_t 계산 — Causal Transformer over (z, a) pairs.

    입력: (z_{0:T-1}, a_{0:T-1}) → 출력: h_{0:T-1}
    h_t 는 t 이전의 (z, a) 히스토리를 요약.
    """

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        token_dim = cfg.stoch_dim + cfg.action_dim

        self.token_proj  = nn.Linear(token_dim, cfg.deter_size)
        self.transformer = CausalTransformer(
            d_model  = cfg.deter_size,
            n_heads  = cfg.n_heads,
            n_layers = cfg.n_layers,
            dropout  = cfg.dropout,
        )

    def forward(
        self,
        z_seq: torch.Tensor,              # (B, T, stoch_dim)
        a_seq: torch.Tensor,              # (B, T, action_dim)
        pad_mask: Optional[torch.Tensor] = None,  # (B, T) True=패딩
    ) -> torch.Tensor:                    # (B, T, deter_size)
        tokens = torch.cat([z_seq, a_seq], dim=-1)   # (B, T, stoch+action)
        x = self.token_proj(tokens)                   # (B, T, deter_size)
        return self.transformer(x, key_padding_mask=pad_mask)


class ObsDecoder(nn.Module):
    """재구성 헤드: (h, z) → obs 예측."""

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        self.net = MLP(cfg.feat_dim, cfg.state_dim,
                       hidden_dim=cfg.mlp_hidden, n_layers=2, out_act=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class RewardHead(nn.Module):
    """보상 예측 헤드: (h, z) → scalar reward.

    수정 4 기반: 향후 성분별 보상으로 분해 가능한 구조.
    현재는 단일 통합 보상 예측.
    """

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        self.net = MLP(cfg.feat_dim, 1,
                       hidden_dim=cfg.mlp_hidden, n_layers=2, out_act=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns: (..., 1)."""
        return self.net(feat)


class ContinueHead(nn.Module):
    """에피소드 continue 예측: (h, z) → Bernoulli logit."""

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        self.net = MLP(cfg.feat_dim, 1,
                       hidden_dim=cfg.mlp_hidden, n_layers=1, out_act=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns: (..., 1) logit."""
        return self.net(feat)


class SymAECHead(nn.Module):
    """Symmetric Action-Effect Contrastive (CLIP-style).

    I(a; Δh)의 하한을 양방향 InfoNCE로 최대화.
    - φ(a): 행동 임베딩, ψ(Δh): 결정론적 상태변화 임베딩
    - L2 정규화 + 코사인 유사도 + 학습가능 온도 τ
    - 대칭: L = ½(CE_row + CE_col)

    v2: Δz(one-hot, {-1,0,1} 이산) → Δh(256-dim 연속)로 변경.
        Δz는 categorical straight-through 차분이라 패턴이 제한적 →
        contrastive 학습 불가(loss≈log N 고정). h는 dynamics 출력이고
        action이 직접 입력되므로 I(a;Δh) 최대화가 "dynamics가 행동에
        반응하라"를 가장 직접적으로 강제.
    """

    def __init__(self, action_dim: int, deter_size: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(action_dim, embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.psi = nn.Sequential(
            nn.Linear(deter_size, embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # CLIP-style learnable log-temperature
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        actions: torch.Tensor,   # (N, action_dim)
        delta_h: torch.Tensor,   # (N, deter_size)
        mode:    str = "sym",    # "sym", "a2dh", "dh2a"
    ) -> torch.Tensor:
        """Returns scalar SymAEC loss."""
        a_emb  = F.normalize(self.phi(actions), dim=-1)   # (N, E)
        dh_emb = F.normalize(self.psi(delta_h), dim=-1)   # (N, E)

        tau = torch.clamp(self.log_tau.exp(), min=0.01, max=100.0)
        logits = a_emb @ dh_emb.T / tau   # (N, N)

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_a2dh = F.cross_entropy(logits,   labels)   # 행: a→Δh
        loss_dh2a = F.cross_entropy(logits.T, labels)   # 열: Δh→a
        if mode == "a2dh":
            return loss_a2dh
        elif mode == "dh2a":
            return loss_dh2a
        else:  # "sym"
            return 0.5 * (loss_a2dh + loss_dh2a)


class ValueHead(nn.Module):
    """V_lambda 예측 헤드: (h, z) → scalar value."""

    def __init__(self, cfg: TSSMConfig) -> None:
        super().__init__()
        self.net = MLP(cfg.feat_dim, 1,
                       hidden_dim=cfg.mlp_hidden, n_layers=2, out_act=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns: (..., 1)."""
        return self.net(feat)


# ── TSSM 메인 ─────────────────────────────────────────────────────────────

class TSSM(nn.Module):
    """TransDreamer Stochastic State Model (멀티에이전트 전투 최적화 버전).

    수정 요약:
      1. Entity-Structured Input + 2. Alive-Mask → RepresentationModel
      3. K-Step Consistent Imagination           → imagine() 메서드
      4. Decomposed Reward Head 구조              → RewardHead (확장 예정)
      5. RSA Latent Strategy Variable            → Phase 3 예정

    사용법:
        cfg   = TSSMConfig()
        model = TSSM(cfg).to(device)

        # World Model 학습
        loss = model(obs_seq, action_seq, done_seq, reward_seq)
        loss.total.backward()

        # Imagination rollout
        result = model.imagine(h, z, actor_network, horizon=15)
        returns = model.compute_lambda_returns(result.rewards, result.values, result.continues)
    """

    def __init__(self, cfg: Optional[TSSMConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TSSMConfig()
        C = self.cfg

        self.repr_model  = RepresentationModel(C)
        self.trans_model = TransitionModel(C)
        self.dyn_model   = DynamicsModel(C)
        self.obs_decoder = ObsDecoder(C)
        self.reward_head = RewardHead(C)
        self.cont_head   = ContinueHead(C)
        self.value_head  = ValueHead(C)

        # SymAEC head (beta>0일 때만 실제 사용, entity encoder 사용 시만 생성)
        if C.use_entity_encoder:
            self.aec_head = SymAECHead(C.action_dim, C.deter_size, C.aec_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 유틸 ─────────────────────────────────────────────────────────────

    def get_state_features(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """(h, z) → 정책/가치 네트워크 입력 특성 벡터."""
        return torch.cat([h, z], dim=-1)

    @staticmethod
    def sample_straight_through(
        logits: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """이산 범주형 straight-through gradient 샘플링.

        sample=True: 범주형 분포에서 *샘플링* (DreamerV2 표준, stochastic latent).
        sample=False: argmax(mode) — 결정론적 eval/ablation용 (원본 동작).
        gradient는 두 경우 모두 probs를 통과(straight-through)하므로 동일.

        Args:
            logits: (..., n_cat, n_cls)
        Returns:
            z: (..., n_cat * n_cls) — one-hot 펼침
        """
        probs = F.softmax(logits, dim=-1)
        if sample:
            idx = torch.distributions.Categorical(probs=probs).sample()
        else:
            idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=logits.shape[-1]).float()
        # Straight-through: forward=hard, backward=soft
        z_st = (hard - probs).detach() + probs
        return z_st.flatten(start_dim=-2)

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """DreamerV3 symlog: 보상 분포 압축 (스파이크성 보상 MSE 학습용)."""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """symlog 역변환."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    @staticmethod
    def kl_categorical(
        post_logits:  torch.Tensor,   # (B, T, n_cat, n_cls) or (B, n_cat, n_cls)
        prior_logits: torch.Tensor,
        free_bits:    float = 1.0,
        kl_balance:   Optional[float] = 0.8,
    ) -> torch.Tensor:
        """KL(posterior || prior) with free bits + DreamerV2 KL balancing.

        free_bits:  KL이 이 값 이하면 floor 처리 (초기 학습 안정화).
        kl_balance: posterior collapse 방지. prior를 posterior 쪽으로 더 빠르게(=balance),
                    posterior를 prior 쪽으로 덜 끌어당김. None이면 원본 대칭 KL(ablation용).
        Returns scalar.
        """
        post_dist  = torch.distributions.OneHotCategorical(logits=post_logits)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)

        if kl_balance is None:
            # ── 원본: 대칭 KL (balancing 없음, 하위호환/ablation) ──
            kl = torch.distributions.kl_divergence(post_dist, prior_dist)  # (..., n_cat)
            kl = kl.sum(dim=-1)                                            # over cats
            return torch.clamp(kl, min=free_bits).mean()

        # ── DreamerV2 KL balancing ──
        # 한쪽은 posterior를 stop-grad → prior만 학습(posterior로 끌려감)
        # 다른쪽은 prior를 stop-grad → posterior만 학습
        post_sg  = torch.distributions.OneHotCategorical(logits=post_logits.detach())
        prior_sg = torch.distributions.OneHotCategorical(logits=prior_logits.detach())
        kl_prior = torch.distributions.kl_divergence(post_sg, prior_dist).sum(dim=-1)  # prior 학습
        kl_post  = torch.distributions.kl_divergence(post_dist, prior_sg).sum(dim=-1)  # posterior 학습
        kl_prior = torch.clamp(kl_prior, min=free_bits)
        kl_post  = torch.clamp(kl_post,  min=free_bits)
        return kl_balance * kl_prior.mean() + (1.0 - kl_balance) * kl_post.mean()

    # ── World Model 순전파 ────────────────────────────────────────────────

    def forward(
        self,
        obs_seq:    torch.Tensor,   # (B, T, state_dim)
        action_seq: torch.Tensor,   # (B, T, action_dim)
        done_seq:   torch.Tensor,   # (B, T) bool
        reward_seq: torch.Tensor,   # (B, T) float
    ) -> TSSMLoss:
        """World Model 학습 순전파 (teacher-forcing 2-pass).

        알고리즘:
          Pass 1: h_seq = Dynamics(z=0, a_shifted) → 초기 h 추정
          Pass 2: post_z = RepModel(obs, h_seq) → posterior 샘플링
                  h_seq  = Dynamics(z_post_shifted, a_shifted) → 정교한 h
                  posterior, prior 재계산 → 손실 계산
        """
        B, T, _ = obs_seq.shape
        C       = self.cfg
        device  = obs_seq.device

        z_zeros = torch.zeros(B, 1, C.stoch_dim,  device=device)
        a_zeros = torch.zeros(B, 1, C.action_dim, device=device)

        # 액션 시퀀스 우측 시프트: a_{t-1} → h_t 입력
        # t=0: a_zero, t=1: a_0, ..., t=T-1: a_{T-2}
        a_shifted = torch.cat([a_zeros, action_seq[:, :-1, :]], dim=1)   # (B, T, action_dim)

        # ── Pass 1: 초기 h 추정 (z=0 사용) ────────────────────────────────
        z_shifted_v1 = z_zeros.expand(B, T, -1)   # (B, T, stoch_dim) all-zeros
        h_seq_v1 = self.dyn_model(z_shifted_v1, a_shifted)   # (B, T, deter_size)

        # ── Pass 1 Posterior: 초기 h로 posterior 계산 ──────────────────────
        obs_flat  = obs_seq.reshape(B * T, -1)
        h_flat_v1 = h_seq_v1.reshape(B * T, -1)
        post_logits_flat, _ = self.repr_model(obs_flat, h_flat_v1)
        post_logits_v1 = post_logits_flat.reshape(B, T, C.n_categoricals, C.n_classes)
        z_post_v1 = self.sample_straight_through(
            post_logits_v1.reshape(B * T, C.n_categoricals, C.n_classes)
        ).reshape(B, T, C.stoch_dim)

        # ── Pass 2: 정교한 h 추정 (posterior z 사용) ──────────────────────
        z_post_shifted = torch.cat([z_zeros, z_post_v1[:, :-1, :]], dim=1)
        h_seq = self.dyn_model(z_post_shifted, a_shifted)   # (B, T, deter_size)

        # ── Pass 2 Posterior & Prior ────────────────────────────────────────
        h_flat = h_seq.reshape(B * T, -1)
        post_logits_flat2, _ = self.repr_model(obs_flat, h_flat)
        post_logits = post_logits_flat2.reshape(B, T, C.n_categoricals, C.n_classes)
        z_post = self.sample_straight_through(
            post_logits.reshape(B * T, C.n_categoricals, C.n_classes)
        ).reshape(B, T, C.stoch_dim)

        prior_logits = self.trans_model(h_flat).reshape(B, T, C.n_categoricals, C.n_classes)

        # ── 손실 계산 ────────────────────────────────────────────────────────
        feat = torch.cat([h_seq, z_post], dim=-1)   # (B, T, feat_dim)
        feat_flat = feat.reshape(B * T, -1)

        # 재구성 손실 (MSE)
        obs_pred = self.obs_decoder(feat_flat).reshape(B, T, -1)
        recon_loss = F.mse_loss(obs_pred, obs_seq)

        # 보상 예측 손실 (MSE in symlog space — 스파이크성 보상 학습)
        reward_pred = self.reward_head(feat_flat).reshape(B, T)
        reward_loss = F.mse_loss(reward_pred, self.symlog(reward_seq))

        # Continue 예측 손실 (BCE)
        cont_pred   = self.cont_head(feat_flat).reshape(B, T)
        cont_target = (~done_seq).float()
        cont_loss   = F.binary_cross_entropy_with_logits(cont_pred, cont_target)

        # KL 손실 (free bits + KL balancing)
        kl_loss = self.kl_categorical(
            post_logits, prior_logits,
            free_bits=C.kl_free, kl_balance=C.kl_balance,
        )

        # SymAEC 손실: I(a_t; Δh_t) 하한 최대화
        if C.aec_beta > 0 and T > 1:
            # Δh_t = h[t+1] - h[t], paired with action_seq[t]
            # h는 dynamics 출력(256-dim 연속) → contrastive 학습 가능
            delta_h = h_seq[:, 1:, :] - h_seq[:, :-1, :]       # (B, T-1, deter_size)
            aec_actions = action_seq[:, :T-1, :]               # (B, T-1, action_dim)
            # flatten to contrastive batch
            aec_loss = self.aec_head(
                aec_actions.reshape(-1, C.action_dim),
                delta_h.reshape(-1, C.deter_size),
                mode=C.aec_mode,
            )
        else:
            aec_loss = torch.tensor(0.0, device=device)

        total = (
            C.recon_scale  * recon_loss
            + C.reward_scale * reward_loss
            + C.cont_scale   * cont_loss
            + C.kl_scale     * kl_loss
            + C.aec_beta     * aec_loss
        )

        return TSSMLoss(
            total     = total,
            recon     = recon_loss,
            reward    = reward_loss,
            continue_ = cont_loss,
            kl        = kl_loss,
            aec       = aec_loss,
            metrics   = {
                "total":    total.item(),
                "recon":    recon_loss.item(),
                "reward":   reward_loss.item(),
                "continue": cont_loss.item(),
                "kl":       kl_loss.item(),
                "aec":      aec_loss.item() if isinstance(aec_loss, torch.Tensor) else 0.0,
            },
        )

    # ── Imagination Rollout ───────────────────────────────────────────────

    def imagine(
        self,
        start_h:  torch.Tensor,             # (B, deter_size)
        start_z:  torch.Tensor,             # (B, stoch_dim)
        policy:   nn.Module,                # (B, feat_dim) → (B, action_dim)
        horizon:  Optional[int] = None,
        seed_z:   Optional[torch.Tensor] = None,  # (B, T_seed, stoch_dim) 실제 trajectory context
        seed_a:   Optional[torch.Tensor] = None,  # (B, T_seed, action_dim)
    ) -> ImagineOutput:
        """Fix 1: Growing-Context Imagination Rollout.

        Transformer는 단일 토큰이 아닌 전체 history를 봐야 훈련과 일치.
        seed_z/seed_a로 실제 배치 context를 전달하면 h 품질이 크게 향상됨.
        매 스텝 ctx에 (z, a)를 추가하여 growing context로 h 계산.
        """
        C = self.cfg
        H = horizon or C.imagination_horizon
        K = C.high_level_interval

        B      = start_h.shape[0]
        device = start_h.device

        z = start_z.clone()
        current_action: Optional[torch.Tensor] = None

        # 시퀀스 컨텍스트 초기화 (seed 있으면 실제 trajectory로 시작)
        if seed_z is not None and seed_a is not None:
            ctx_z = seed_z   # (B, T_seed, stoch_dim)
            ctx_a = seed_a   # (B, T_seed, action_dim)
        else:
            ctx_z = torch.empty(B, 0, C.stoch_dim,  device=device)
            ctx_a = torch.empty(B, 0, C.action_dim, device=device)

        all_feats:    list[torch.Tensor] = []
        all_rewards:  list[torch.Tensor] = []
        all_values:   list[torch.Tensor] = []
        all_conts:    list[torch.Tensor] = []
        all_actions:  list[torch.Tensor] = []

        for t in range(H):
            # h_t: Transformer가 전체 context 활용 → 훈련과 일치하는 h 품질
            if ctx_z.shape[1] > 0:
                h = self.dyn_model(ctx_z, ctx_a)[:, -1, :]  # (B, deter_size)
            else:
                h = start_h.clone()  # seed 없을 때 fallback

            feat = self.get_state_features(h, z)   # (B, feat_dim)

            # K 스텝마다 정책 재실행 (서브골 갱신)
            if t % K == 0:
                current_action = policy(feat)   # (B, action_dim), with grad

            assert current_action is not None

            reward = self.reward_head(feat).squeeze(-1)   # symlog 공간 그대로 (bounded ≤7)
            value  = self.value_head(feat).squeeze(-1)
            cont   = torch.sigmoid(self.cont_head(feat).squeeze(-1))

            all_feats.append(feat)
            all_rewards.append(reward)
            all_values.append(value)
            all_conts.append(cont)
            all_actions.append(current_action)

            # ctx에 현재 (z, a) 추가 → 다음 스텝의 h 계산에 활용
            ctx_z = torch.cat([ctx_z, z.unsqueeze(1)], dim=1)
            ctx_a = torch.cat([ctx_a, current_action.unsqueeze(1)], dim=1)

            # 다음 z: prior 샘플링
            prior_logits = self.trans_model(h)
            z = self.sample_straight_through(prior_logits)

        return ImagineOutput(
            states    = torch.stack(all_feats,   dim=1),   # (B, H, feat_dim)
            rewards   = torch.stack(all_rewards, dim=1),   # (B, H)
            values    = torch.stack(all_values,  dim=1),   # (B, H)
            continues = torch.stack(all_conts,   dim=1),   # (B, H)
            actions   = torch.stack(all_actions, dim=1),   # (B, H, action_dim)
        )

    def compute_lambda_returns(
        self,
        rewards:   torch.Tensor,   # (B, H)
        values:    torch.Tensor,   # (B, H)
        continues: torch.Tensor,   # (B, H)
    ) -> torch.Tensor:             # (B, H)
        """V_lambda 리턴 계산 (역방향 누적).

        V_lambda_t = r_t + gamma * cont_t * [
            (1 - lambda) * V_t+1  +  lambda * V_lambda_t+1
        ]
        """
        C = self.cfg
        H = rewards.shape[1]

        lambda_returns = torch.zeros_like(rewards)
        next_val = values[:, -1]   # bootstrap from last step

        for t in reversed(range(H)):
            if t == H - 1:
                lambda_returns[:, t] = rewards[:, t] + C.gamma * continues[:, t] * next_val
            else:
                next_ret = lambda_returns[:, t + 1]
                lambda_returns[:, t] = (
                    rewards[:, t]
                    + C.gamma * continues[:, t] * (
                        (1 - C.lambda_) * values[:, t + 1]
                        + C.lambda_ * next_ret
                    )
                )
        return lambda_returns

    # ── 온라인 추론 (단일 스텝) ───────────────────────────────────────────

    def encode_obs(
        self,
        obs: torch.Tensor,                    # (B, state_dim)
        h:   Optional[torch.Tensor] = None,   # (B, deter_size)
    ) -> TSSMState:
        """단일 관측 → TSSMState (온라인 추론 / 초기화 용).

        Args:
            obs: 현재 관측
            h:   이전 결정론적 상태 (None이면 zero 초기화)
        Returns:
            TSSMState(h, z) — 현재 잠재 상태
        """
        B      = obs.shape[0]
        C      = self.cfg
        device = obs.device

        if h is None:
            h = torch.zeros(B, C.deter_size, device=device)

        with torch.no_grad():
            post_logits, _ = self.repr_model(obs, h)
            z = self.sample_straight_through(post_logits)   # (B, stoch_dim)
        return TSSMState(h=h, z=z)

    def step_dynamics(
        self,
        state:  TSSMState,          # 현재 잠재 상태
        action: torch.Tensor,       # (B, action_dim)
    ) -> TSSMState:
        """한 스텝 dynamics 전이 (온라인 추론 용).

        Returns:
            다음 TSSMState
        """
        z_in = state.z.unsqueeze(1)    # (B, 1, stoch_dim)
        a_in = action.unsqueeze(1)     # (B, 1, action_dim)
        h_next = self.dyn_model(z_in, a_in).squeeze(1)    # (B, deter_size)

        with torch.no_grad():
            prior_logits = self.trans_model(h_next)
            z_next = self.sample_straight_through(prior_logits)
        return TSSMState(h=h_next, z=z_next)
