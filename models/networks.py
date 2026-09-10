"""
models/networks.py
공통 신경망 빌딩 블록.

포함 내용:
  - MLP            : 다층 퍼셉트론 (LayerNorm + SiLU)
  - EntityEncoder  : 120-dim 글로벌 상태 → 26개 엔티티 임베딩 + alive 마스크
  - TransformerBlock: Pre-LN Transformer 블록 (alive/causal 마스크 지원)
  - CausalTransformer: 인과적 Transformer (TSSM Dynamics 모델용)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Flash Attention (선택적 의존성) ──────────────────────────────────────────
# flash_attn 2.x: A40 / A100 / H100 등 Ampere+ 아키텍처, CUDA 11.6+
# 미설치 시 nn.MultiheadAttention으로 자동 폴백
import os as _os
_FLASH_ATTN_AVAILABLE: bool = False
_flash_attn_func = None
if _os.environ.get("WM_DISABLE_FLASH", "") not in ("", "0", "false", "False"):
    # Turing(RTX 6000) / 일부 Blackwell 등 미지원 GPU에서 강제 비활성 (런타임 크래시 방지)
    print("[networks] WM_DISABLE_FLASH 설정 — flash 비활성, 표준 attention 사용")
else:
    try:
        from flash_attn import flash_attn_func as _fa_func  # flash_attn 2.x API
        _flash_attn_func = _fa_func
        _FLASH_ATTN_AVAILABLE = True
        print("[networks] Flash Attention detected - applied to CausalTransformer")
    except ImportError:
        print("[networks] flash_attn not installed - using standard MultiheadAttention")


# ── 기본 MLP ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """다층 퍼셉트론 (LayerNorm + SiLU 기본값).

    Args:
        in_dim:     입력 차원
        out_dim:    출력 차원
        hidden_dim: 은닉층 차원
        n_layers:   은닉층 수 (0 = 단층 선형)
        act:        활성화 함수 클래스
        norm:       은닉층에 LayerNorm 적용 여부
        out_act:    출력층에 활성화 함수 적용 여부
    """

    def __init__(
        self,
        in_dim:     int,
        out_dim:    int,
        hidden_dim: int  = 256,
        n_layers:   int  = 2,
        act:        type = nn.SiLU,
        norm:       bool = True,
        out_act:    bool = False,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or out_act:
                if norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Entity Encoder ───────────────────────────────────────────────────────
#
# 120-dim 글로벌 상태 레이아웃 (대칭, padding 없음):
#   ally_drone  [  0: 40] → (8, 5): x/G, y/G, z/Z_max, hp/max_hp, alive  (5 atk + 3 def)
#   ally_base   [ 40: 60] → (5, 4): x/G, y/G,          hp/max_hp, alive
#   enemy_drone [ 60:100] → (8, 5): x/G, y/G, z/Z_max, hp/max_hp, alive  (5 atk + 3 def)
#   enemy_base  [100:120] → (5, 4): x/G, y/G,          hp/max_hp, alive
# 총 엔티티 수 = 8 + 5 + 8 + 5 = 26

def make_entity_layout(n_ally_drone=8, n_ally_base=5, n_enemy_drone=8, n_enemy_base=5):
    """유닛 수 → 엔티티 레이아웃. raw_dim: drone=5(x,y,z,hp,alive), base=4(x,y,hp,alive)."""
    return [
        ("ally_drone",   n_ally_drone,  5),
        ("ally_base",    n_ally_base,   4),
        ("enemy_drone",  n_enemy_drone, 5),
        ("enemy_base",   n_enemy_base,  4),
    ]

def state_dim_of(layout) -> int:
    return sum(n * raw for _, n, raw in layout)

# 기본 레이아웃 (8 ally drone[5atk+3def] + 5 base + 8 enemy drone + 5 base = 120-dim, 26 엔티티)
ENTITY_LAYOUT: list[tuple[str, int, int]] = make_entity_layout()
N_ENTITIES: int = sum(n for _, n, _ in ENTITY_LAYOUT)   # 26
STATE_DIM:  int = state_dim_of(ENTITY_LAYOUT)            # 120


class EntityEncoder(nn.Module):
    """120-dim 글로벌 상태 → 26개 엔티티 임베딩 시퀀스.

    각 엔티티의 마지막 특성이 alive 플래그 (0/1).
    dead 엔티티 (alive=0)는 임베딩이 zero 벡터가 되도록 마스킹.
    """

    def __init__(self, embed_dim: int = 64, entity_layout=None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.entity_layout = entity_layout if entity_layout is not None else ENTITY_LAYOUT

        # 엔티티 타입별 인코더 (alive 플래그 제외한 특성 → embed_dim)
        encoders: dict[str, nn.Module] = {}
        for name, _, raw_dim in self.entity_layout:
            feat_dim = raw_dim - 1   # alive 플래그 제외
            encoders[name] = MLP(feat_dim, embed_dim, hidden_dim=embed_dim,
                                 n_layers=1, norm=True)
        self.encoders = nn.ModuleDict(encoders)

        # 타입 임베딩 (엔티티 타입 구분)
        self.type_embed = nn.Embedding(len(self.entity_layout), embed_dim)

    def forward(
        self,
        state: torch.Tensor,   # (B, 120)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embeds:    (B, 26, embed_dim)  — 엔티티 임베딩 시퀀스
            alive_mask:(B, 26)  float      — 1.0=alive, 0.0=dead
        """
        B      = state.shape[0]
        device = state.device

        embed_list: list[torch.Tensor] = []
        mask_list:  list[torch.Tensor] = []

        ptr = 0
        for type_idx, (name, n, raw_dim) in enumerate(self.entity_layout):
            chunk = state[:, ptr: ptr + n * raw_dim].reshape(B, n, raw_dim)
            ptr  += n * raw_dim

            alive = chunk[..., -1].clamp(0.0, 1.0)   # (B, n)
            feats = chunk[..., :-1]                   # (B, n, feat_dim)

            # feat_dim × n 엔티티를 배치 처리: reshape → MLP → reshape
            feats_flat = feats.reshape(B * n, raw_dim - 1)
            emb_flat   = self.encoders[name](feats_flat)         # (B*n, embed_dim)
            emb        = emb_flat.reshape(B, n, self.embed_dim)  # (B, n, embed_dim)

            # 타입 임베딩 추가
            type_idx_t = torch.full((B, n), type_idx,
                                    dtype=torch.long, device=device)
            emb = emb + self.type_embed(type_idx_t)              # (B, n, embed_dim)

            # dead 엔티티 마스킹 (alive=0 → 임베딩 0)
            emb = emb * alive.unsqueeze(-1)

            embed_list.append(emb)
            mask_list.append(alive)

        embeds     = torch.cat(embed_list, dim=1)   # (B, 43, embed_dim)
        alive_mask = torch.cat(mask_list,  dim=1)   # (B, 43)
        return embeds, alive_mask


# ── Transformer Block ────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LN Transformer 블록.

    Flash Attention 우선 사용 (flash_attn 2.x 설치 시):
      - CausalTransformer의 causal self-attention에 최적화
      - key_padding_mask 없는 경우(= CausalTransformer 표준 호출)만 flash 경로 사용
      - key_padding_mask 있는 경우 → F.scaled_dot_product_attention 경로 (SDPA)

    flash_attn 미설치 시:
      - nn.MultiheadAttention으로 자동 폴백

    key_padding_mask: (B, T) — True 위치는 attention에서 무시 (dead 엔티티 등)
    attn_mask:        (T, T) — causal mask (True = 마스킹)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int   = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) % n_heads({n_heads}) != 0"
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.dropout_p = dropout
        self._use_flash = _FLASH_ATTN_AVAILABLE

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if self._use_flash:
            # Flash Attention: Q/K/V 프로젝션 (packed QKV)
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj  = nn.Linear(d_model, d_model,     bias=True)
        else:
            # 표준 Multi-head Attention
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ff_drop = nn.Dropout(dropout)

    def forward(
        self,
        x:                torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        x_norm  = self.norm1(x)

        if self._use_flash:
            # QKV 계산: (B, T, 3*D) → 각각 (B, T, H, head_dim)
            qkv = self.qkv_proj(x_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(B, T, self.n_heads, self.head_dim).contiguous()
            k = k.reshape(B, T, self.n_heads, self.head_dim).contiguous()
            v = v.reshape(B, T, self.n_heads, self.head_dim).contiguous()

            is_causal  = (attn_mask is not None)
            dropout_p  = self.dropout_p if self.training else 0.0

            if key_padding_mask is None:
                # ── Flash Attention 경로 ─────────────────────────────────────
                # flash_attn 2.x 는 fp16 / bf16 필요
                orig_dtype = q.dtype
                if orig_dtype not in (torch.float16, torch.bfloat16):
                    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

                out = _flash_attn_func(
                    q, k, v,
                    dropout_p    = dropout_p,
                    causal       = is_causal,
                    softmax_scale= None,       # 기본값 (1/√head_dim)
                )                              # (B, T, H, head_dim)
                out = out.to(orig_dtype).reshape(B, T, D)
            else:
                # ── SDPA 경로 (key_padding_mask 지원) ────────────────────────
                # F.scaled_dot_product_attention: PyTorch 2.0+, Ampere에서 flash 자동 사용
                q_t = q.transpose(1, 2)   # (B, H, T, head_dim)
                k_t = k.transpose(1, 2)
                v_t = v.transpose(1, 2)

                # attention bias 빌드: causal + padding 마스크 합산
                attn_bias: Optional[torch.Tensor] = None
                if is_causal:
                    # bool → additive: True 위치 = -inf
                    attn_bias = torch.zeros(T, T, dtype=q.dtype, device=x.device)
                    attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))
                    attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
                if key_padding_mask is not None:
                    pad_bias = torch.zeros(
                        B, 1, 1, T, dtype=q.dtype, device=x.device
                    ).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
                    attn_bias = pad_bias if attn_bias is None else attn_bias + pad_bias

                out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t,
                    attn_mask = attn_bias,
                    dropout_p = dropout_p,
                )
                out = out.transpose(1, 2).reshape(B, T, D)

            attn_out = self.out_proj(out)

        else:
            # ── 표준 MultiheadAttention 폴백 ─────────────────────────────────
            attn_out, _ = self.attn(
                x_norm, x_norm, x_norm,
                key_padding_mask = key_padding_mask,
                attn_mask        = attn_mask,
                need_weights     = False,
            )

        # Pre-LN residual
        x = x + attn_out
        x = x + self.ff_drop(self.ff(self.norm2(x)))
        return x


class CausalTransformer(nn.Module):
    """인과적 Transformer 시퀀스 모델.

    TSSM Dynamics Model에서 (z_{0:t-1}, a_{0:t-1}) → h_t 계산에 사용.
    시퀀스 내 미래 정보 차단 (causal attention mask).
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_layers: int,
        dropout:  float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # 포지셔널 인코딩 (최대 seq_len=512 지원)
        self.pos_embed = nn.Embedding(512, d_model)

    def forward(
        self,
        x:                torch.Tensor,              # (B, T, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=마스킹
    ) -> torch.Tensor:                               # (B, T, d_model)
        B, T, _ = x.shape
        device = x.device

        # 포지셔널 임베딩 추가
        pos = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        x   = x + self.pos_embed(pos)

        # Causal attention mask: 미래 위치 차단
        causal = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=causal)
        return self.norm(x)


# ── Entity Transformer ───────────────────────────────────────────────────

class EntityAttentionPool(nn.Module):
    """엔티티 임베딩 시퀀스 → 글로벌 컨텍스트 벡터 (alive-masked pooling).

    alive 마스크를 이용해 dead 엔티티를 제외한 학습 가능한 가중합.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        embeds:     torch.Tensor,   # (B, N, embed_dim)
        alive_mask: torch.Tensor,   # (B, N) float — 1=alive, 0=dead
    ) -> torch.Tensor:              # (B, embed_dim)
        B = embeds.shape[0]
        q = self.query.expand(B, -1, -1)   # (B, 1, embed_dim)

        # key_padding_mask: True = 무시 (dead 엔티티)
        pad_mask = (alive_mask < 0.5)       # (B, N) bool

        # 모든 엔티티가 dead인 경우 fallback: 평균
        all_dead = pad_mask.all(dim=1, keepdim=True)   # (B, 1)
        pad_mask_safe = pad_mask & ~all_dead.expand_as(pad_mask)

        ctx, _ = self.cross_attn(
            q, embeds, embeds,
            key_padding_mask=pad_mask_safe,
            need_weights=False,
        )   # (B, 1, embed_dim)
        return self.norm(ctx.squeeze(1))   # (B, embed_dim)
