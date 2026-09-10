"""
wm_upper_v2.py — WM+HRL 상위 에이전트 (이산 액션 버전)

기존 wm_upper.py에서 연속 16-dim 액션 → 이산 40-dim (8x5 Categorical) 으로 변경.
OOD 문제 해결: actor가 직접 타겟 인덱스를 선택하므로 스냅 불일치 없음.
  - 공격드론 5개: 각각 5개 적 기지 중 1개 선택 → Categorical(5) x 5
  - 방어드론 3개: 각각 5개 아군 기지 구역 중 1개 선택 → Categorical(5) x 3
  - 총 행동: 8 x Categorical(5), one-hot 펼침 = 40-dim
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import time
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tssm import TSSM, TSSMConfig
from models.networks import STATE_DIM
from env.drone_env import DroneEnv
from env.units import UnitType
from lower_agent import rulebased_actions
from config import ENV_CFG, K, WM_CFG, WANDB_PROJECT, WANDB_ENTITY

GRID_SIZE  = 100
ACTION_DIM = 64   # v6: 8 에이전트 x 8 타겟 (기지 5 + 방어드론 3)


# ── 이산 Actor 네트워크 ────────────────────────────────────────────────────

class DiscreteActor(nn.Module):
    """이산 타겟 선택 — 8 에이전트 x 5 타겟 Categorical.

    straight-through gradient로 one-hot 액션 생성.
    TSSM의 stochastic state z와 동일한 패턴.
    """

    def __init__(self, feat_dim: int, n_agents: int = 8, n_targets: int = 8,
                 hidden: int = 256, n_attackers: int = 5, n_ally_zones: int = 5):
        super().__init__()
        self.n_agents     = n_agents
        self.n_targets    = n_targets
        self.n_attackers  = n_attackers   # 공격드론 수 (이후 인덱스 = 방어드론)
        self.n_ally_zones = n_ally_zones  # 방어드론이 고를 수 있는 아군 zone 수
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden,   hidden), nn.ELU(),
            nn.Linear(hidden,   n_agents * n_targets),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Straight-through one-hot 액션 반환 (Categorical sampling 기반 — exploration).

        v6: 방어드론 (agent idx 5-7)의 target [5-7]은 -inf mask
        (aliasing 방지 — attackers만 target [5-7]=적 방어드론 사용 가능).
        """
        B = feat.shape[0]
        logits = self.net(feat).reshape(B, self.n_agents, self.n_targets)
        logits = self._apply_defender_mask(logits, self.n_attackers, self.n_ally_zones)
        dist   = torch.distributions.Categorical(logits=logits)
        idx    = dist.sample()                                       # (B, n_agents)
        probs  = dist.probs
        hard   = F.one_hot(idx, self.n_targets).float()
        action = (hard - probs).detach() + probs                     # straight-through
        return action.flatten(start_dim=-2)                          # (B, 64)

    @staticmethod
    def _apply_defender_mask(logits, n_attackers: int = 5, n_ally_zones: int = 5):
        """Defender agents (idx>=n_attackers)는 target idx>=n_ally_zones를 -inf mask
        (방어드론은 아군 zone [0:n_ally_zones]만 선택 가능)."""
        logits = logits.clone()
        logits[..., n_attackers:, n_ally_zones:] = float("-inf")
        return logits

    def get_indices(self, feat: torch.Tensor) -> torch.Tensor:
        """Greedy 타겟 인덱스 선택 — 환경 상호작용용 (defender mask 적용).

        Args:
            feat: (B, feat_dim)
        Returns:
            indices: (B, 8) int — 각 에이전트의 타겟 인덱스
        """
        B = feat.shape[0]
        logits = self.net(feat).reshape(B, self.n_agents, self.n_targets)
        logits = self._apply_defender_mask(logits, self.n_attackers, self.n_ally_zones)
        return logits.argmax(dim=-1)   # (B, 8)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden,   hidden), nn.ELU(),
            nn.Linear(hidden,   1),
        )
        # 마지막 레이어 near-zero 초기화 → 초기 bootstrap 값 ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


# ── 리플레이 버퍼 ─────────────────────────────────────────────────────────

class ReplayBuffer:
    """시퀀스 단위 리플레이 버퍼 (오프-폴리시)."""

    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)

    def add_episode(self, ep: dict):
        """ep: {'obs': (T,205), 'actions': (T,40), 'rewards': (T,), 'dones': (T,)}"""
        self.episodes.append(ep)

    def __len__(self):
        return sum(len(e["rewards"]) for e in self.episodes)

    def sample_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator
                     ) -> Dict[str, torch.Tensor]:
        """길이 seq_len 서브시퀀스를 batch_size개 샘플링."""
        obs_list, act_list, rew_list, done_list = [], [], [], []
        episodes = [e for e in self.episodes if len(e["rewards"]) >= seq_len]
        if not episodes:
            raise RuntimeError(f"버퍼에 seq_len={seq_len} 이상인 에피소드 없음")

        chosen = rng.choice(len(episodes), size=batch_size, replace=True)
        for idx in chosen:
            ep  = episodes[idx]
            T   = len(ep["rewards"])
            start = rng.integers(0, T - seq_len + 1)
            sl    = slice(start, start + seq_len)
            obs_list.append(ep["obs"][sl])
            act_list.append(ep["actions"][sl])
            rew_list.append(ep["rewards"][sl])
            done_list.append(ep["dones"][sl])

        return {
            "obs":     torch.tensor(np.array(obs_list),  dtype=torch.float32),
            "actions": torch.tensor(np.array(act_list),  dtype=torch.float32),
            "rewards": torch.tensor(np.array(rew_list),  dtype=torch.float32),
            "dones":   torch.tensor(np.array(done_list), dtype=torch.float32),
        }


# ── 이산 서브골 변환 유틸 ──────────────────────────────────────────────────

def indices_to_onehot(indices: np.ndarray, n_targets: int = 8) -> np.ndarray:
    """타겟 인덱스 (8,) → one-hot 펼침 (40,).

    Args:
        indices:   (8,) int — 각 에이전트의 타겟 인덱스 (0~4)
        n_targets: 타겟 수
    Returns:
        action: (40,) float — one-hot 펼침
    """
    n_agents = len(indices)
    onehot = np.zeros((n_agents, n_targets), dtype=np.float32)
    for i, idx in enumerate(indices):
        onehot[i, idx] = 1.0
    return onehot.flatten()


def align_reward_on_arrival(kstep_obs, kstep_act, kstep_rew, kstep_done, terminal_obs):
    """수집된 K-step 시퀀스를 표준 TransDreamer 'reward-on-arrival' 규약으로 정렬.

    원본 수집: obs[t]=a[t-1]의 결과(resulting), reward[t]=a[t]가 일으킨 보상(originating)
              → obs와 reward가 한 칸 불일치 (off-by-one).
    표준 정렬: reward'[t] = 직전 매크로스텝 보상(= a[t-1]이 원인) → obs[t]와 동일 정렬.
              종말 보상 보존을 위해 종말 obs를 한 칸 추가(레퍼런스가 [1:]로 trim하는 것과 동형).
    모델(tssm.forward)은 손대지 않음 — 데이터 정렬만 표준에 맞춤.

    반환: (obs, actions, rewards, dones) 각 길이 n+1.
    """
    n = min(len(kstep_obs), len(kstep_act), len(kstep_rew), len(kstep_done))
    if terminal_obs is None:
        terminal_obs = kstep_obs[n - 1] if n > 0 else np.zeros(STATE_DIM, dtype=np.float32)

    dummy_act = np.zeros_like(kstep_act[0]) if n > 0 else np.zeros(ACTION_DIM, dtype=np.float32)
    obs_seq  = list(kstep_obs[:n])  + [np.asarray(terminal_obs, dtype=np.float32)]
    act_seq  = list(kstep_act[:n])  + [dummy_act]   # 종말 더미(a_shift로 미사용, 실제 action_dim)
    rew_seq  = [0.0]                + list(kstep_rew[:n])    # reward'[t] = reward(a[t-1]); reward'[0]=0
    done_seq = [0.0]                + list(kstep_done[:n])   # 종말 done이 종말 obs와 정렬

    return (
        np.array(obs_seq,  dtype=np.float32),   # (n+1, STATE_DIM)
        np.array(act_seq,  dtype=np.float32),   # (n+1, ACTION_DIM)
        np.array(rew_seq,  dtype=np.float32),   # (n+1,)
        np.array(done_seq, dtype=np.float32),   # (n+1,)
    )


def set_discrete_subgoals(env: DroneEnv, action_indices: np.ndarray) -> None:
    """이산 타겟 인덱스를 환경 서브골로 변환 (v6).

    action_indices: (8,) numpy int array, values in [0, 8)
      [0:5] = 공격드론:
                0-4 → 적 기지 인덱스
                5-7 → 적 방어드론 인덱스 (idx-5)
      [5:8] = 방어드론:
                0-4 → 아군 기지 구역
                5-7 → fallback: (idx-5) 기지 구역 (padding 해석)
    """
    subgoals = {}
    enemy_bases = env._enemy_bases
    ally_bases  = env._ally_bases
    enemy_defs  = [u for u in env._enemy_units
                   if u.unit_type == UnitType.DEFEND_DRONE]

    n_atk = env.n_ally_attack
    n_def = env.n_ally_defend
    n_eb  = env.n_enemy_base    # 타겟 idx < n_eb → 적 기지, 이상 → 적 방어드론
    n_ab  = env.n_ally_base

    # ── 공격드론 → 적 기지(idx<n_eb) or 적 방어드론(idx>=n_eb) ──────────
    for i in range(n_atk):
        aid  = f"ally_attack_{i}"
        unit = env._units.get(aid)
        if unit is None or not unit.alive:
            continue
        target_idx = int(action_indices[i])
        if target_idx < n_eb:
            base = enemy_bases[target_idx] if target_idx < len(enemy_bases) else None
            if base is not None and base.alive:
                subgoals[aid] = (base.x, base.y)
            else:
                alive = [b for b in enemy_bases if b.alive]
                if alive:
                    nearest = min(alive, key=lambda b: abs(b.x - unit.x) + abs(b.y - unit.y))
                    subgoals[aid] = (nearest.x, nearest.y)
        else:
            def_idx = target_idx - n_eb
            if def_idx < len(enemy_defs) and enemy_defs[def_idx].alive:
                d = enemy_defs[def_idx]
                subgoals[aid] = (d.x, d.y)
            else:
                alive_defs = [d for d in enemy_defs if d.alive]
                if alive_defs:
                    nearest = min(alive_defs, key=lambda d: abs(d.x - unit.x) + abs(d.y - unit.y))
                    subgoals[aid] = (nearest.x, nearest.y)
                else:
                    alive_bases = [b for b in enemy_bases if b.alive]
                    if alive_bases:
                        nearest = min(alive_bases, key=lambda b: abs(b.x - unit.x) + abs(b.y - unit.y))
                        subgoals[aid] = (nearest.x, nearest.y)

    # ── 방어드론 → 아군 기지 zone (idx in [0, n_ab)) ──────────────────
    for i in range(n_def):
        aid  = f"ally_defend_{i}"
        unit = env._units.get(aid)
        if unit is None or not unit.alive:
            continue
        zone_idx = int(action_indices[n_atk + i])
        if zone_idx >= n_ab:
            zone_idx = zone_idx % n_ab
        base = ally_bases[zone_idx]
        subgoals[aid] = (base.x, base.y)

    env.set_subgoals(subgoals)


# ── WM+HRL 트레이너 (이산 액션 버전) ──────────────────────────────────────

class WMUpperTrainerV2:
    """
    TSSM 기반 WM+HRL 상위 에이전트 — 이산 액션 버전.

    기존 WMUpperTrainer와 동일한 구조, 다음만 변경:
      - Actor: 연속 Sigmoid → 이산 Categorical (straight-through)
      - 서브골: set_subgoals(coords) → set_discrete_subgoals(indices)
      - 액션 표현: 스냅된 연속 좌표 → one-hot 펼침 (40-dim)
      - WM 입력: 항상 valid one-hot → OOD 문제 없음

    Stage 1 (warmup): WM만 학습 (랜덤 타겟 인덱스로 데이터 수집)
    Stage 2 (AC):     WM 고정 + imagination rollout → Actor/Critic 학습
    """

    def __init__(self, cfg: dict = WM_CFG, device: str = "cuda"):
        self.cfg    = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.step   = 0
        self.rng    = np.random.default_rng()

        self.n_agents     = cfg.get("n_action_agents",  8)
        self.n_targets    = cfg.get("n_action_targets", 5)
        self.n_attackers  = cfg.get("n_action_attackers", 5)
        self.n_ally_zones = cfg.get("n_action_ally_zones", 5)

        tssm_cfg    = TSSMConfig(**cfg["tssm"])
        self.tssm   = TSSM(tssm_cfg).to(self.device)
        feat_dim    = tssm_cfg.feat_dim

        ac_hidden = tssm_cfg.mlp_hidden
        self.actor       = DiscreteActor(feat_dim, self.n_agents, self.n_targets,
                                         hidden=ac_hidden,
                                         n_attackers=self.n_attackers,
                                         n_ally_zones=self.n_ally_zones).to(self.device)
        self.critic      = Critic(feat_dim, hidden=ac_hidden).to(self.device)
        self.slow_critic = Critic(feat_dim, hidden=ac_hidden).to(self.device)
        self.slow_critic.load_state_dict(self.critic.state_dict())

        # AEC head 파라미터도 WM optimizer에 포함 (aec_beta>0이면 gradient 흐름)
        self.wm_opt    = torch.optim.Adam(self.tssm.parameters(),  lr=cfg["wm_lr"])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["ac_lr"])
        self.critic_opt= torch.optim.Adam(self.critic.parameters(),lr=cfg["ac_lr"])

        self.buffer    = ReplayBuffer(max_episodes=cfg["buffer_max_size"] // 500)

    # ── 에피소드 수집 ─────────────────────────────────────────────────────

    def _collect_episode(self, seed: Optional[int] = None, eval_mode: bool = False
                         ) -> Tuple[dict, bool, bool, str]:
        """K-step 단위 데이터 수집 (이산 액션 버전).

        WM이 K-step 단위 전이를 학습하도록 매 K 환경 스텝마다 obs/action 기록.
        액션: one-hot 펼침 (40-dim) — 항상 valid distribution.
        """
        cfg       = self.cfg
        dev       = self.device
        cfg_gamma = cfg["tssm"]["gamma"]

        env = DroneEnv(config=ENV_CFG, render_mode=None)
        env.reset(seed=seed)

        kstep_obs:  List[np.ndarray] = []
        kstep_act:  List[np.ndarray] = []
        kstep_rew:  List[float]      = []
        kstep_done: List[float]      = []

        state = self.tssm.encode_obs(
            torch.zeros(1, self.tssm.cfg.state_dim, device=dev),
        )

        last_infos   = {}
        cur_indices  = np.zeros(self.n_agents, dtype=np.int64)   # 현재 타겟 인덱스
        cur_onehot   = np.zeros(self.n_agents * self.n_targets, dtype=np.float32)    # 현재 one-hot 액션
        cum_rew      = 0.0
        step_in_K    = 0   # 현재 K-step 사이클 내 위치 (0 ~ K-1)
        env_steps    = 0   # 실제 env.step 호출 횟수
        terminal_obs = None   # reward-on-arrival 정렬용 종말 상태 (종말 보상 보존)

        # warmup 전략: 50% 랜덤 / 50% 순차 배정 (드론 i → 기지 i)
        use_greedy_warmup = (random.random() < 0.5)

        while env.agents:
            obs = env.get_global_state()   # (205,)

            # ── K-step 경계: 새 서브골 결정 ──────────────────────────────────
            if step_in_K == 0:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                    state = self.tssm.encode_obs(obs_t, state.h)
                    feat  = torch.cat([state.h, state.z], dim=-1)

                    if self.step >= cfg["wm_warmup_steps"] or eval_mode:
                        if eval_mode:
                            # AC eval: argmax greedy
                            cur_indices = self.actor.get_indices(feat).squeeze(0).cpu().numpy()
                        else:
                            # AC train: Categorical sampling for exploration (defender-masked)
                            logits_a = self.actor.net(feat).reshape(1, self.n_agents, self.n_targets)
                            logits_a = DiscreteActor._apply_defender_mask(logits_a, self.n_attackers, self.n_ally_zones)
                            dist_a   = torch.distributions.Categorical(logits=logits_a)
                            cur_indices = dist_a.sample().squeeze(0).cpu().numpy()
                    else:
                        # warmup: 랜덤 또는 순차 타겟 인덱스 (v6: defender는 [0-4]만)
                        if use_greedy_warmup:
                            # 순차 배정: 공격드론 i → 기지 i, 방어드론 j → 구역 j
                            alive_base_indices = [bi for bi, b in enumerate(env._enemy_bases) if b.alive]
                            alive_zone_indices = [zi for zi, b in enumerate(env._ally_bases) if b.alive]
                            for i in range(self.n_attackers):
                                if alive_base_indices:
                                    cur_indices[i] = alive_base_indices[i % len(alive_base_indices)]
                                else:
                                    cur_indices[i] = np.random.randint(0, self.n_targets)
                            for i in range(self.n_agents - self.n_attackers):
                                if alive_zone_indices:
                                    cur_indices[self.n_attackers + i] = alive_zone_indices[i % len(alive_zone_indices)]
                                else:
                                    cur_indices[self.n_attackers + i] = np.random.randint(0, self.n_ally_zones)
                        else:
                            # 완전 랜덤 (defender는 zone[0:n_ally_zones]만)
                            n_d = self.n_agents - self.n_attackers
                            cur_indices[:self.n_attackers] = np.random.randint(0, self.n_targets, size=self.n_attackers)
                            cur_indices[self.n_attackers:] = np.random.randint(0, self.n_ally_zones, size=n_d)

                # 이산 서브골 → 환경 적용
                set_discrete_subgoals(env, cur_indices)

                # one-hot 액션 생성 (WM 학습용)
                cur_onehot = indices_to_onehot(cur_indices, self.n_targets)

                with torch.no_grad():
                    act_t = torch.tensor(cur_onehot, dtype=torch.float32, device=dev).unsqueeze(0)
                    state = self.tssm.step_dynamics(state, act_t)

                # K-step 결정 시점 기록
                kstep_obs.append(obs)
                kstep_act.append(cur_onehot.copy())
                cum_rew = 0.0

            # ── 하위 에이전트 실행 ────────────────────────────────────────────
            lower_actions = rulebased_actions(env)
            _, rewards, terminations, truncations, infos = env.step(lower_actions)
            env_steps += 1
            if infos:
                last_infos = next(iter(infos.values()))

            step_rew = float(np.mean(list(rewards.values()))) if rewards else 0.0
            cum_rew += (cfg_gamma ** step_in_K) * step_rew
            step_in_K += 1

            # ── 서브골 없는 에이전트 즉시 재할당 (K 주기 무관) ──────────────
            agents_no_subgoal = [a for a in env.agents if env._subgoals.get(a) is None]
            if agents_no_subgoal:
                in_ac = self.step >= cfg["wm_warmup_steps"] or eval_mode
                if in_ac:
                    with torch.no_grad():
                        obs_new = env.get_global_state()
                        obs_t   = torch.tensor(obs_new, dtype=torch.float32, device=dev).unsqueeze(0)
                        state   = self.tssm.encode_obs(obs_t, state.h)
                        feat    = torch.cat([state.h, state.z], dim=-1)
                        if eval_mode:
                            new_indices = self.actor.get_indices(feat).squeeze(0).cpu().numpy()
                        else:
                            logits_a = self.actor.net(feat).reshape(1, self.n_agents, self.n_targets)
                            logits_a = DiscreteActor._apply_defender_mask(logits_a, self.n_attackers, self.n_ally_zones)
                            dist_a   = torch.distributions.Categorical(logits=logits_a)
                            new_indices = dist_a.sample().squeeze(0).cpu().numpy()

                    # 서브골 없는 에이전트만 부분 재할당 (v6: n_targets=8)
                    partial = {}
                    enemy_bases = env._enemy_bases
                    ally_bases  = env._ally_bases
                    enemy_defs  = [u for u in env._enemy_units
                                   if u.unit_type == UnitType.DEFEND_DRONE]
                    for a in agents_no_subgoal:
                        unit = env._units.get(a)
                        if unit is None or not unit.alive:
                            continue
                        idx = env.possible_agents.index(a)
                        target_idx = int(new_indices[idx])
                        if unit.unit_type == UnitType.ATTACK_DRONE:
                            if target_idx < env.n_enemy_base:
                                base = enemy_bases[target_idx]
                                if base.alive:
                                    partial[a] = (base.x, base.y)
                                else:
                                    alive = [b for b in enemy_bases if b.alive]
                                    if alive:
                                        nearest = min(alive,
                                                      key=lambda b: abs(b.x - base.x) + abs(b.y - base.y))
                                        partial[a] = (nearest.x, nearest.y)
                            else:
                                def_idx = target_idx - env.n_enemy_base
                                if def_idx < len(enemy_defs) and enemy_defs[def_idx].alive:
                                    d = enemy_defs[def_idx]
                                    partial[a] = (d.x, d.y)
                                else:
                                    alive_d = [d for d in enemy_defs if d.alive]
                                    if alive_d:
                                        nearest = min(alive_d,
                                                      key=lambda d: abs(d.x - unit.x) + abs(d.y - unit.y))
                                        partial[a] = (nearest.x, nearest.y)
                                    else:
                                        alive_b = [b for b in enemy_bases if b.alive]
                                        if alive_b:
                                            nearest = min(alive_b,
                                                          key=lambda b: abs(b.x - unit.x) + abs(b.y - unit.y))
                                            partial[a] = (nearest.x, nearest.y)
                        else:  # DEFEND_DRONE
                            zone_idx = target_idx if target_idx < env.n_ally_base else target_idx % env.n_ally_base
                            base = ally_bases[zone_idx]
                            partial[a] = (base.x, base.y)
                else:
                    # warmup: 랜덤 타겟 할당 (v6: attacker는 기지+방어드론 포함)
                    partial = {}
                    enemy_bases = env._enemy_bases
                    ally_bases  = env._ally_bases
                    for a in agents_no_subgoal:
                        unit = env._units.get(a)
                        if unit is None or not unit.alive:
                            continue
                        if unit.unit_type == UnitType.ATTACK_DRONE:
                            alive_b = [b for b in enemy_bases if b.alive]
                            alive_d = [d for d in env._enemy_units
                                       if d.alive and d.unit_type == UnitType.DEFEND_DRONE]
                            candidates = alive_b + alive_d
                            if candidates:
                                t = random.choice(candidates)
                                partial[a] = (t.x, t.y)
                        else:  # DEFEND_DRONE
                            alive_atk = [e for e in env._enemy_units
                                         if e.alive and e.unit_type == UnitType.ATTACK_DRONE]
                            if alive_atk:
                                e = random.choice(alive_atk)
                                partial[a] = (e.x, e.y)

                if partial:
                    env.set_subgoals(partial)

                    # ── WM 기록: 서브골 무효화 = K-step 조기 종료 + 새 K-step 시작 ──
                    if step_in_K > 0:
                        kstep_rew.append(cum_rew)
                        kstep_done.append(0.0)   # 에피소드 종료 아님

                        # 새 K-step 시작: 현재 obs + 현재 서브골의 one-hot 기록 (v6)
                        obs_now = env.get_global_state()
                        new_indices_snap = np.zeros(self.n_agents, dtype=np.int64)
                        enemy_defs_snap = [u for u in env._enemy_units
                                           if u.unit_type == UnitType.DEFEND_DRONE]
                        for j, aid_snap in enumerate(env.possible_agents):
                            sg = env._subgoals.get(aid_snap)
                            if sg is None:
                                new_indices_snap[j] = 0
                                continue
                            sx, sy = sg
                            unit_snap = env._units.get(aid_snap)
                            if unit_snap is not None and unit_snap.unit_type == UnitType.ATTACK_DRONE:
                                # 스냅된 좌표와 기지 vs 방어드론 중 가장 가까운 쪽 선택
                                base_dists = [abs(b.x - sx) + abs(b.y - sy)
                                              for b in env._enemy_bases]
                                def_dists  = [abs(d.x - sx) + abs(d.y - sy)
                                              for d in enemy_defs_snap]
                                min_b_idx = int(np.argmin(base_dists))
                                min_b_d   = base_dists[min_b_idx]
                                if def_dists and min(def_dists) < min_b_d:
                                    new_indices_snap[j] = env.n_enemy_base + int(np.argmin(def_dists))
                                else:
                                    new_indices_snap[j] = min_b_idx
                            else:
                                # 방어드론: 가장 가까운 아군 기지 (0-4)
                                dists = [abs(b.x - sx) + abs(b.y - sy)
                                         for b in env._ally_bases]
                                new_indices_snap[j] = int(np.argmin(dists))

                        onehot_now = indices_to_onehot(new_indices_snap, self.n_targets)

                        with torch.no_grad():
                            obs_t2 = torch.tensor(obs_now, dtype=torch.float32, device=dev).unsqueeze(0)
                            state  = self.tssm.encode_obs(obs_t2, state.h)
                            act_t2 = torch.tensor(onehot_now, dtype=torch.float32, device=dev).unsqueeze(0)
                            state  = self.tssm.step_dynamics(state, act_t2)

                        kstep_obs.append(obs_now)
                        kstep_act.append(onehot_now)
                        cum_rew   = 0.0
                        step_in_K = 0

            ep_done = (
                all(terminations.values()) or all(truncations.values())
                or bool(last_infos.get("won")) or bool(last_infos.get("lost"))
            )

            # ── K-step 사이클 완료 또는 에피소드 종료 시 기록 ────────────────
            if step_in_K == K or ep_done:
                kstep_rew.append(cum_rew)
                kstep_done.append(float(ep_done))
                step_in_K = 0

            if ep_done:
                terminal_obs = env.get_global_state()   # a[n-1]의 결과 = 종말 상태
                break

        rsa = env._current_strategy

        # ── Episode meta ──────────────────────────────────────────────────
        won  = bool(last_infos.get("won",  False))
        lost = bool(last_infos.get("lost", False))
        enemy_bases_destroyed = sum(1 for b in env._enemy_bases if not b.alive)
        ally_bases_destroyed  = sum(1 for b in env._ally_bases  if not b.alive)
        ally_atk_alive = sum(1 for i in range(env.n_ally_attack)
                              if env._units.get(f"ally_attack_{i}")
                              and env._units[f"ally_attack_{i}"].alive)
        ally_def_alive = sum(1 for i in range(env.n_ally_defend)
                              if env._units.get(f"ally_defend_{i}")
                              and env._units[f"ally_defend_{i}"].alive)
        drones_survived = ally_atk_alive + ally_def_alive
        if won:
            term_reason = "win"
        elif lost:
            term_reason = "lost"
        elif ally_atk_alive == 0 and ally_def_alive == 0:
            term_reason = "drones_exhausted"
        else:
            term_reason = "timeout"
        # Per-target distribution (attackers only, idx 0-4 → target idx 0-7)
        atk_target_counts = [0] * self.n_targets
        if kstep_act:
            acts_arr = np.array(kstep_act).reshape(-1, self.n_agents, self.n_targets)
            atk_idx = acts_arr[:, :5, :].argmax(axis=-1).flatten()
            for ti in atk_idx:
                atk_target_counts[int(ti)] += 1

        meta = {
            "env_steps":             env_steps,
            "enemy_bases_destroyed": enemy_bases_destroyed,
            "ally_bases_destroyed":  ally_bases_destroyed,
            "ally_atk_alive":        ally_atk_alive,
            "ally_def_alive":        ally_def_alive,
            "drones_survived":       drones_survived,
            "termination_reason":    term_reason,
            "atk_target_counts":     atk_target_counts,
        }

        env.close()

        # 표준 TransDreamer reward-on-arrival 정렬 (모델 불변, 데이터만)
        obs_arr, act_arr, rew_arr, done_arr = align_reward_on_arrival(
            kstep_obs, kstep_act, kstep_rew, kstep_done, terminal_obs
        )
        ep = {
            "obs":     obs_arr,    # (n+1, 205)
            "actions": act_arr,    # (n+1, 64)
            "rewards": rew_arr,    # (n+1,)  reward'[t]=reward(a[t-1])
            "dones":   done_arr,   # (n+1,)
            "meta":    meta,
        }
        return ep, won, lost, rsa

    # ── WM 학습 스텝 ──────────────────────────────────────────────────────

    def _train_wm(self, batch: Dict[str, torch.Tensor]) -> dict:
        obs     = batch["obs"].to(self.device)      # (B, T, 205)
        actions = batch["actions"].to(self.device)   # (B, T, 40)
        rewards = batch["rewards"].to(self.device)   # (B, T)
        dones   = batch["dones"].to(self.device).bool()  # (B, T)

        loss_out = self.tssm(obs, actions, dones, rewards)
        self.wm_opt.zero_grad()
        loss_out.total.backward()
        nn.utils.clip_grad_norm_(self.tssm.parameters(), self.cfg["grad_clip"])
        self.wm_opt.step()
        return loss_out.metrics

    # ── AC 학습 스텝 ──────────────────────────────────────────────────────

    def _train_ac(self, batch: Dict[str, torch.Tensor]) -> dict:
        cfg = self.cfg
        dev = self.device
        obs     = batch["obs"].to(dev)
        actions = batch["actions"].to(dev)
        B, T, _ = obs.shape

        tssm_cfg = self.tssm.cfg
        C        = tssm_cfg
        H        = C.imagination_horizon
        # K-step WM: imagination 각 스텝 = K env스텝
        gamma    = C.gamma ** K   # K-step 간 discount
        lambda_  = C.lambda_

        # 배치 전체를 2-pass로 돌려 posterior 추출
        with torch.no_grad():
            a_zeros   = torch.zeros(B, 1, C.action_dim, device=dev)
            z_zeros   = torch.zeros(B, T, C.stoch_dim,  device=dev)
            a_shifted = torch.cat([a_zeros, actions[:, :-1]], dim=1)
            # Pass 1: z=0으로 rough h 계산
            h_v1 = self.tssm.dyn_model(z_zeros, a_shifted)
            obs_flat = obs.reshape(B * T, -1)
            p1, _ = self.tssm.repr_model(obs_flat, h_v1.reshape(B * T, -1))
            z1 = TSSM.sample_straight_through(p1).reshape(B, T, C.stoch_dim)
            # Pass 2: posterior z로 refined h 계산
            z_shifted = torch.cat([torch.zeros(B, 1, C.stoch_dim, device=dev),
                                   z1[:, :-1]], dim=1)
            h_seq  = self.tssm.dyn_model(z_shifted, a_shifted)
            p2, _  = self.tssm.repr_model(obs_flat, h_seq.reshape(B * T, -1))
            z_post = TSSM.sample_straight_through(p2).reshape(B, T, C.stoch_dim)
            start_h = h_seq[:, -1, :].detach()
            start_z = z_post[:, -1, :].detach()
            # seed context: 마지막 10 K-step 결정
            N_seed = min(10, T)
            seed_z = z_post[:, -N_seed:, :].detach()
            seed_a = actions[:, -N_seed:, :].detach()

        # ── TSSM 파라미터 freeze (업데이트 차단, but gradient는 통과) ──────
        for p in self.tssm.parameters():
            p.requires_grad_(False)

        try:
            # imagination rollout: actor가 one-hot 액션 직접 생성 → OOD 없음
            imag_out = self.tssm.imagine(start_h, start_z, self.actor, H,
                                         seed_z=seed_z, seed_a=seed_a)

            # bootstrap: slow_critic 사용
            with torch.no_grad():
                all_vals = self.slow_critic(
                    imag_out.states.reshape(B * H, -1).detach()
                ).reshape(B, H)

            # lambda-return (symlog-reward 공간)
            ret_list: List[torch.Tensor] = [torch.empty(0)] * H
            ret_list[-1] = (imag_out.rewards[:, -1]
                            + gamma * imag_out.continues[:, -1] * all_vals[:, -1].detach())
            for t in reversed(range(H - 1)):
                bootstrap = ((1 - lambda_) * all_vals[:, t + 1].detach()
                             + lambda_ * ret_list[t + 1])
                ret_list[t] = (imag_out.rewards[:, t]
                               + gamma * imag_out.continues[:, t] * bootstrap)
            returns = torch.stack(ret_list, dim=1)   # (B, H)

            # critic용 데이터 미리 저장
            states_det = imag_out.states.detach()
            target     = returns.detach()
            imag_rew   = imag_out.rewards.detach().mean().item()

            # DreamerV3 percentile 정규화
            ret_flat   = returns.detach().reshape(-1).float()
            lo         = torch.quantile(ret_flat, 0.05)
            hi         = torch.quantile(ret_flat, 0.95)
            scale      = torch.clamp(hi - lo, min=1.0)

            # Entropy bonus (Dreamer-style exploration regularization)
            # v6: defender mask 적용 — aliasing 방지 + entropy 이론값 일치
            ent_coef   = cfg.get("entropy_coef", 1e-3)
            feats_flat = imag_out.states.reshape(B * H, -1)
            logits_act = self.actor.net(feats_flat).reshape(
                B * H, self.n_agents, self.n_targets
            )
            logits_act = self.actor._apply_defender_mask(logits_act, self.actor.n_attackers, self.actor.n_ally_zones)
            dist_act   = torch.distributions.Categorical(logits=logits_act)
            entropy_m  = dist_act.entropy().sum(dim=-1).mean()

            actor_loss = -((returns - lo) / scale).mean() - ent_coef * entropy_m

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg["grad_clip"])
            self.actor_opt.step()

        finally:
            # 반드시 TSSM 파라미터 unfreeze
            for p in self.tssm.parameters():
                p.requires_grad_(True)

        # Critic loss
        critic_preds = self.critic(states_det.reshape(B * H, -1)).reshape(B, H)
        critic_loss  = F.mse_loss(critic_preds, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg["grad_clip"])
        self.critic_opt.step()

        # Slow critic EMA 업데이트
        tau = cfg["slow_critic_tau"]
        for p, sp in zip(self.critic.parameters(), self.slow_critic.parameters()):
            sp.data.mul_(1 - tau).add_(p.data * tau)

        return {
            "actor_loss":   actor_loss.item(),
            "critic_loss":  critic_loss.item(),
            "lambda_ret":   target.mean().item(),
            "imag_reward":  imag_rew,
            "actor_entropy": entropy_m.item(),
        }

    # ── 평가 ──────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 30) -> dict:
        wins = 0
        for _ in range(n_episodes):
            seed = int(self.rng.integers(0, 100_000))
            _, won, _, _ = self._collect_episode(seed=seed, eval_mode=True)
            wins += int(won)
        return {"win_rate": wins / n_episodes}

    # ── 체크포인트 ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "tssm":        self.tssm.state_dict(),
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "wm_opt":      self.wm_opt.state_dict(),
            "actor_opt":   self.actor_opt.state_dict(),
            "critic_opt":  self.critic_opt.state_dict(),
            "step":        self.step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.tssm.load_state_dict(ckpt["tssm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.slow_critic.load_state_dict(ckpt["slow_critic"])
        # Forward-compat: optimizer state added later — skip if absent.
        if "wm_opt" in ckpt:
            self.wm_opt.load_state_dict(ckpt["wm_opt"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.step = ckpt.get("step", 0)

    # ── 메인 학습 루프 ─────────────────────────────────────────────────────

    def train(self, out_dir: str, seed: int = 0, use_wandb: bool = False, run_name: str = None):
        os.makedirs(out_dir, exist_ok=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        cfg    = self.cfg
        dev    = self.device
        log_f  = open(os.path.join(out_dir, "log.csv"), "w", newline="")
        writer = csv.DictWriter(log_f, fieldnames=[
            "step", "win_rate", "recon", "kl", "rew_pred", "aec",
            "actor_loss", "critic_loss", "lambda_ret", "actor_entropy"
        ])
        writer.writeheader()

        # 타겟 분포 CSV (attacker target indices, 100 ep window 평균 fraction)
        tgt_log_f = open(os.path.join(out_dir, "target_dist.csv"), "w", newline="")
        tgt_writer = csv.DictWriter(tgt_log_f, fieldnames=[
            "step",
            *[f"tgt_{i}_frac" for i in range(self.n_targets)],
            "n_decisions",
        ])
        tgt_writer.writeheader()
        tgt_window = deque(maxlen=100)   # last 100 episodes' atk_target_counts

        # 에피소드 단위 CSV (RSA 전략, 승패, 보상)
        ep_log_f  = open(os.path.join(out_dir, "episodes.csv"), "w", newline="")
        ep_writer = csv.DictWriter(ep_log_f, fieldnames=[
            "episode", "step", "rsa", "result", "reward",
            "macro_length", "env_length", "termination_reason",
            "enemy_bases_destroyed", "ally_bases_destroyed", "drones_survived",
        ])
        ep_writer.writeheader()

        # WandB 초기화
        wandb_run = None
        if use_wandb:
            import wandb
            wandb_run = wandb.init(
                project = WANDB_PROJECT,
                entity  = WANDB_ENTITY,
                name    = run_name or f"wm_v2_s{seed}",
                group   = "WM+HRL-v2-discrete",
                config  = {**{k: v for k, v in cfg.items() if k != "tssm"},
                           **cfg["tssm"], "seed": seed, "K": K,
                           "model": "WM+HRL-v2-discrete"},
                reinit  = True,
            )
            # 모든 지표의 x축을 env_step(실제 환경 상호작용량)으로 — 학습 step 대신
            wandb_run.define_metric("env_step")
            wandb_run.define_metric("*", step_metric="env_step")

        print(f"\n{'='*55}")
        print(f"  WM+HRL v2 (이산 액션)  (seed={seed})")
        print(f"  총 스텝: {cfg['total_steps']:,}  |  warmup: {cfg['wm_warmup_steps']:,}")
        print(f"  ACTION_DIM={self.n_agents * self.n_targets}  ({self.n_agents} agents x {self.n_targets} targets)")
        print(f"  TSSM feat_dim={self.tssm.cfg.feat_dim}  |  out={out_dir}")
        print(f"{'='*55}")

        win_history: List[int] = []
        ep_count       = 0
        total_env_steps = 0   # 누적 환경 상호작용 스텝 (wandb x축)
        t0          = time.time()
        wm_window   = {k: deque(maxlen=100) for k in ["total", "recon", "kl", "reward", "aec"]}
        ac_window   = {k: deque(maxlen=100) for k in
                       ["actor_loss", "critic_loss", "lambda_ret", "imag_reward", "actor_entropy"]}

        while self.step < cfg["total_steps"]:
            # ── 에피소드 수집 ──────────────────────────────────────────────
            ep, won, lost, rsa = self._collect_episode(seed=ep_count)
            ep_count  += 1
            win_history.append(int(won))
            ep_len    = len(ep["rewards"])
            ep_reward = float(np.sum(ep["rewards"]))
            result    = "win" if won else "lost" if lost else "draw"
            if ep_len >= cfg["seq_len"]:
                self.buffer.add_episode(ep)

            # ── 에피소드 단위 CSV 기록 ────────────────────────────────────
            meta = ep.get("meta", {})
            env_len        = int(meta.get("env_steps", 0))
            total_env_steps += env_len
            term_reason    = meta.get("termination_reason", "unknown")
            enemy_bd       = int(meta.get("enemy_bases_destroyed", 0))
            ally_bd        = int(meta.get("ally_bases_destroyed", 0))
            drones_surv    = int(meta.get("drones_survived", 0))
            ep_writer.writerow({
                "episode":              ep_count,
                "step":                 self.step,
                "rsa":                  rsa,
                "result":               result,
                "reward":               round(ep_reward, 4),
                "macro_length":         ep_len,
                "env_length":           env_len,
                "termination_reason":   term_reason,
                "enemy_bases_destroyed": enemy_bd,
                "ally_bases_destroyed":  ally_bd,
                "drones_survived":       drones_surv,
            })
            ep_log_f.flush()

            # 타겟 분포 누적 (per-episode count vector, length n_targets)
            atk_tc = meta.get("atk_target_counts")
            if atk_tc is not None:
                tgt_window.append(atk_tc)

            # ── 에피소드 단위 WandB 로깅 (정리: 핵심 지표만)
            if wandb_run:
                wandb_run.log({
                    "env_step":                       total_env_steps,
                    "episode/env_length":             env_len,
                    "episode/reward":                 ep_reward,
                    "episode/enemy_bases_destroyed":  enemy_bd,
                    "episode/ally_bases_destroyed":   ally_bd,
                    "episode/drones_survived":        drones_surv,
                }, commit=False)

            if len(self.buffer) < cfg["batch_size"] * cfg["seq_len"]:
                continue

            # ── AEC β 스케줄링 (warmup 때 높게, AC 때 낮게) ──────────────
            if "aec_beta_warmup" in cfg and cfg["aec_beta_warmup"] is not None:
                if self.step < cfg["wm_warmup_steps"]:
                    self.tssm.cfg.aec_beta = cfg["aec_beta_warmup"]
                else:
                    self.tssm.cfg.aec_beta = cfg["tssm"]["aec_beta"]

            # ── WM 학습 ───────────────────────────────────────────────────
            try:
                batch      = self.buffer.sample_batch(cfg["batch_size"], cfg["seq_len"], self.rng)
                wm_metrics = self._train_wm(batch)
                for k, v in wm_metrics.items():
                    if k in wm_window:
                        wm_window[k].append(v)
            except RuntimeError as e:
                print(f"  [WM 오류] step {self.step}: {e}")
                self.step += 1
                continue

            # ── AC 학습 (warmup 이후) ─────────────────────────────────────
            if self.step >= cfg["wm_warmup_steps"]:
                try:
                    ac_metrics = self._train_ac(batch)
                    for k, v in ac_metrics.items():
                        ac_window[k].append(v)
                except RuntimeError as e:
                    print(f"  [AC 오류] step {self.step}: {e}")

            self.step += 1

            # ── 스텝 단위 로깅 ────────────────────────────────────────────
            if self.step % cfg["log_every"] == 0:
                wr      = float(np.mean(win_history[-100:])) if win_history else 0.0
                elapsed = time.time() - t0
                wm_m    = {k: float(np.mean(v)) if v else 0.0 for k, v in wm_window.items()}
                ac_m    = {k: float(np.mean(v)) if v else 0.0 for k, v in ac_window.items()}
                in_ac   = self.step >= cfg["wm_warmup_steps"]

                if in_ac:
                    print(f"  step={self.step:>6d} | wr={wr:.3f} | "
                          f"recon={wm_m['recon']:.4f} | kl={wm_m['kl']:.4f} | "
                          f"actor={ac_m.get('actor_loss', 0):+.4f} | "
                          f"critic={ac_m.get('critic_loss', 0):.4f} | "
                          f"ret={ac_m.get('lambda_ret', 0):.2f} | "
                          f"H={ac_m.get('actor_entropy', 0):.3f} | "
                          f"({elapsed/60:.1f}m)")
                else:
                    print(f"  [WM warmup] step={self.step:>6d} | wr={wr:.3f} | "
                          f"recon={wm_m['recon']:.4f} | kl={wm_m['kl']:.4f} | "
                          f"({elapsed/60:.1f}m)")

                row = {
                    "step":          self.step,
                    "win_rate":      round(wr, 4),
                    "recon":         round(wm_m["recon"], 6),
                    "kl":            round(wm_m["kl"], 6),
                    "rew_pred":      round(wm_m["reward"], 6),
                    "aec":           round(wm_m["aec"], 6),
                    "actor_loss":    round(ac_m.get("actor_loss", 0), 6) if in_ac else 0,
                    "critic_loss":   round(ac_m.get("critic_loss", 0), 6) if in_ac else 0,
                    "lambda_ret":    round(ac_m.get("lambda_ret", 0), 6) if in_ac else 0,
                    "actor_entropy": round(ac_m.get("actor_entropy", 0), 6) if in_ac else 0,
                }
                writer.writerow(row)
                log_f.flush()

                # target distribution 로깅 (지난 100 ep 누적)
                if tgt_window:
                    tot = np.zeros(self.n_targets, dtype=np.int64)
                    for c in tgt_window:
                        tot += np.array(c, dtype=np.int64)
                    n_dec = int(tot.sum())
                    fracs = tot / max(1, n_dec)
                    tgt_row = {"step": self.step, "n_decisions": n_dec}
                    for i in range(self.n_targets):
                        tgt_row[f"tgt_{i}_frac"] = round(float(fracs[i]), 4)
                    tgt_writer.writerow(tgt_row)
                    tgt_log_f.flush()
                if wandb_run:
                    # 정리: wm/total_loss 제거 (recon+kl+rew의 합, 중복)
                    log_dict = {
                        "env_step":       total_env_steps,
                        "win_rate":       wr,
                        "wm/recon":       wm_m["recon"],
                        "wm/kl":          wm_m["kl"],
                        "wm/rew_pred":    wm_m["reward"],
                        "wm/aec":         wm_m["aec"],
                    }
                    if in_ac:
                        log_dict.update({
                            "ac/actor_loss":    ac_m.get("actor_loss", 0),
                            "ac/critic_loss":   ac_m.get("critic_loss", 0),
                            "ac/lambda_ret":    ac_m.get("lambda_ret", 0),
                            "ac/imag_reward":   ac_m.get("imag_reward", 0),
                            "ac/actor_entropy": ac_m.get("actor_entropy", 0),
                        })
                    wandb_run.log(log_dict, step=self.step)

            # ── 평가 ──────────────────────────────────────────────────────
            if self.step % cfg["eval_every"] == 0:
                eval_r = self.evaluate(n_episodes=10)
                print(f"  [EVAL] step={self.step} | win_rate={eval_r['win_rate']:.3f}")
                if wandb_run:
                    wandb_run.log({"eval/win_rate": eval_r["win_rate"],
                                   "env_step": total_env_steps}, step=self.step)

            # ── 저장 ──────────────────────────────────────────────────────
            if self.step % cfg["save_every"] == 0:
                ckpt_p = os.path.join(out_dir, f"wm_v2_step{self.step:06d}.pt")
                self.save(ckpt_p)

        # ── 최종 저장 ─────────────────────────────────────────────────────
        final_wr = self.evaluate(n_episodes=50)["win_rate"]
        print(f"\n  최종 win_rate (50ep): {final_wr:.3f}")
        self.save(os.path.join(out_dir, "wm_v2_final.pt"))
        log_f.close()
        ep_log_f.close()
        tgt_log_f.close()
        if wandb_run:
            wandb_run.summary["final_win_rate"] = final_wr
            wandb_run.finish()
