"""ppo_hrl_upper.py — PPO+HRL 상위 에이전트 (no world model baseline).

v6 (WM+HRL)와 동일한 환경/액션공간/Lower agent를 사용하되 다음만 변경:
  - TSSM encoder 제거 → MLP encoder
  - Imagination + V_lambda 제거 → 환경 직접 학습 (on-policy)
  - Replay buffer 제거 → Rollout buffer (macro-step 단위)
  - Actor-Critic 학습 → Standard PPO (clip + GAE)

비교 fairness:
  - 같은 ENV_CFG (config_v3 monkey-patch)
  - 같은 K=10
  - 같은 8x8 discrete action + defender mask
  - 같은 rule-based lower agent
  - Categorical sampling (학습 + eval, argmax 사용 안 함 — 사용자 명시)
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

from models.networks import STATE_DIM
from env.drone_env import DroneEnv
from env.units import UnitType
from lower_agent import rulebased_actions
from wm_upper_v2 import set_discrete_subgoals
from config import K, WANDB_PROJECT, WANDB_ENTITY


# ── PPO 하이퍼파라미터 (Codex 권고 기준) ──────────────────────────────────
PPO_CFG = {
    "hidden_dim":         256,
    "lr":                 3e-4,
    "gamma":              0.99,         # env step 단위 (macro: gamma^K_actual)
    "gae_lambda":         0.95,
    "clip_eps":           0.2,
    "n_epochs":           4,
    "minibatch_size":     256,
    "rollout_macro_steps": 2048,        # update당 macro 수
    "entropy_coef":       0.01,
    "value_coef":         0.5,
    "grad_clip":          0.5,

    "total_episodes":    100_000,        # v6 self.step 의미와 동일 (episode count)
    "log_every":         100,            # episodes
    "eval_every":        5_000,          # episodes
    "save_every":        5_000,          # episodes

    # 평가
    "eval_episodes":     10,             # 학습 중 sampling
    "eval_episodes_argmax": 10,          # 학습 중 argmax (보조 로그용)
    "final_eval_episodes": 50,
}


# ── 네트워크 ────────────────────────────────────────────────────────────────

class MLPEncoder(nn.Module):
    """Global state 205 → feat 256 (MLP, TSSM 대체)."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(),
            nn.Linear(hidden,    hidden), nn.ELU(),
        )
        self.feat_dim = hidden

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPODiscreteActor(nn.Module):
    """8 agents × 8 targets Categorical. v6 DiscreteActor와 같은 mask, log_prob 명시."""

    def __init__(self, feat_dim: int, n_agents: int = 8, n_targets: int = 8,
                 hidden: int = 256, n_attackers: int = 5, n_ally_zones: int = 5):
        super().__init__()
        self.n_agents     = n_agents
        self.n_targets    = n_targets
        self.n_attackers  = n_attackers
        self.n_ally_zones = n_ally_zones
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden,   hidden), nn.ELU(),
            nn.Linear(hidden,   n_agents * n_targets),
        )

    def _apply_defender_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """v6와 동일: defenders(idx>=n_attackers) → ally-zone 밖 targets(idx>=n_ally_zones) -inf."""
        logits = logits.clone()
        logits[..., self.n_attackers:, self.n_ally_zones:] = float("-inf")
        return logits

    def _logits(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        logits = self.net(feat).reshape(B, self.n_agents, self.n_targets)
        return self._apply_defender_mask(logits)

    def act(self, feat: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """샘플링 (학습 + sampling-eval).

        Returns:
            indices: (B, n_agents) int
            log_prob: (B,) — 8 agents 독립 Categorical joint log prob
            entropy: (B,) — joint entropy
        """
        logits = self._logits(feat)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            indices = logits.argmax(dim=-1)
            log_prob = dist.log_prob(indices).sum(-1)
        else:
            indices = dist.sample()
            log_prob = dist.log_prob(indices).sum(-1)
        entropy = dist.entropy().sum(-1)
        return indices, log_prob, entropy

    def evaluate(self, feat: torch.Tensor, indices: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """주어진 indices의 log_prob + entropy 재계산 (PPO update용)."""
        logits = self._logits(feat)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(indices).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden,   hidden), nn.ELU(),
            nn.Linear(hidden,   1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


# ── Rollout 버퍼 ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Macro-step 단위 on-policy 버퍼. 가변 K (조기 종료 처리)."""

    def __init__(self):
        self.states:   List[np.ndarray] = []
        self.indices:  List[np.ndarray] = []
        self.log_probs: List[float]      = []
        self.values:   List[float]      = []
        self.rewards:  List[float]      = []
        self.dones:    List[float]      = []      # episode done after this macro
        self.k_actual: List[int]        = []      # 실제 macro 길이 (1 ~ K)
        self.next_value: float           = 0.0    # bootstrap (last macro 이후)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.indices.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.k_actual.clear()
        self.next_value = 0.0

    def add(self, state: np.ndarray, indices: np.ndarray,
            log_prob: float, value: float,
            reward: float, done: bool, k_actual: int) -> None:
        self.states.append(state)
        self.indices.append(indices)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.k_actual.append(k_actual)

    def compute_gae(self, gamma_env: float, lam: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Macro-step GAE. macro discount = gamma_env**k_actual.

        Returns:
            advantages: (T,)
            returns:    (T,)
        """
        n = len(self.states)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            macro_gamma = gamma_env ** self.k_actual[t]
            if t == n - 1:
                next_v = self.next_value
                next_nonterminal = 1.0 - self.dones[t]
            else:
                next_v = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + macro_gamma * next_v * next_nonterminal
                     - self.values[t])
            last_gae = (delta
                        + macro_gamma * lam * next_nonterminal * last_gae)
            adv[t] = last_gae
        ret = adv + np.array(self.values, dtype=np.float32)
        return adv, ret

    def to_tensors(self, device: torch.device,
                   gamma_env: float, lam: float) -> dict:
        adv, ret = self.compute_gae(gamma_env, lam)
        # advantage 정규화
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "states":    torch.tensor(np.array(self.states),  dtype=torch.float32, device=device),
            "indices":   torch.tensor(np.array(self.indices), dtype=torch.long,    device=device),
            "log_probs": torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            "advantages": torch.tensor(adv, dtype=torch.float32, device=device),
            "returns":   torch.tensor(ret, dtype=torch.float32, device=device),
        }


# ── 트레이너 ────────────────────────────────────────────────────────────────

class PPOHRLTrainer:
    """PPO+HRL 상위 에이전트 트레이너."""

    def __init__(self, cfg: dict = PPO_CFG, device: str = "cuda"):
        self.cfg    = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.episode_count = 0
        self.macro_step_count = 0
        self.env_step_count   = 0

        self.n_agents     = cfg.get("n_action_agents",    8)
        self.n_targets    = cfg.get("n_action_targets",   8)
        self.n_attackers  = cfg.get("n_action_attackers", 5)
        self.n_ally_zones = cfg.get("n_action_ally_zones", 5)
        state_dim         = cfg.get("state_dim", STATE_DIM)

        self.encoder = MLPEncoder(state_dim, cfg["hidden_dim"]).to(self.device)
        feat_dim     = self.encoder.feat_dim
        self.actor   = PPODiscreteActor(feat_dim, self.n_agents, self.n_targets,
                                        cfg["hidden_dim"],
                                        n_attackers=self.n_attackers,
                                        n_ally_zones=self.n_ally_zones).to(self.device)
        self.critic  = Critic(feat_dim, cfg["hidden_dim"]).to(self.device)

        params = list(self.encoder.parameters()) + \
                 list(self.actor.parameters())   + \
                 list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg["lr"])

        self.buffer = RolloutBuffer()

    # ── 에피소드 수집 (v6 _collect_episode 구조와 동일) ──────────────────────

    def _collect_episode(self, seed: Optional[int] = None,
                         eval_mode: bool = False, eval_argmax: bool = False
                         ) -> Tuple[dict, bool, bool, str]:
        """1 에피소드 수집. eval_mode이면 buffer add 안 함, sampling.

        eval_argmax: True면 deterministic argmax (학습엔 사용 X, 보조 평가용)
        """
        from wm_upper_v2 import ENV_CFG as _ENV_CFG
        cfg       = self.cfg
        dev       = self.device
        gamma_env = cfg["gamma"]

        env = DroneEnv(config=_ENV_CFG, render_mode=None)
        env.reset(seed=seed)

        last_infos = {}
        cur_indices = np.zeros(self.n_agents, dtype=np.int64)
        cum_rew     = 0.0
        ep_reward   = 0.0   # 전체 에피소드 누적 보상 (undiscounted, mean across agents)
        step_in_K   = 0
        env_steps   = 0

        # K-step 시작 시 캐시
        cache_state    = None
        cache_indices  = None
        cache_log_prob = 0.0
        cache_value    = 0.0

        macro_count = 0

        while env.agents:
            obs = env.get_global_state()  # (205,)

            # ── K-step 경계: 새 서브골 결정 ──
            if step_in_K == 0:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                    feat  = self.encoder(obs_t)
                    indices_t, log_prob_t, _ = self.actor.act(feat, deterministic=eval_argmax)
                    value_t                   = self.critic(feat)
                    cur_indices  = indices_t.squeeze(0).cpu().numpy()
                    cache_state    = obs.copy()
                    cache_indices  = cur_indices.copy()
                    cache_log_prob = float(log_prob_t.item())
                    cache_value    = float(value_t.item())

                set_discrete_subgoals(env, cur_indices)
                cum_rew = 0.0

            # ── Lower 실행 ──
            lower_actions = rulebased_actions(env)
            _, rewards, terminations, truncations, infos = env.step(lower_actions)
            env_steps += 1
            if infos:
                last_infos = next(iter(infos.values()))

            step_rew = float(np.mean(list(rewards.values()))) if rewards else 0.0
            cum_rew  += (gamma_env ** step_in_K) * step_rew
            ep_reward += step_rew
            step_in_K += 1

            # ── 서브골 없는 에이전트 즉시 재할당 (K 주기 무관, v6와 동일) ──
            agents_no_subgoal = [a for a in env.agents
                                  if env._subgoals.get(a) is None]
            if agents_no_subgoal:
                with torch.no_grad():
                    obs_new = env.get_global_state()
                    obs_t   = torch.tensor(obs_new, dtype=torch.float32, device=dev).unsqueeze(0)
                    feat    = self.encoder(obs_t)
                    new_indices_t, _, _ = self.actor.act(feat, deterministic=eval_argmax)
                    new_indices = new_indices_t.squeeze(0).cpu().numpy()

                # v6와 동일: 부분 재할당 — 서브골 없는 agent만
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
                                    nearest = min(alive, key=lambda b: abs(b.x - base.x) + abs(b.y - base.y))
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

                if partial:
                    env.set_subgoals(partial)
                    # 부분 재할당 시 macro 종료 + 새 macro 시작 (v6 패턴)
                    if step_in_K > 0:
                        ep_done_now = (
                            all(terminations.values()) or all(truncations.values())
                            or bool(last_infos.get("won")) or bool(last_infos.get("lost"))
                        )
                        if not eval_mode:
                            self.buffer.add(
                                state=cache_state,
                                indices=cache_indices,
                                log_prob=cache_log_prob,
                                value=cache_value,
                                reward=cum_rew,
                                done=ep_done_now,
                                k_actual=max(1, step_in_K),
                            )
                        macro_count += 1
                        # 새 macro 시작: 현재 indices를 cache로 저장
                        cur_indices_new = new_indices.copy()
                        with torch.no_grad():
                            obs_t2 = torch.tensor(env.get_global_state(),
                                                  dtype=torch.float32, device=dev).unsqueeze(0)
                            feat2  = self.encoder(obs_t2)
                            log_prob_t2, _ = self.actor.evaluate(
                                feat2,
                                torch.tensor(cur_indices_new, dtype=torch.long,
                                             device=dev).unsqueeze(0)
                            )
                            value_t2 = self.critic(feat2)
                            cache_state    = env.get_global_state().copy()
                            cache_indices  = cur_indices_new
                            cache_log_prob = float(log_prob_t2.item())
                            cache_value    = float(value_t2.item())
                        cum_rew   = 0.0
                        step_in_K = 0

            ep_done = (
                all(terminations.values()) or all(truncations.values())
                or bool(last_infos.get("won")) or bool(last_infos.get("lost"))
            )

            # ── K-step 사이클 완료 또는 에피소드 종료 시 macro 기록 ──
            if step_in_K == K or ep_done:
                if not eval_mode:
                    self.buffer.add(
                        state=cache_state,
                        indices=cache_indices,
                        log_prob=cache_log_prob,
                        value=cache_value,
                        reward=cum_rew,
                        done=ep_done,
                        k_actual=max(1, step_in_K),
                    )
                macro_count += 1
                cum_rew   = 0.0
                step_in_K = 0

            if ep_done:
                break

        rsa = env._current_strategy
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
        elif drones_survived == 0:
            term_reason = "drones_exhausted"
        else:
            term_reason = "timeout"

        env.close()

        meta = {
            "env_steps":             env_steps,
            "macro_steps":           macro_count,
            "enemy_bases_destroyed": enemy_bases_destroyed,
            "ally_bases_destroyed":  ally_bases_destroyed,
            "drones_survived":       drones_survived,
            "termination_reason":    term_reason,
            "ep_reward":             ep_reward,   # episode 누적 보상 (undiscounted)
        }
        ep = {"meta": meta}
        return ep, won, lost, rsa

    # ── 평가 (학습용 sampling + argmax 보조) ─────────────────────────────

    def evaluate(self, n_episodes: int = 10, base_seed: int = 0,
                 use_argmax: bool = False) -> Dict[str, float]:
        wins = 0
        for i in range(n_episodes):
            _, won, _, _ = self._collect_episode(
                seed=base_seed + i, eval_mode=True, eval_argmax=use_argmax)
            if won:
                wins += 1
        return {"win_rate": wins / max(1, n_episodes)}

    # ── PPO 업데이트 ────────────────────────────────────────────────────────

    def _ppo_update(self) -> dict:
        cfg = self.cfg
        data = self.buffer.to_tensors(self.device, cfg["gamma"], cfg["gae_lambda"])
        N = data["states"].shape[0]
        idx_all = np.arange(N)

        losses = {"actor": [], "critic": [], "entropy": [], "clip_frac": [], "kl": []}
        for _ in range(cfg["n_epochs"]):
            np.random.shuffle(idx_all)
            for start in range(0, N, cfg["minibatch_size"]):
                mb = idx_all[start: start + cfg["minibatch_size"]]
                if len(mb) < 8:
                    continue
                mb_t = torch.tensor(mb, dtype=torch.long, device=self.device)
                feat = self.encoder(data["states"].index_select(0, mb_t))
                log_prob, entropy = self.actor.evaluate(
                    feat, data["indices"].index_select(0, mb_t))
                value_pred = self.critic(feat)

                ratio = torch.exp(log_prob - data["log_probs"].index_select(0, mb_t))
                adv   = data["advantages"].index_select(0, mb_t)
                ret   = data["returns"].index_select(0, mb_t)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (value_pred - ret).pow(2).mean()
                entropy_bonus = entropy.mean()
                loss = (actor_loss
                        + cfg["value_coef"]   * critic_loss
                        - cfg["entropy_coef"] * entropy_bonus)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for g in [self.encoder, self.actor, self.critic]
                     for p in g.parameters()],
                    cfg["grad_clip"])
                self.opt.step()

                with torch.no_grad():
                    kl = (data["log_probs"].index_select(0, mb_t) - log_prob).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()
                losses["actor"].append(float(actor_loss.item()))
                losses["critic"].append(float(critic_loss.item()))
                losses["entropy"].append(float(entropy_bonus.item()))
                losses["kl"].append(float(kl.item()))
                losses["clip_frac"].append(float(clip_frac.item()))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}

    # ── 저장 / 로드 ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
            "opt":     self.opt.state_dict(),
            "episode": self.episode_count,
            "macro":   self.macro_step_count,
            "env":     self.env_step_count,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ck["encoder"])
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.opt.load_state_dict(ck["opt"])
        self.episode_count    = ck.get("episode", 0)
        self.macro_step_count = ck.get("macro", 0)
        self.env_step_count   = ck.get("env", 0)

    # ── 메인 학습 루프 ──────────────────────────────────────────────────────

    def train(self, out_dir: str, seed: int = 0,
              use_wandb: bool = False, run_name: Optional[str] = None) -> None:
        cfg = self.cfg
        os.makedirs(out_dir, exist_ok=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        log_f  = open(os.path.join(out_dir, "log.csv"), "w", newline="")
        writer = csv.DictWriter(log_f, fieldnames=[
            "episode", "macro_step", "env_step",
            "win_rate", "ep_reward",
            "actor_loss", "critic_loss", "entropy", "kl", "clip_frac",
        ])
        writer.writeheader()

        ep_log_f = open(os.path.join(out_dir, "episodes.csv"), "w", newline="")
        ep_writer = csv.DictWriter(ep_log_f, fieldnames=[
            "episode", "macro_step", "env_step", "rsa", "result", "reward",
            "macro_length", "env_length",
            "termination_reason",
            "enemy_bases_destroyed", "ally_bases_destroyed", "drones_survived",
        ])
        ep_writer.writeheader()

        wandb_run = None
        if use_wandb:
            import wandb
            wandb_run = wandb.init(
                project = WANDB_PROJECT,
                entity  = WANDB_ENTITY,
                name    = run_name or f"ppo_hrl_s{seed}",
                group   = "PPO+HRL",
                config  = {**{k: v for k, v in cfg.items()}, "seed": seed, "K": K,
                           "model": "PPO+HRL"},
                reinit  = True,
            )

        print(f"\n{'='*55}")
        print(f"  PPO+HRL (no WM)  (seed={seed})")
        print(f"  총 episodes: {cfg['total_episodes']:,}  |  rollout: {cfg['rollout_macro_steps']} macro")
        print(f"  Action: 8x8 discrete + defender mask  |  out={out_dir}")
        print(f"{'='*55}")

        win_history: List[int] = []
        rew_history: List[float] = []
        latest_loss = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        t0 = time.time()

        while self.episode_count < cfg["total_episodes"]:
            ep_seed = seed * 1_000_000 + self.episode_count
            ep, won, lost, rsa = self._collect_episode(seed=ep_seed, eval_mode=False)
            self.episode_count += 1
            meta = ep["meta"]
            self.macro_step_count += int(meta.get("macro_steps", 0))
            self.env_step_count   += int(meta.get("env_steps", 0))
            win_history.append(int(won))
            rew_history.append(float(meta.get("ep_reward", 0.0)))

            # 에피소드 로그
            ep_writer.writerow({
                "episode":   self.episode_count,
                "macro_step": self.macro_step_count,
                "env_step":   self.env_step_count,
                "rsa":        rsa,
                "result":     "win" if won else "lost" if lost else "draw",
                "reward":     round(float(meta.get("ep_reward", 0.0)), 4),
                "macro_length": int(meta.get("macro_steps", 0)),
                "env_length":   int(meta.get("env_steps", 0)),
                "termination_reason":     meta.get("termination_reason", "unknown"),
                "enemy_bases_destroyed":  meta.get("enemy_bases_destroyed", 0),
                "ally_bases_destroyed":   meta.get("ally_bases_destroyed", 0),
                "drones_survived":        meta.get("drones_survived", 0),
            })
            ep_log_f.flush()

            # WandB 에피소드 로그 (정리: 핵심 지표만)
            if wandb_run:
                wandb_run.log({
                    "episode/env_length":           int(meta.get("env_steps", 0)),
                    "episode/reward":               float(meta.get("ep_reward", 0.0)),
                    "episode/enemy_bases_destroyed": meta.get("enemy_bases_destroyed", 0),
                    "episode/ally_bases_destroyed":  meta.get("ally_bases_destroyed", 0),
                    "episode/drones_survived":       meta.get("drones_survived", 0),
                }, commit=False)

            # ── PPO update: rollout이 충분히 쌓이면 ──
            if len(self.buffer) >= cfg["rollout_macro_steps"]:
                # bootstrap value (마지막 macro가 done이 아니면 마지막 state value 사용)
                # 단순화: 마지막 episode의 마지막 macro가 done이면 next_value=0
                # 이미 buffer.add에서 done flag 들어감, GAE에서 처리됨
                self.buffer.next_value = 0.0
                metrics = self._ppo_update()
                latest_loss = metrics
                self.buffer.clear()

            # ── 스텝 로그 ──
            if self.episode_count % cfg["log_every"] == 0:
                wr = float(np.mean(win_history[-100:]))
                rw = float(np.mean(rew_history[-100:])) if rew_history else 0.0
                elapsed = time.time() - t0
                print(f"  ep={self.episode_count:>6d} | macro={self.macro_step_count:>7d} | "
                      f"env={self.env_step_count:>8d} | wr={wr:.3f} | rew={rw:+.2f} | "
                      f"actor={latest_loss.get('actor', 0):+.4f} | "
                      f"critic={latest_loss.get('critic', 0):.4f} | "
                      f"H={latest_loss.get('entropy', 0):.3f} | "
                      f"kl={latest_loss.get('kl', 0):+.4f} | "
                      f"clip={latest_loss.get('clip_frac', 0):.3f} | "
                      f"({elapsed/60:.1f}m)")
                writer.writerow({
                    "episode":      self.episode_count,
                    "macro_step":   self.macro_step_count,
                    "env_step":     self.env_step_count,
                    "win_rate":     round(wr, 4),
                    "ep_reward":    round(rw, 4),
                    "actor_loss":   round(latest_loss.get("actor", 0), 6),
                    "critic_loss": round(latest_loss.get("critic", 0), 6),
                    "entropy":      round(latest_loss.get("entropy", 0), 6),
                    "kl":           round(latest_loss.get("kl", 0), 6),
                    "clip_frac":    round(latest_loss.get("clip_frac", 0), 6),
                })
                log_f.flush()
                if wandb_run:
                    wandb_run.log({
                        "win_rate":         wr,
                        "ep_reward_avg100": rw,
                        "ppo/actor_loss":   latest_loss.get("actor", 0),
                        "ppo/critic_loss":  latest_loss.get("critic", 0),
                        "ppo/entropy":      latest_loss.get("entropy", 0),
                        "ppo/kl":           latest_loss.get("kl", 0),
                        "ppo/clip_frac":    latest_loss.get("clip_frac", 0),
                    }, step=self.episode_count)

            # ── 평가 (정리: argmax 평가 제거 — sampling만)
            if self.episode_count % cfg["eval_every"] == 0:
                eval_smp = self.evaluate(n_episodes=cfg["eval_episodes"],
                                         base_seed=10**7, use_argmax=False)
                print(f"  [EVAL] ep={self.episode_count} | "
                      f"sampling_wr={eval_smp['win_rate']:.3f}")
                if wandb_run:
                    wandb_run.log({
                        "eval/win_rate": eval_smp["win_rate"],
                    }, step=self.episode_count)

            # ── 저장 ──
            if self.episode_count % cfg["save_every"] == 0:
                ckpt_p = os.path.join(out_dir, f"ppo_hrl_ep{self.episode_count:06d}.pt")
                self.save(ckpt_p)

        # ── 최종 평가 / 저장 ──
        final_smp = self.evaluate(n_episodes=cfg["final_eval_episodes"],
                                  base_seed=10**8, use_argmax=False)
        final_arg = self.evaluate(n_episodes=cfg["final_eval_episodes"],
                                  base_seed=10**8, use_argmax=True)
        print(f"\n  최종 win_rate (sampling, {cfg['final_eval_episodes']}ep): "
              f"{final_smp['win_rate']:.3f}")
        print(f"  최종 win_rate (argmax,   {cfg['final_eval_episodes']}ep): "
              f"{final_arg['win_rate']:.3f}")
        self.save(os.path.join(out_dir, "ppo_hrl_final.pt"))
        log_f.close(); ep_log_f.close()
        if wandb_run:
            wandb_run.summary["final_win_rate"]        = final_smp["win_rate"]
            wandb_run.summary["final_win_rate_argmax"] = final_arg["win_rate"]
            wandb_run.finish()
