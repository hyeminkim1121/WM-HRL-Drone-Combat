"""
env/drone_env.py
PettingZoo ParallelEnv 기반 드론 전투 시뮬레이션 환경 (3D 고도 지원).

에이전트: 아군 드론 8개 (공격 5 + 방어 3)
적 유닛 (내부 AI): 공격드론 8개 + 방어드론 5개 + 기지 5개
승리 조건: 적 기지 5개 전부 폭파
패배 조건: 아군 기지 5개 전부 폭파

Z축 설계:
    Z_MAX           = 20   최대 고도
    OBSTACLE_HEIGHT = 6    z <= 6 에서 장애물 차단 / z > 6 에서 자유 비행
    DRONE_INIT_Z    = 5    드론 초기 고도 (장애물 아래에서 출발)
    기지는 항상 z = 0 (지상)
    공격 범위(3) 기준: 기지(z=0) 공격하려면 z <= 3 으로 하강 필요
"""

from __future__ import annotations

import functools
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from env.map import GridMap
from env.reinforcement import ReinforcementManager
from env.units import Team, Unit, UnitType

# ---------------------------------------------------------------------------
# 3D 상수
# ---------------------------------------------------------------------------

GRID_SIZE       = 100
Z_MAX           = 20   # 최대 고도 (격자 단위)
OBSTACLE_HEIGHT = 6    # 장애물 높이: z <= 6 차단, z > 6 자유 비행
DRONE_INIT_Z    = 5    # 드론 초기 고도

# 액션 (6개 → 8개)
STAY        = 0
MOVE_UP     = 1   # y-1
MOVE_DOWN   = 2   # y+1
MOVE_LEFT   = 3   # x-1
MOVE_RIGHT  = 4   # x+1
MOVE_Z_UP   = 5   # z+1  고도 상승
MOVE_Z_DOWN = 6   # z-1  고도 하강
ATTACK      = 7

MOVE_DELTAS: Dict[int, Tuple[int, int, int]] = {
    STAY:       ( 0,  0,  0),
    MOVE_UP:    ( 0, -1,  0),
    MOVE_DOWN:  ( 0,  1,  0),
    MOVE_LEFT:  (-1,  0,  0),
    MOVE_RIGHT: ( 1,  0,  0),
    MOVE_Z_UP:  ( 0,  0,  1),
    MOVE_Z_DOWN:( 0,  0, -1),
}

MAX_ENEMY_ATTACK = 20

# OBS_SIZE 계산 (178 → 212)
# self          : x, y, z, hp, alive           = 5
# subgoal       : tx, ty                        = 2   (기지 z=0 고정, tz 생략)
# ally attack   : (x,y,z,hp,alive) × 5         = 25
# ally defend   : (x,y,z,hp,alive) × 3         = 15
# ally bases    : (x,y,hp,alive)   × 5         = 20  (z=0 고정 생략)
# enemy attack  : (x,y,z,hp,alive) × 20        = 100
# enemy defend  : (x,y,z,hp,alive) × 5         = 25
# enemy bases   : (x,y,hp,alive)   × 5         = 20  (z=0 고정 생략)
OBS_SIZE = 5 + 2 + 5*5 + 3*5 + 5*4 + MAX_ENEMY_ATTACK*5 + 5*5 + 5*4  # 212

R_STEP         = -0.001
R_KILL_DRONE   =  2.0
R_KILL_BASE    = 20.0
R_BASE_SHARED  =  5.0
R_DRONE_KILLED = -2.0
R_BASE_LOST    = -20.0
R_WIN          =  50.0
R_LOSE         = -50.0
R_DAMAGE_DRONE =  1.0   # 적 드론 데미지 비례 보상 계수
R_DAMAGE_BASE  =  2.0   # 적 기지 데미지 비례 보상 계수

# ---------------------------------------------------------------------------
# RSA (Random Strategy Adversary)
# ---------------------------------------------------------------------------
RSA_STRATEGIES      = ("A", "B", "C", "D", "M")
RSA_ACTIVATION_DIST = 35
RSA_PATROL_RADIUS   = 15

# ---------------------------------------------------------------------------
# 초기 배치 (x, y, z) — 드론, (x, y) — 기지
# ---------------------------------------------------------------------------
ALLY_ATTACK_INIT = [
    (14, 16, DRONE_INIT_Z), (14, 30, DRONE_INIT_Z), (14, 50, DRONE_INIT_Z),
    (14, 70, DRONE_INIT_Z), (14, 84, DRONE_INIT_Z),
]
ALLY_DEFEND_INIT = [
    (8, 36, DRONE_INIT_Z), (8, 50, DRONE_INIT_Z), (8, 64, DRONE_INIT_Z),
]
ALLY_BASE_INIT = [(4, 16), (4, 36), (4, 50), (4, 64), (4, 84)]
ALLY_BASE_HP   = [150, 120, 100, 120, 150]

ENEMY_ATTACK_INIT = [
    # v6 대칭 수정: 앞 5개 y={16,30,50,70,84} 아군 ALLY_ATTACK_INIT과 동일
    (86, 16, DRONE_INIT_Z), (86, 30, DRONE_INIT_Z),
    (86, 50, DRONE_INIT_Z), (86, 70, DRONE_INIT_Z),
    (86, 84, DRONE_INIT_Z),
    # 6-8번째: 기존 (reinforcement 용 슬롯)
    (86, 28, DRONE_INIT_Z), (86, 40, DRONE_INIT_Z), (86, 10, DRONE_INIT_Z),
]
ENEMY_DEFEND_INIT = [
    # v6 대칭 수정: 앞 3개 y={36,50,64} 아군 ALLY_DEFEND_INIT과 동일
    (92, 36, DRONE_INIT_Z), (92, 50, DRONE_INIT_Z), (92, 64, DRONE_INIT_Z),
    # 4-5번째: 기존 슬롯
    (92, 16, DRONE_INIT_Z), (92, 84, DRONE_INIT_Z),
]
ENEMY_BASE_INIT = [(96, 16), (96, 36), (96, 50), (96, 64), (96, 84)]
# 대칭: ALLY_BASE_HP와 동일 (config_v3에서 이미 override 중이지만 모듈 상수도 일관성)
ENEMY_BASE_HP   = [150, 120, 100, 120, 150]


# ── N·M parametric: 임의 유닛 수에 대해 위치를 y축 균등 배치로 생성 ──
def _spread_drone(x, n, z=DRONE_INIT_Z, y0=10, y1=90):
    """x 컬럼에 드론 n개를 y축 균등 배치 → [(x,y,z), ...]."""
    if n <= 0:  return []
    ys = [(y0 + y1) / 2] if n == 1 else [y0 + (y1 - y0) * i / (n - 1) for i in range(n)]
    return [(x, int(round(y)), z) for y in ys]

def _spread_base(x, n, y0=12, y1=88):
    """x 컬럼에 기지 n개를 y축 균등 배치 → [(x,y), ...]."""
    if n <= 0:  return []
    ys = [(y0 + y1) / 2] if n == 1 else [y0 + (y1 - y0) * i / (n - 1) for i in range(n)]
    return [(x, int(round(y))) for y in ys]


class DroneEnv(ParallelEnv):
    """드론 전투 시뮬레이션 (PettingZoo ParallelEnv, 3D 고도 지원).

    외부 인터페이스:
        env.set_subgoals(dict)   — 상위 에이전트가 매 K스텝마다 호출
        env.get_global_state()   — 상위 에이전트 입력용 글로벌 상태 벡터
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "drone_combat_v0"}

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None) -> None:
        cfg = config or {}
        self.grid_size           = cfg.get("grid_size",           GRID_SIZE)
        self.z_max               = cfg.get("z_max",               Z_MAX)
        self.obstacle_height     = cfg.get("obstacle_height",     OBSTACLE_HEIGHT)
        self.reinforce_interval  = cfg.get("reinforce_interval",  50)
        self.high_level_interval = cfg.get("high_level_interval", 20)
        self.max_steps           = cfg.get("max_episode_steps",   1000)
        self.enemy_strategy      = cfg.get("enemy_strategy",      "auto")
        self.mirror_greedy       = cfg.get("mirror_greedy",       False)

        # Reward config (defaults = module-level constants, backward compatible)
        self.R_STEP         = cfg.get("R_STEP",         R_STEP)
        self.R_KILL_DRONE   = cfg.get("R_KILL_DRONE",   R_KILL_DRONE)
        self.R_KILL_BASE    = cfg.get("R_KILL_BASE",    R_KILL_BASE)
        self.R_BASE_SHARED  = cfg.get("R_BASE_SHARED",  R_BASE_SHARED)
        self.R_DRONE_KILLED = cfg.get("R_DRONE_KILLED", R_DRONE_KILLED)
        self.R_BASE_LOST    = cfg.get("R_BASE_LOST",    R_BASE_LOST)
        self.R_WIN          = cfg.get("R_WIN",          R_WIN)
        self.R_LOSE         = cfg.get("R_LOSE",         R_LOSE)
        self.R_DAMAGE_DRONE = cfg.get("R_DAMAGE_DRONE", R_DAMAGE_DRONE)
        self.R_DAMAGE_BASE  = cfg.get("R_DAMAGE_BASE",  R_DAMAGE_BASE)
        self.n_enemy_attack = cfg.get("n_enemy_attack", 8)
        self.n_enemy_defend = cfg.get("n_enemy_defend", 5)
        self.enemy_base_hp  = cfg.get("enemy_base_hp", ENEMY_BASE_HP)
        # 적 유닛 스탯 (대칭 설정용)
        self.enemy_atk_hp    = cfg.get("enemy_atk_hp",    40)
        self.enemy_atk_range = cfg.get("enemy_atk_range",  3)
        self.enemy_atk_power = cfg.get("enemy_atk_power", 12)
        self.enemy_def_hp    = cfg.get("enemy_def_hp",    35)
        self.enemy_def_range = cfg.get("enemy_def_range",  2)
        self.enemy_def_power = cfg.get("enemy_def_power",  8)
        self.reinforce_enabled = cfg.get("reinforce_enabled", True)

        # ── 유닛 수 (N·M parametric) + 위치/HP 동적 생성 ──
        self.n_ally_attack = cfg.get("n_ally_attack", 5)
        self.n_ally_defend = cfg.get("n_ally_defend", 3)
        self.n_ally_base   = cfg.get("n_ally_base",   5)
        self.n_enemy_base  = cfg.get("n_enemy_base",  5)
        _bhp = cfg.get("ally_base_hp", 130)
        self._ally_attack_init  = _spread_drone(14, self.n_ally_attack)
        self._ally_defend_init  = _spread_drone(8,  self.n_ally_defend)
        self._ally_base_init    = _spread_base(4,   self.n_ally_base)
        self._ally_base_hp_gen  = [_bhp] * self.n_ally_base
        self._enemy_attack_init = _spread_drone(86, self.n_enemy_attack)
        self._enemy_defend_init = _spread_drone(92, self.n_enemy_defend)
        self._enemy_base_init   = _spread_base(96,  self.n_enemy_base)
        self.enemy_base_hp      = [_bhp] * self.n_enemy_base   # line~158 값 override (가변 수)
        # focus-fire: 기지에 동시 ATTACK하는 아군이 K명 미만이면 데미지 0 (1=비활성=기존)
        self.focus_fire_k       = cfg.get("focus_fire_k", 1)
        self.no_fallback        = cfg.get("no_fallback", False)
        # head-to-head 교전: 적을 외부 정책이 지휘 (mirror 자동배정 스킵, _enemy_subgoals 외부 설정)
        self.external_enemy     = cfg.get("external_enemy", False)
        # 하위 에이전트 obs 크기 (parametric; 우리 파이프라인에선 미사용이나 reset이 생성)
        self.obs_size = (5 + 2 + self.n_ally_attack * 5 + self.n_ally_defend * 5
                         + self.n_ally_base * 4 + self.n_enemy_attack * 5
                         + self.n_enemy_defend * 5 + self.n_enemy_base * 4)

        self._current_strategy: str             = "A"
        self._siege_target: Optional[Unit]      = None
        self._enemy_patrol_centers: List[Tuple] = []
        self._mirror_subgoal_timer: int         = 0   # Mirror RSA K-step 타이머
        self._enemy_subgoals:   Dict[str, Tuple[int, int]] = {}   # mirror: enemy atk → ally base pos
        self._enemy_atk_targets: Dict[str, str]             = {}  # mirror: enemy def → ally atk drone id
        self.render_mode                        = render_mode

        self.possible_agents: List[str] = (
            [f"ally_attack_{i}" for i in range(self.n_ally_attack)]
            + [f"ally_defend_{i}" for i in range(self.n_ally_defend)]
        )

        self.grid_map      = GridMap(self.grid_size)
        self.reinforce_mgr = ReinforcementManager(self.reinforce_interval, self.grid_size)

        self.agents:       List[str]                 = []
        self._step_count:  int                       = 0
        self._units:       Dict[str, Unit]           = {}
        self._ally_bases:  List[Unit]                = []
        self._enemy_units: List[Unit]                = []
        self._enemy_bases: List[Unit]                = []
        self._subgoals:       Dict[str, Tuple[int, int]] = {}
        self._defend_targets: Dict[str, str]             = {}  # defend agent → 추적 중인 적 드론 unit_id

    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(8)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._step_count      = 0
        self._defend_targets  = {}
        self._mirror_subgoal_timer = 0
        self.reinforce_mgr.reset()
        self._build_units()
        self.agents = self.possible_agents[:]
        self._init_subgoals()

        if self.enemy_strategy == "mirror":
            self._current_strategy = "M"
        elif self.enemy_strategy == "auto":
            self._current_strategy = random.choice(RSA_STRATEGIES)
        else:
            self._current_strategy = self.enemy_strategy

        self._enemy_patrol_centers = [
            (u.x, u.y, u.z) for u in self._enemy_units
            if u.unit_type == UnitType.DEFEND_DRONE
        ]
        self._siege_target = (
            random.choice(self._ally_bases)
            if self._current_strategy == "C" else None
        )

        # Mirror strategy: assign random subgoals to enemy drones
        self._enemy_subgoals   = {}
        self._enemy_atk_targets = {}
        if self._current_strategy == "M" and not self.external_enemy:
            self._init_enemy_mirror_subgoals()

        obs   = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        self._step_count += 1
        rewards = {a: self.R_STEP for a in self.agents}

        # First-mover 비대칭 제거: step마다 ally/enemy 처리 순서 교대
        # (짝수 step → ally first, 홀수 step → enemy first)
        # 평균적으로 양측 동등한 first-mover 빈도 → 통계적 대칭
        if self._step_count % 2 == 0:
            self._apply_ally_actions(actions, rewards)
            self._step_enemy_ai(rewards)
        else:
            self._step_enemy_ai(rewards)
            self._apply_ally_actions(actions, rewards)

        self._spawn_reinforcements()
        self._refresh_subgoals()

        enemy_bases_alive = sum(1 for b in self._enemy_bases if b.alive)
        ally_bases_alive  = sum(1 for b in self._ally_bases  if b.alive)

        won  = enemy_bases_alive == 0
        lost = ally_bases_alive  == 0

        # Tie-break (env-level termination):
        # 양측 공격드론 모두 사망 시 base 추가 파괴 불가능 → 즉시 종료
        # base 수가 더 많은 쪽 승, 같으면 draw로 종료
        tiebreak_done = False
        if not won and not lost:
            ally_atk_alive = any(
                self._units.get(f"ally_attack_{i}")
                and self._units[f"ally_attack_{i}"].alive
                for i in range(self.n_ally_attack)
            )
            enemy_atk_alive = any(
                u.alive for u in self._enemy_units
                if u.unit_type == UnitType.ATTACK_DRONE
            )
            if (not ally_atk_alive) and (not enemy_atk_alive):
                tiebreak_done = True
                if ally_bases_alive > enemy_bases_alive:
                    won = True
                elif enemy_bases_alive > ally_bases_alive:
                    lost = True
                # else: draw (won=False, lost=False, ep_done은 tiebreak_done로 강제)

        truncated_global = self._step_count >= self.max_steps

        terminations: Dict[str, bool] = {}
        truncations:  Dict[str, bool] = {}

        for agent in self.agents:
            unit    = self._units[agent]
            ep_done = won or lost or tiebreak_done
            terminations[agent] = (not unit.alive) or ep_done
            truncations[agent]  = truncated_global and not terminations[agent]
            if not unit.alive:
                rewards[agent] += self.R_DRONE_KILLED

        if won:
            for a in self.agents: rewards[a] += self.R_WIN
        if lost:
            for a in self.agents: rewards[a] += self.R_LOSE

        obs   = {a: self._obs(a) for a in self.agents}
        infos = {a: {
            "step":              self._step_count,
            "enemy_bases_alive": enemy_bases_alive,
            "ally_bases_alive":  ally_bases_alive,
            "won":               won,
            "lost":              lost,
            "tiebreak":          tiebreak_done,
        } for a in self.agents}

        self.agents = [
            a for a in self.agents
            if not terminations[a] and not truncations[a]
        ]
        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None
        from utils.visualize import render_env
        return render_env(self)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # High-level agent interface
    # ------------------------------------------------------------------

    def set_subgoals(self, subgoals: Dict[str, Tuple[int, int]]) -> None:
        """상위 에이전트 서브골 할당.

        v6: 공격드론 스냅 후보 = 적 기지 + 적 방어드론 통합 (action space 확장).
        공격드론: (tx,ty)에 가장 가까운 살아있는 적 기지 or 적 방어드론으로 스냅.
        방어드론: (tx,ty)에 가장 가까운 살아있는 적 공격드론으로 스냅 + unit_id 등록.
        """
        for aid, (tx, ty) in subgoals.items():
            unit = self._units.get(aid)
            if unit is None or not unit.alive:
                continue
            if unit.unit_type == UnitType.ATTACK_DRONE:
                alive_bases = [b for b in self._enemy_bases if b.alive]
                alive_defs  = [d for d in self._enemy_units
                               if d.alive and d.unit_type == UnitType.DEFEND_DRONE]
                candidates = alive_bases + alive_defs
                if candidates:
                    tgt = min(candidates, key=lambda c: abs(c.x - tx) + abs(c.y - ty))
                    self._subgoals[aid] = (tgt.x, tgt.y)
            else:  # DEFEND_DRONE
                alive_atk = [e for e in self._enemy_units
                             if e.alive and e.unit_type == UnitType.ATTACK_DRONE]
                if alive_atk:
                    tgt = min(alive_atk, key=lambda e: abs(e.x - tx) + abs(e.y - ty))
                    self._defend_targets[aid] = tgt.unit_id
                    self._subgoals[aid] = (tgt.x, tgt.y)

    def get_global_state(self) -> np.ndarray:
        """대칭 120-dim 글로벌 상태:
          - ally drones   8 × 5 = 40   (5 atk + 3 def, possible_agents 순서)
          - ally bases    5 × 4 = 20
          - enemy drones  8 × 5 = 40   (5 atk + 3 def, _enemy_units 생성 순서)
          - enemy bases   5 × 4 = 20
        """
        G  = float(self.grid_size)
        ZM = float(self.z_max)
        buf: List[float] = []

        # ally drones 8 × (x,y,z,hp,alive) = 40 (5 atk + 3 def)
        for aid in self.possible_agents:
            u = self._units.get(aid)
            if u and u.alive:
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # ally bases 5 × (x,y,hp,alive) = 20
        for b in self._ally_bases:
            buf.extend([b.x/G, b.y/G, b.hp/b.max_hp, float(b.alive)])

        # enemy drones 8 × (x,y,z,hp,alive) = 40 (5 atk + 3 def, ally와 mirror)
        ea = [u for u in self._enemy_units if u.unit_type == UnitType.ATTACK_DRONE]
        for i in range(self.n_enemy_attack):
            if i < len(ea) and ea[i].alive:
                u = ea[i]
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        ed = [u for u in self._enemy_units if u.unit_type == UnitType.DEFEND_DRONE]
        for i in range(self.n_enemy_defend):
            if i < len(ed) and ed[i].alive:
                u = ed[i]
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # enemy bases 5 × (x,y,hp,alive) = 20
        for b in self._enemy_bases:
            buf.extend([b.x/G, b.y/G, b.hp/b.max_hp, float(b.alive)])

        return np.array(buf, dtype=np.float32)  # 120-dim

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _build_units(self) -> None:
        units: Dict[str, Unit] = {}

        for i, (x, y, z) in enumerate(self._ally_attack_init):
            uid = f"ally_attack_{i}"
            units[uid] = Unit(uid, Team.ALLY, UnitType.ATTACK_DRONE,
                              x, y, z, max_hp=50, attack_range=3, attack_power=15)

        for i, (x, y, z) in enumerate(self._ally_defend_init):
            uid = f"ally_defend_{i}"
            units[uid] = Unit(uid, Team.ALLY, UnitType.DEFEND_DRONE,
                              x, y, z, max_hp=40, attack_range=2, attack_power=10)

        ally_bases = [
            Unit(f"ally_base_{i}", Team.ALLY, UnitType.BASE,
                 x, y, 0, max_hp=hp, attack_range=0, attack_power=0)
            for i, ((x, y), hp) in enumerate(zip(self._ally_base_init, self._ally_base_hp_gen))
        ]

        enemy_units: List[Unit] = [
            Unit(f"enemy_attack_{i}", Team.ENEMY, UnitType.ATTACK_DRONE,
                 x, y, z, max_hp=self.enemy_atk_hp,
                 attack_range=self.enemy_atk_range, attack_power=self.enemy_atk_power)
            for i, (x, y, z) in enumerate(self._enemy_attack_init)
        ]
        enemy_units += [
            Unit(f"enemy_defend_{i}", Team.ENEMY, UnitType.DEFEND_DRONE,
                 x, y, z, max_hp=self.enemy_def_hp,
                 attack_range=self.enemy_def_range, attack_power=self.enemy_def_power)
            for i, (x, y, z) in enumerate(self._enemy_defend_init)
        ]

        enemy_bases = [
            Unit(f"enemy_base_{i}", Team.ENEMY, UnitType.BASE,
                 x, y, 0, max_hp=hp, attack_range=0, attack_power=0)
            for i, ((x, y), hp) in enumerate(zip(self._enemy_base_init, self.enemy_base_hp))
        ]

        self._units       = units
        self._ally_bases  = ally_bases
        self._enemy_units = enemy_units
        self._enemy_bases = enemy_bases

    def _init_subgoals(self) -> None:
        """초기 서브골: 랜덤 배정 (상위 에이전트가 학습으로 결정할 영역).

        v6: 공격드론 초기 타겟 후보 = 살아있는 적 기지 + 살아있는 적 방어드론 (대칭).
        """
        alive_bases    = [b for b in self._enemy_bases if b.alive]
        alive_atk_enm  = [e for e in self._enemy_units
                          if e.alive and e.unit_type == UnitType.ATTACK_DRONE]
        alive_def_enm  = [e for e in self._enemy_units
                          if e.alive and e.unit_type == UnitType.DEFEND_DRONE]
        atk_targets = alive_bases + alive_def_enm
        for aid in self.agents:
            unit = self._units[aid]
            if unit.unit_type == UnitType.ATTACK_DRONE:
                if atk_targets:
                    t = random.choice(atk_targets)
                    self._subgoals[aid] = (t.x, t.y)
            else:  # DEFEND_DRONE
                if alive_atk_enm:
                    e = random.choice(alive_atk_enm)
                    self._defend_targets[aid] = e.unit_id
                    self._subgoals[aid] = (e.x, e.y)

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def _apply_ally_actions(self, actions: Dict[str, int], rewards: Dict[str, float]) -> None:
        # 1st pass: 이동 처리 + ATTACK 후보 수집 (focus-fire 집계용)
        attacks = []   # (aid, unit, target)
        for aid, action in actions.items():
            unit = self._units.get(aid)
            if unit is None or not unit.alive:
                continue
            if action in MOVE_DELTAS:
                dx, dy, dz = MOVE_DELTAS[action]
                unit.move(dx, dy, dz, self.grid_size, self.z_max,
                          self.grid_map.obstacles, self.obstacle_height)
            elif action == ATTACK:
                target = self._find_attack_target(unit, aid)
                if target is not None:
                    attacks.append((aid, unit, target))

        # focus-fire: 같은 기지를 동시에 ATTACK하는 아군 수가 K 미만이면 데미지 흡수(방어막)
        if self.focus_fire_k > 1:
            base_hits: Dict[int, int] = {}
            for _, _, target in attacks:
                if target.unit_type == UnitType.BASE:
                    base_hits[id(target)] = base_hits.get(id(target), 0) + 1

        # 2nd pass: 데미지 적용
        for aid, unit, target in attacks:
            if (self.focus_fire_k > 1 and target.unit_type == UnitType.BASE
                    and base_hits.get(id(target), 0) < self.focus_fire_k):
                continue   # 집중도 부족 → 데미지 0 (방어막 흡수)
            actual_dmg = min(unit.attack_power, target.hp)
            killed = target.take_damage(unit.attack_power)
            if target.unit_type == UnitType.BASE:
                rewards[aid] += self.R_DAMAGE_BASE * actual_dmg / target.max_hp
                if killed:
                    rewards[aid] += self.R_KILL_DRONE
                    for a in self.agents:
                        rewards[a] += self.R_KILL_BASE
            else:
                rewards[aid] += self.R_DAMAGE_DRONE * actual_dmg / target.max_hp
                if killed:
                    rewards[aid] += self.R_KILL_DRONE

    def _find_attack_target(self, unit: Unit, agent_id: str) -> Optional[Unit]:
        """서브골에 할당된 타겟만 공격 (상위 에이전트 서브골 엄수).

        v6: 공격드론은 적 기지 + 적 방어드론 둘 다 공격 가능 (action space 확장).
        """
        if unit.unit_type == UnitType.ATTACK_DRONE:
            subgoal = self._subgoals.get(agent_id)
            if subgoal is None:
                return None
            tx, ty = subgoal
            # 1순위: 적 기지
            target = next(
                (b for b in self._enemy_bases if b.alive and b.x == tx and b.y == ty),
                None,
            )
            if target is not None and unit.chebyshev(target) <= unit.attack_range:
                return target
            # 2순위: 적 방어드론 (v6 신규)
            target = next(
                (d for d in self._enemy_units
                 if d.alive and d.unit_type == UnitType.DEFEND_DRONE
                 and d.x == tx and d.y == ty),
                None,
            )
            if target is not None and unit.chebyshev(target) <= unit.attack_range:
                return target
            return None
        else:
            # 할당된 적 공격드론만 공격
            target_id = self._defend_targets.get(agent_id)
            if target_id is None:
                return None
            target = next(
                (u for u in self._enemy_units if u.unit_id == target_id and u.alive),
                None,
            )
            if target is not None and unit.chebyshev(target) <= unit.attack_range:
                return target
            return None

    # ------------------------------------------------------------------
    # RSA (Random Strategy Adversary)
    # ------------------------------------------------------------------

    def _step_enemy_ai(self, rewards: Dict[str, float]) -> None:
        s = self._current_strategy
        if   s == "A": self._rsa_rush(rewards)
        elif s == "B": self._rsa_turtle(rewards)
        elif s == "C": self._rsa_siege(rewards)
        elif s == "M": self._rsa_mirror(rewards)
        else:          self._rsa_random(rewards)

    def _rsa_rush(self, rewards: Dict[str, float]) -> None:
        ally_bases  = [b for b in self._ally_bases  if b.alive]
        ally_drones = [u for u in self._units.values() if u.alive]

        for enemy in self._enemy_units:
            if not enemy.alive:
                continue
            if enemy.unit_type == UnitType.ATTACK_DRONE:
                nearest = self._nearest_unit(enemy, ally_bases)
                if nearest is None:
                    continue
                if enemy.chebyshev(nearest) <= enemy.attack_range:
                    if nearest.take_damage(enemy.attack_power):
                        for a in rewards: rewards[a] += self.R_BASE_LOST
                else:
                    self._move_toward(enemy, nearest)
            else:
                nearest = self._nearest_unit(enemy, ally_drones)
                if nearest is None:
                    continue
                if enemy.chebyshev(nearest) <= enemy.attack_range:
                    nearest.take_damage(enemy.attack_power)
                else:
                    self._move_toward(enemy, nearest)

    def _rsa_turtle(self, rewards: Dict[str, float]) -> None:
        ally_drones = [u for u in self._units.values() if u.alive]
        ally_bases  = [b for b in self._ally_bases  if b.alive]

        for enemy in self._enemy_units:
            if not enemy.alive:
                continue
            if enemy.unit_type == UnitType.ATTACK_DRONE:
                nearest_ally = self._nearest_unit(enemy, ally_drones)
                activated = (nearest_ally is not None
                             and enemy.manhattan(nearest_ally) <= RSA_ACTIVATION_DIST)
                if activated:
                    nearest_base = self._nearest_unit(enemy, ally_bases)
                    if nearest_base is None:
                        continue
                    if enemy.chebyshev(nearest_base) <= enemy.attack_range:
                        if nearest_base.take_damage(enemy.attack_power):
                            for a in rewards: rewards[a] += self.R_BASE_LOST
                    else:
                        self._move_toward(enemy, nearest_base)

        defend_enemies = [u for u in self._enemy_units if u.unit_type == UnitType.DEFEND_DRONE]
        for enemy, (cx, cy, cz) in zip(defend_enemies, self._enemy_patrol_centers):
            if not enemy.alive:
                continue
            nearby = [u for u in ally_drones
                      if abs(u.x - cx) + abs(u.y - cy) <= RSA_PATROL_RADIUS]
            if nearby:
                target = min(nearby, key=lambda u: enemy.chebyshev(u))
                if enemy.chebyshev(target) <= enemy.attack_range:
                    target.take_damage(enemy.attack_power)
                else:
                    self._move_toward(enemy, target)
            else:
                self._move_to_pos(enemy, cx, cy, cz)

    def _rsa_siege(self, rewards: Dict[str, float]) -> None:
        if self._siege_target is None or not self._siege_target.alive:
            alive_bases = [b for b in self._ally_bases if b.alive]
            self._siege_target = random.choice(alive_bases) if alive_bases else None

        ally_drones = [u for u in self._units.values() if u.alive]

        for enemy in self._enemy_units:
            if not enemy.alive:
                continue
            if enemy.unit_type == UnitType.ATTACK_DRONE:
                target = self._siege_target
                if target is None:
                    continue
                if enemy.chebyshev(target) <= enemy.attack_range:
                    if target.take_damage(enemy.attack_power):
                        for a in rewards: rewards[a] += self.R_BASE_LOST
                else:
                    self._move_toward(enemy, target)
            else:
                in_range = self._nearest_in_range(enemy, ally_drones, enemy.attack_range)
                if in_range is not None:
                    in_range.take_damage(enemy.attack_power)
                elif self._siege_target is not None:
                    self._move_toward(enemy, self._siege_target)

    def _rsa_random(self, rewards: Dict[str, float]) -> None:
        ally_bases  = [b for b in self._ally_bases  if b.alive]
        ally_drones = [u for u in self._units.values() if u.alive]

        for enemy in self._enemy_units:
            if not enemy.alive:
                continue
            action = random.randint(0, 7)
            if action in MOVE_DELTAS:
                dx, dy, dz = MOVE_DELTAS[action]
                enemy.move(dx, dy, dz, self.grid_size, self.z_max,
                           self.grid_map.obstacles, self.obstacle_height)
            elif action == ATTACK:
                if enemy.unit_type == UnitType.ATTACK_DRONE:
                    target = self._nearest_in_range(enemy, ally_bases, enemy.attack_range)
                    if target is not None:
                        if target.take_damage(enemy.attack_power):
                            for a in rewards: rewards[a] += self.R_BASE_LOST
                else:
                    target = self._nearest_in_range(enemy, ally_drones, enemy.attack_range)
                    if target is not None:
                        target.take_damage(enemy.attack_power)

    # ------------------------------------------------------------------
    # Mirror strategy helpers
    # ------------------------------------------------------------------

    def _init_enemy_mirror_subgoals(self) -> None:
        """Assign symmetric-mirrored subgoals to enemy drones.

        Symmetric to ally's set_discrete_subgoals + env.set_subgoals:
          - Enemy attackers: random target_idx [0,8) → ally_base[idx] or ally_def[idx-5]
            (alive 우선, fallback 가장 가까운 alive)
          - Enemy defenders: random zone_idx [0,5) → 해당 enemy_base 좌표 기준
            가장 가까운 alive ally attacker (zone defense)
        """
        alive_ally_bases = [b for b in self._ally_bases if b.alive]
        alive_ally_atk   = [self._units[f"ally_attack_{i}"]
                            for i in range(self.n_ally_attack)
                            if self._units.get(f"ally_attack_{i}")
                            and self._units[f"ally_attack_{i}"].alive]
        alive_ally_defs  = [self._units[f"ally_defend_{i}"]
                            for i in range(self.n_ally_defend)
                            if self._units.get(f"ally_defend_{i}")
                            and self._units[f"ally_defend_{i}"].alive]

        # Enemy attackers: greedy = nearest alive base (or defender), random = random idx
        for enemy in self._enemy_units:
            if not enemy.alive or enemy.unit_type != UnitType.ATTACK_DRONE:
                continue
            tgt = None
            if self.mirror_greedy:
                candidates = alive_ally_bases + alive_ally_defs
                if candidates:
                    tgt = min(candidates,
                              key=lambda u: abs(u.x - enemy.x) + abs(u.y - enemy.y))
            else:
                target_idx = random.randint(0, self.n_ally_base + self.n_ally_defend - 1)
                if target_idx < self.n_ally_base:
                    if target_idx < len(self._ally_bases) and self._ally_bases[target_idx].alive:
                        tgt = self._ally_bases[target_idx]
                    elif alive_ally_bases:
                        tgt = min(alive_ally_bases,
                                  key=lambda b: abs(b.x - enemy.x) + abs(b.y - enemy.y))
                else:
                    def_idx = target_idx - self.n_ally_base
                    if def_idx < len(alive_ally_defs):
                        tgt = alive_ally_defs[def_idx]
                    elif alive_ally_defs:
                        tgt = min(alive_ally_defs,
                                  key=lambda d: abs(d.x - enemy.x) + abs(d.y - enemy.y))
                    elif alive_ally_bases:
                        tgt = min(alive_ally_bases,
                                  key=lambda b: abs(b.x - enemy.x) + abs(b.y - enemy.y))
            if tgt is not None:
                self._enemy_subgoals[enemy.unit_id] = (tgt.x, tgt.y)

        # Enemy defenders: greedy = globally nearest ally attacker, random = zone-based
        if alive_ally_atk:
            for enemy in self._enemy_units:
                if not enemy.alive or enemy.unit_type != UnitType.DEFEND_DRONE:
                    continue
                if self.mirror_greedy:
                    tgt = min(alive_ally_atk,
                              key=lambda e: abs(e.x - enemy.x) + abs(e.y - enemy.y))
                else:
                    zone_idx = random.randint(0, self.n_enemy_base - 1)
                    zone_base = self._enemy_bases[zone_idx]
                    tx, ty = zone_base.x, zone_base.y
                    tgt = min(alive_ally_atk,
                              key=lambda e: abs(e.x - tx) + abs(e.y - ty))
                self._enemy_atk_targets[enemy.unit_id] = tgt.unit_id
                self._enemy_subgoals[enemy.unit_id] = (tgt.x, tgt.y)

    def _rsa_mirror(self, rewards: Dict[str, float]) -> None:
        """Mirror strategy: enemy uses rulebased 3-phase navigation with random subgoals.

        아군과 동일하게 K스텝마다 랜덤 서브골 재할당.
        K 주기 사이에는 기존 서브골 따라 이동 (타겟 사망 시에만 즉시 재할당).
        """
        APPROACH_DIST = 12

        # K스텝마다 전체 서브골 랜덤 재할당
        self._mirror_subgoal_timer += 1
        if self._mirror_subgoal_timer >= self.high_level_interval:
            if not self.external_enemy:
                self._init_enemy_mirror_subgoals()
            self._mirror_subgoal_timer = 0

        # focus-fire 대칭: 아군 기지에 사정거리 내 적 공격드론 K명 미만이면 데미지 흡수
        self._ff_vuln_ally = set()
        if self.focus_fire_k > 1:
            for b in self._ally_bases:
                if not b.alive:
                    continue
                cnt = sum(1 for e in self._enemy_units
                          if e.alive and e.unit_type == UnitType.ATTACK_DRONE
                          and e.chebyshev(b) <= e.attack_range)
                if cnt >= self.focus_fire_k:
                    self._ff_vuln_ally.add(id(b))

        # 타겟 사망 시에만 즉시 재할당 (K 주기 무관)
        alive_ally_bases = [b for b in self._ally_bases if b.alive]
        alive_ally_atk   = [self._units[f"ally_attack_{i}"]
                            for i in range(self.n_ally_attack)
                            if self._units.get(f"ally_attack_{i}")
                            and self._units[f"ally_attack_{i}"].alive]
        alive_ally_defs  = [self._units[f"ally_defend_{i}"]
                            for i in range(self.n_ally_defend)
                            if self._units.get(f"ally_defend_{i}")
                            and self._units[f"ally_defend_{i}"].alive]

        for enemy in self._enemy_units:
            if not enemy.alive:
                continue
            eid = enemy.unit_id

            if enemy.unit_type == UnitType.ATTACK_DRONE:
                # v6: 타겟은 아군 기지 OR 아군 방어드론 (alive 체크 엄격, codex 지적 fix)
                sg = self._enemy_subgoals.get(eid)
                if sg is not None:
                    tx, ty = sg
                    target_alive = any(b for b in self._ally_bases
                                       if b.x == tx and b.y == ty and b.alive)
                    if not target_alive:
                        target_alive = any(d for d in alive_ally_defs
                                           if d.alive and d.x == tx and d.y == ty)
                    if not target_alive:
                        sg = None  # need re-assign
                if sg is None:
                    candidates = alive_ally_bases + alive_ally_defs
                    if candidates:
                        if self.mirror_greedy:
                            tgt = min(candidates,
                                      key=lambda u: abs(u.x - enemy.x) + abs(u.y - enemy.y))
                        else:
                            tgt = random.choice(candidates)
                        self._enemy_subgoals[eid] = (tgt.x, tgt.y)
                    else:
                        continue  # no targets left

                tx, ty = self._enemy_subgoals[eid]

                # 타겟 찾기: 기지 우선, 그 다음 방어드론 (alive 체크 엄격)
                target = next(
                    (b for b in self._ally_bases if b.alive and b.x == tx and b.y == ty),
                    None,
                )
                if target is None:
                    target = next(
                        (d for d in alive_ally_defs if d.alive and d.x == tx and d.y == ty),
                        None,
                    )

                # Phase 0: In attack range → attack
                if (target is not None
                        and enemy.chebyshev(target) <= enemy.attack_range):
                    if (self.focus_fire_k > 1 and target.unit_type == UnitType.BASE
                            and id(target) not in self._ff_vuln_ally):
                        continue   # 집중도 부족 → 데미지 0 (방어막)
                    killed = target.take_damage(enemy.attack_power)
                    if target.unit_type == UnitType.BASE and killed:
                        for a in rewards:
                            rewards[a] += self.R_BASE_LOST
                    # 방어드론 kill 시 R_DRONE_KILLED는 step() 말미에서 자동 적용
                    continue

                # 3-phase navigation
                dx = tx - enemy.x
                dy = ty - enemy.y
                xy_dist = abs(dx) + abs(dy)

                if xy_dist > APPROACH_DIST:
                    # Phase 1: Far — go up if below obstacles, else move horizontally
                    if enemy.z <= self.obstacle_height:
                        enemy.move(0, 0, 1, self.grid_size, self.z_max,
                                   self.grid_map.obstacles, self.obstacle_height)
                    else:
                        if abs(dx) >= abs(dy):
                            mx = 1 if dx > 0 else -1
                            enemy.move(mx, 0, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                        else:
                            my = 1 if dy > 0 else -1
                            enemy.move(0, my, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                else:
                    # Phase 2: Close — descend then approach
                    if enemy.z > 3:
                        enemy.move(0, 0, -1, self.grid_size, self.z_max,
                                   self.grid_map.obstacles, self.obstacle_height)
                    elif dx == 0 and dy == 0:
                        # Directly above target, just descend
                        if enemy.z > 0:
                            enemy.move(0, 0, -1, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                    else:
                        if abs(dx) >= abs(dy):
                            mx = 1 if dx > 0 else -1
                            enemy.move(mx, 0, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                        else:
                            my = 1 if dy > 0 else -1
                            enemy.move(0, my, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)

            elif enemy.unit_type == UnitType.DEFEND_DRONE:
                # Check if tracked ally attack drone still alive
                target_id = self._enemy_atk_targets.get(eid)
                target_unit = None
                if target_id:
                    target_unit = self._units.get(target_id)
                    if target_unit is None or not target_unit.alive:
                        target_unit = None
                        self._enemy_atk_targets.pop(eid, None)

                if target_unit is None:
                    # Re-assign: greedy = globally nearest, random = zone-based nearest
                    if alive_ally_atk:
                        if self.mirror_greedy:
                            tgt = min(alive_ally_atk,
                                      key=lambda e: abs(e.x - enemy.x) + abs(e.y - enemy.y))
                        else:
                            zone_idx = random.randint(0, self.n_enemy_base - 1)
                            zone_base = self._enemy_bases[zone_idx]
                            tx, ty = zone_base.x, zone_base.y
                            tgt = min(alive_ally_atk,
                                      key=lambda e: abs(e.x - tx) + abs(e.y - ty))
                        self._enemy_atk_targets[eid] = tgt.unit_id
                        target_unit = tgt
                    else:
                        continue

                # Update subgoal to target's current position
                self._enemy_subgoals[eid] = (target_unit.x, target_unit.y)

                # Phase 0: In attack range → attack
                if enemy.chebyshev(target_unit) <= enemy.attack_range:
                    target_unit.take_damage(enemy.attack_power)
                    continue

                # Navigate toward target
                dx = target_unit.x - enemy.x
                dy = target_unit.y - enemy.y
                xy_dist = abs(dx) + abs(dy)

                if xy_dist > APPROACH_DIST:
                    if enemy.z <= self.obstacle_height:
                        enemy.move(0, 0, 1, self.grid_size, self.z_max,
                                   self.grid_map.obstacles, self.obstacle_height)
                    else:
                        if abs(dx) >= abs(dy):
                            mx = 1 if dx > 0 else -1
                            enemy.move(mx, 0, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                        else:
                            my = 1 if dy > 0 else -1
                            enemy.move(0, my, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                else:
                    # Match altitude to target, then approach
                    dz = target_unit.z - enemy.z
                    if abs(dz) > 1:
                        mz = 1 if dz > 0 else -1
                        enemy.move(0, 0, mz, self.grid_size, self.z_max,
                                   self.grid_map.obstacles, self.obstacle_height)
                    elif dx == 0 and dy == 0:
                        pass  # on top of target
                    else:
                        if abs(dx) >= abs(dy):
                            mx = 1 if dx > 0 else -1
                            enemy.move(mx, 0, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)
                        else:
                            my = 1 if dy > 0 else -1
                            enemy.move(0, my, 0, self.grid_size, self.z_max,
                                       self.grid_map.obstacles, self.obstacle_height)

    # ------------------------------------------------------------------
    # 이동 헬퍼 (3D)
    # ------------------------------------------------------------------

    def _move_toward(self, unit: Unit, target: Unit) -> None:
        """target 방향으로 3D 이동 (장애물 차단 시 고도 상승으로 우회)."""
        dx = int(np.sign(target.x - unit.x))
        dy = int(np.sign(target.y - unit.y))
        dz = int(np.sign(target.z - unit.z))

        if unit.move(dx, dy, dz, self.grid_size, self.z_max,
                     self.grid_map.obstacles, self.obstacle_height):
            return
        if unit.move(dx, dy, 0, self.grid_size, self.z_max,
                     self.grid_map.obstacles, self.obstacle_height):
            return
        # 장애물에 막힌 경우: 고도 올려 장애물 위로 우회
        nx, ny = unit.x + dx, unit.y + dy
        if (nx, ny) in self.grid_map.obstacles and unit.z <= self.obstacle_height:
            if unit.move(0, 0, 1, self.grid_size, self.z_max,
                         self.grid_map.obstacles, self.obstacle_height):
                return
        if not unit.move(dx, 0, dz, self.grid_size, self.z_max,
                         self.grid_map.obstacles, self.obstacle_height):
            unit.move(0, dy, dz, self.grid_size, self.z_max,
                      self.grid_map.obstacles, self.obstacle_height)

    def _move_to_pos(self, unit: Unit, tx: int, ty: int, tz: int = 0) -> None:
        """(tx, ty, tz) 방향으로 3D 이동."""
        dx = int(np.sign(tx - unit.x))
        dy = int(np.sign(ty - unit.y))
        dz = int(np.sign(tz - unit.z))

        if unit.move(dx, dy, dz, self.grid_size, self.z_max,
                     self.grid_map.obstacles, self.obstacle_height):
            return
        if unit.move(dx, dy, 0, self.grid_size, self.z_max,
                     self.grid_map.obstacles, self.obstacle_height):
            return
        nx, ny = unit.x + dx, unit.y + dy
        if (nx, ny) in self.grid_map.obstacles and unit.z <= self.obstacle_height:
            if unit.move(0, 0, 1, self.grid_size, self.z_max,
                         self.grid_map.obstacles, self.obstacle_height):
                return
        if not unit.move(dx, 0, dz, self.grid_size, self.z_max,
                         self.grid_map.obstacles, self.obstacle_height):
            unit.move(0, dy, dz, self.grid_size, self.z_max,
                      self.grid_map.obstacles, self.obstacle_height)

    # ------------------------------------------------------------------
    # Reinforcements
    # ------------------------------------------------------------------

    def _spawn_reinforcements(self) -> None:
        if not self.reinforce_enabled:
            return
        specs = self.reinforce_mgr.get_reinforcements(
            self._step_count, self.grid_map.obstacles
        )
        for spec in specs:
            self._enemy_units.append(
                Unit(spec["unit_id"], Team.ENEMY, UnitType.ATTACK_DRONE,
                     spec["x"], spec["y"], DRONE_INIT_Z,
                     max_hp=spec["max_hp"],
                     attack_range=spec["attack_range"],
                     attack_power=spec["attack_power"])
            )

    # ------------------------------------------------------------------
    # Subgoals
    # ------------------------------------------------------------------

    def _refresh_subgoals(self) -> None:
        """매 스텝 호출.
        공격드론: 타겟 기지/방어드론 파괴 시 서브골 클리어 (상위 에이전트가 재할당).
        방어드론: 추적 중인 적 드론의 현재 좌표로 서브골 갱신, 드론 사망 시 클리어.
        """
        for aid in self.agents:
            unit = self._units.get(aid)
            if unit is None:
                continue
            if unit.unit_type == UnitType.ATTACK_DRONE:
                subgoal = self._subgoals.get(aid)
                if subgoal is not None:
                    tx, ty = subgoal
                    # v6: 좌표에 살아있는 기지 OR 살아있는 방어드론 있는지 확인
                    alive_at_coord = any(
                        b.alive for b in self._enemy_bases
                        if b.x == tx and b.y == ty
                    )
                    if not alive_at_coord:
                        alive_at_coord = any(
                            d.alive for d in self._enemy_units
                            if d.unit_type == UnitType.DEFEND_DRONE
                            and d.x == tx and d.y == ty
                        )
                    if not alive_at_coord:
                        self._subgoals.pop(aid, None)
            else:  # DEFEND_DRONE
                target_id = self._defend_targets.get(aid)
                if target_id:
                    tgt = next((e for e in self._enemy_units if e.unit_id == target_id), None)
                    if tgt and tgt.alive:
                        self._subgoals[aid] = (tgt.x, tgt.y)   # 현재 위치로 매 스텝 갱신
                    else:
                        self._defend_targets.pop(aid, None)
                        self._subgoals.pop(aid, None)

    def _nearest_enemy_base(self, x: int, y: int) -> Optional[Unit]:
        alive = [b for b in self._enemy_bases if b.alive]
        if not alive:
            return None
        return min(alive, key=lambda b: abs(b.x - x) + abs(b.y - y))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs(self, agent_id: str) -> np.ndarray:
        G  = float(self.grid_size)
        ZM = float(self.z_max)
        buf: List[float] = []
        unit = self._units[agent_id]

        # self (5)
        buf.extend([unit.x/G, unit.y/G, unit.z/ZM, unit.hp/unit.max_hp, float(unit.alive)])

        # subgoal (2)
        tx, ty = self._subgoals.get(agent_id, (G-2, G/2))
        buf.extend([tx/G, ty/G])

        # ally attack drones n×5
        for i in range(self.n_ally_attack):
            u = self._units.get(f"ally_attack_{i}")
            if u and u.alive:
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # ally defend drones n×5
        for i in range(self.n_ally_defend):
            u = self._units.get(f"ally_defend_{i}")
            if u and u.alive:
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # ally bases 5×4 = 20 (z=0 고정 생략)
        for b in self._ally_bases:
            if b.alive:
                buf.extend([b.x/G, b.y/G, b.hp/b.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0])

        # enemy attack drones n×5
        ea = [u for u in self._enemy_units if u.unit_type == UnitType.ATTACK_DRONE]
        for i in range(self.n_enemy_attack):
            if i < len(ea) and ea[i].alive:
                u = ea[i]
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # enemy defend drones n×5
        ed = [u for u in self._enemy_units if u.unit_type == UnitType.DEFEND_DRONE]
        for i in range(self.n_enemy_defend):
            if i < len(ed) and ed[i].alive:
                u = ed[i]
                buf.extend([u.x/G, u.y/G, u.z/ZM, u.hp/u.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # enemy bases 5×4 = 20 (z=0 고정 생략)
        for b in self._enemy_bases:
            if b.alive:
                buf.extend([b.x/G, b.y/G, b.hp/b.max_hp, 1.0])
            else:
                buf.extend([0.0, 0.0, 0.0, 0.0])

        arr = np.array(buf, dtype=np.float32)
        assert len(arr) == self.obs_size, f"obs_size mismatch: {len(arr)} != {self.obs_size}"
        return arr

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_unit(ref: Unit, candidates: List[Unit]) -> Optional[Unit]:
        alive = [u for u in candidates if u.alive]
        if not alive:
            return None
        return min(alive, key=lambda u: ref.manhattan(u))

    @staticmethod
    def _nearest_in_range(ref: Unit, candidates: List[Unit], max_range: int) -> Optional[Unit]:
        in_range = [u for u in candidates if u.alive and ref.chebyshev(u) <= max_range]
        if not in_range:
            return None
        return min(in_range, key=lambda u: ref.chebyshev(u))
