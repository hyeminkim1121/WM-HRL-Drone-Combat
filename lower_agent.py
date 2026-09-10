"""
lower_agent.py — 룰베이스 하위 에이전트 (상위 에이전트 서브골 완전 추종)

Stage 1 가정: 하위 에이전트는 상위 에이전트 서브골을 최적으로 수행.
  - ATTACK_DRONE : 상위 에이전트가 지정한 서브골(적 기지)로 이동 → 사정거리 내 공격
  - DEFEND_DRONE : 상위 에이전트가 지정한 서브골(방어 위치)로 이동 → 사정거리 내 적 드론 공격

서브골 미지정 시 fallback:
  - ATTACK_DRONE : 가장 가까운 적 기지
  - DEFEND_DRONE : 현재 위치 유지

나중에 MAPPO 하위 에이전트 학습 시 이 에이전트가 달성하는 성능을 기준선으로 사용.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict


def rulebased_actions(env) -> Dict[str, int]:
    from env.drone_env import (
        STAY, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
        MOVE_Z_UP, MOVE_Z_DOWN, ATTACK, OBSTACLE_HEIGHT,
    )
    from env.units import UnitType

    APPROACH_DIST = 12   # 이 거리 이하면 하강하며 공격 준비

    actions: Dict[str, int] = {}

    for agent in env.agents:
        unit = env._units.get(agent)
        if unit is None or not unit.alive:
            actions[agent] = STAY
            continue

        # ── 서브골 없으면 대기 (상위 에이전트가 다음 K 스텝에 재할당) ──
        subgoal = env._subgoals.get(agent)
        if subgoal is None:
            actions[agent] = STAY
            continue

        tx, ty = subgoal
        dx = tx - unit.x
        dy = ty - unit.y

        if unit.unit_type == UnitType.ATTACK_DRONE:
            # v6: 서브골 좌표는 적 기지 OR 적 방어드론 좌표
            target = next(
                (b for b in env._enemy_bases if b.alive and b.x == tx and b.y == ty),
                None,
            )
            if target is None:
                target = next(
                    (d for d in env._enemy_units
                     if d.alive and d.unit_type == UnitType.DEFEND_DRONE
                     and d.x == tx and d.y == ty),
                    None,
                )
            if (target is not None and
                    max(abs(target.x - unit.x),
                        abs(target.y - unit.y),
                        abs(target.z - unit.z)) <= unit.attack_range):
                actions[agent] = ATTACK
            else:
                xy_dist = abs(dx) + abs(dy)
                if xy_dist > APPROACH_DIST:
                    if unit.z <= OBSTACLE_HEIGHT:
                        actions[agent] = MOVE_Z_UP
                    else:
                        if abs(dx) >= abs(dy):
                            actions[agent] = MOVE_RIGHT if dx > 0 else MOVE_LEFT
                        else:
                            actions[agent] = MOVE_DOWN if dy > 0 else MOVE_UP
                else:
                    if unit.z > 3:
                        actions[agent] = MOVE_Z_DOWN
                    elif dx == 0 and dy == 0:
                        actions[agent] = STAY
                    else:
                        if abs(dx) >= abs(dy):
                            actions[agent] = MOVE_RIGHT if dx > 0 else MOVE_LEFT
                        else:
                            actions[agent] = MOVE_DOWN if dy > 0 else MOVE_UP

        else:  # DEFEND_DRONE
            # 서브골 = 추적 중인 적 공격드론의 현재 좌표 (매 스텝 env._refresh_subgoals가 갱신)
            # 해당 드론이 사정거리 내면 공격, 아니면 서브골 방향 추적
            target_id  = env._defend_targets.get(agent)
            target_enm = next(
                (e for e in env._enemy_units if e.unit_id == target_id and e.alive),
                None,
            ) if target_id else None
            if (target_enm is not None and
                    max(abs(target_enm.x - unit.x),
                        abs(target_enm.y - unit.y),
                        abs(target_enm.z - unit.z)) <= unit.attack_range):
                actions[agent] = ATTACK
            else:
                # Mirror enemy defender 로직 (drone_env._rsa_mirror): xy + z 동시 추적
                # Phase 1 (far): obstacle 위로 ascend or xy 이동
                # Phase 2 (close): target z 일치 후 xy 정렬
                xy_dist = abs(dx) + abs(dy)
                if xy_dist > APPROACH_DIST:
                    if unit.z <= OBSTACLE_HEIGHT:
                        actions[agent] = MOVE_Z_UP
                    elif abs(dx) >= abs(dy):
                        actions[agent] = MOVE_RIGHT if dx > 0 else MOVE_LEFT
                    else:
                        actions[agent] = MOVE_DOWN if dy > 0 else MOVE_UP
                else:
                    # close: z를 target z에 일치시킨 후 xy 접근
                    target_z = target_enm.z if target_enm is not None else unit.z
                    dz = target_z - unit.z
                    if abs(dz) > 1:
                        actions[agent] = MOVE_Z_UP if dz > 0 else MOVE_Z_DOWN
                    elif dx == 0 and dy == 0:
                        actions[agent] = STAY
                    elif abs(dx) >= abs(dy):
                        actions[agent] = MOVE_RIGHT if dx > 0 else MOVE_LEFT
                    else:
                        actions[agent] = MOVE_DOWN if dy > 0 else MOVE_UP

    return actions
