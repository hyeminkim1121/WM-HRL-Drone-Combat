from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple


class ReinforcementManager:
    """매 N스텝마다 적 공격드론을 맵 오른쪽 가장자리에 증원 생성."""

    def __init__(self, interval: int = 50, grid_size: int = 50) -> None:
        self.interval = interval
        self.grid_size = grid_size
        self._wave_count = 0  # 누적 증원 횟수

    def reset(self) -> None:
        self._wave_count = 0

    def should_reinforce(self, step: int) -> bool:
        return step > 0 and step % self.interval == 0

    def _spawn_positions(
        self, n: int, obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """맵 오른쪽 가장자리(x=grid_size-1)에서 유효한 위치 n개 반환."""
        x = self.grid_size - 1
        candidates = [
            (x, y)
            for y in range(1, self.grid_size - 1)
            if (x, y) not in obstacles
        ]
        random.shuffle(candidates)
        return candidates[:n]

    def get_reinforcements(
        self, step: int, obstacles: Set[Tuple[int, int]]
    ) -> List[Dict]:
        """현재 스텝에 증원이 있으면 유닛 스펙 리스트를 반환. 없으면 []."""
        if not self.should_reinforce(step):
            return []

        self._wave_count += 1
        positions = self._spawn_positions(2, obstacles)

        units = []
        for i, (x, y) in enumerate(positions):
            units.append(
                {
                    "unit_id": f"enemy_attack_w{self._wave_count}_{i}",
                    "x": x,
                    "y": y,
                    "max_hp": 35,
                    "attack_range": 3,
                    "attack_power": 10,
                }
            )
        return units
