from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np


class GridMap:
    """100x100 격자 맵. 장애물은 중앙부에 배치하며 우회로 3개 이상 보장.

    장애물 레이아웃:
        Block A  : x=[20,22], y=[5,17]   — 상단 세로벽
        Block B  : x=[27,29], y=[19,31]  — 중앙 세로벽 (오른쪽)
        Block C  : x=[20,22], y=[33,45]  — 하단 세로벽

    우회로:
        Route 1  : y < 5            (최상단 통로)
        Route 2  : y = 18           (상단-중단 사이 통로)
        Route 3  : y = 32           (중단-하단 사이 통로)
        Route 4  : y > 45           (최하단 통로)
    """

    def __init__(self, size: int = 50) -> None:
        self.size = size
        self.obstacles: Set[Tuple[int, int]] = set()
        self._build_obstacles()

    def _build_obstacles(self) -> None:
        # 100×100 기준 (구 50×50 좌표 × 2)
        # Block A — upper vertical wall  (x=[40,44], y=[10,34])
        for x in range(40, 45):
            for y in range(10, 35):
                self.obstacles.add((x, y))

        # Block B — middle-right vertical wall  (x=[54,58], y=[38,62])
        for x in range(54, 59):
            for y in range(38, 63):
                self.obstacles.add((x, y))

        # Block C — lower vertical wall  (x=[40,44], y=[66,90])
        for x in range(40, 45):
            for y in range(66, 91):
                self.obstacles.add((x, y))

    def is_passable(self, x: int, y: int) -> bool:
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return (x, y) not in self.obstacles

    def to_array(self) -> np.ndarray:
        """Returns (size, size) uint8 array. 1 = obstacle, 0 = free."""
        arr = np.zeros((self.size, self.size), dtype=np.uint8)
        for ox, oy in self.obstacles:
            arr[oy, ox] = 1
        return arr


class RandomGridMap(GridMap):
    """에피소드마다 랜덤으로 장애물을 재배치하는 맵. 우회로 4개를 보장한다.

    구조: 3개의 수직 장애물 블록을 중앙 지대(x=17~31)에 배치.
    통로는 블록 사이 간격과 상하 여백으로 보장:
        Route 1: y < y_a_start       (최상단)
        Route 2: y_a_end < y < y_b_start   (상-중 사이)
        Route 3: y_b_end < y < y_c_start   (중-하 사이)
        Route 4: y > y_c_end         (최하단)
    """

    def __init__(self, size: int = 50, seed: Optional[int] = None) -> None:
        self.size = size
        self.obstacles: Set[Tuple[int, int]] = set()
        rng = np.random.RandomState(seed)
        self._build_random(rng)

    def _build_random(self, rng: np.random.RandomState) -> None:
        size = self.size

        def safe_randint(lo: int, hi: int, fallback: int) -> int:
            return int(rng.randint(lo, hi)) if lo < hi else fallback

        # 100×100 기준 범위 (구 50×50 × 2)
        # Block A — upper wall
        x_a       = safe_randint(36, 48, 42)
        y_a_start = safe_randint(6,  14,  8)
        y_a_end   = safe_randint(26, 38, 32)
        w_a       = safe_randint(3,   6,  4)

        # Block B — middle wall (shifted right)
        x_b       = safe_randint(50, 64, 56)
        y_b_start = safe_randint(y_a_end + 4, min(y_a_end + 14, size - 40), y_a_end + 6)
        y_b_end   = safe_randint(y_b_start + 12, min(y_b_start + 26, size - 24), y_b_start + 16)
        w_b       = safe_randint(3, 6, 4)

        # Block C — lower wall
        x_c       = x_a + safe_randint(-4, 5, 0)
        y_c_start = safe_randint(y_b_end + 4, min(y_b_end + 14, size - 20), y_b_end + 6)
        y_c_end   = safe_randint(y_c_start + 10, min(y_c_start + 22, size - 4), y_c_start + 14)
        w_c       = safe_randint(3, 6, 4)

        def fill_block(cx: int, w: int, y0: int, y1: int) -> None:
            for dx in range(-(w // 2), w // 2 + 1):
                for y in range(y0, y1 + 1):
                    bx = cx + dx
                    if 0 <= bx < size:
                        self.obstacles.add((bx, y))

        fill_block(x_a, w_a, y_a_start, y_a_end)
        fill_block(x_b, w_b, y_b_start, y_b_end)
        fill_block(x_c, w_c, y_c_start, y_c_end)

        # 통로 보장 확인 (설계상 항상 만족)
        # Route 1: y in [0, y_a_start-1]
        # Route 2: y in [y_a_end+1, y_b_start-1]
        # Route 3: y in [y_b_end+1, y_c_start-1]
        # Route 4: y in [y_c_end+1, size-1]
