from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Tuple


class Team(Enum):
    ALLY = "ally"
    ENEMY = "enemy"


class UnitType(Enum):
    ATTACK_DRONE = "attack_drone"
    DEFEND_DRONE = "defend_drone"
    BASE = "base"


@dataclass
class Unit:
    unit_id: str
    team: Team
    unit_type: UnitType
    x: int
    y: int
    z: int
    max_hp: int
    attack_range: int
    attack_power: int
    hp: int = field(init=False)
    alive: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.hp = self.max_hp

    @property
    def pos(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def take_damage(self, damage: int) -> bool:
        """Apply damage. Returns True if unit was killed."""
        self.hp = max(0, self.hp - damage)
        if self.hp == 0:
            self.alive = False
        return not self.alive

    def move(
        self,
        dx: int,
        dy: int,
        dz: int,
        grid_size: int,
        z_max: int,
        obstacles: Set[Tuple[int, int]],
        obstacle_height: int,
    ) -> bool:
        """Attempt to move by (dx, dy, dz). Returns True on success.

        Obstacle cells (2D) block movement only at z <= obstacle_height.
        At z > obstacle_height drones fly freely over obstacles.
        """
        new_x = self.x + dx
        new_y = self.y + dy
        new_z = self.z + dz
        if not (0 <= new_x < grid_size and 0 <= new_y < grid_size and 0 <= new_z <= z_max):
            return False
        if (new_x, new_y) in obstacles and new_z <= obstacle_height:
            return False
        self.x, self.y, self.z = new_x, new_y, new_z
        return True

    def chebyshev(self, other: "Unit") -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z))

    def manhattan(self, other: "Unit") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)
