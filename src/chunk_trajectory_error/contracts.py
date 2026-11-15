from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


Observation = Mapping[str, Any]
Snapshot = Any
Action = Sequence[float]


class SimulationBackend(Protocol):
    def reset(self, seed: int | None = None, init_state: Any | None = None) -> Observation: ...

    def observe(self) -> Observation: ...

    def snapshot(self) -> Snapshot: ...

    def restore(self, snapshot: Snapshot) -> None: ...

    def step(self, env_action: Action) -> Observation: ...

    def ee_position(self) -> tuple[float, float, float]: ...

    def is_success(self) -> bool: ...


@dataclass(frozen=True)
class ChunkSpec:
    horizon: int
    execute_steps: int
    history_steps: int
    evaluation_steps: int
    delay: int = 0

    def validate(self) -> None:
        if min(self.horizon, self.execute_steps, self.history_steps, self.evaluation_steps) <= 0:
            raise ValueError("Chunk lengths must be positive")
        if self.delay < 0:
            raise ValueError("delay must be non-negative")
        if self.horizon < self.delay + max(self.execute_steps, self.evaluation_steps):
            raise ValueError("horizon must cover delay plus the longer execution/evaluation window")


@dataclass(frozen=True)
class ReplayAudit:
    anchor_id: str
    steps: int
    max_ee_position_error_m: float
    mean_ee_position_error_m: float
    within_tolerance: bool
    status: str