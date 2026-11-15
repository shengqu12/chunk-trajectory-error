from typing import Any

from chunk_trajectory_error.contracts import Observation, SimulationBackend


class LiberoBackend(SimulationBackend):
    """Adapter boundary for LIBERO; keep simulator state separate from policy input."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        raise NotImplementedError("Implement after pinning LIBERO, MuJoCo, and LeRobot revisions")

    def reset(self, seed: int | None = None, init_state: Any | None = None) -> Observation:
        raise NotImplementedError

    def observe(self) -> Observation:
        raise NotImplementedError

    def snapshot(self) -> Any:
        raise NotImplementedError

    def restore(self, snapshot: Any) -> None:
        raise NotImplementedError

    def step(self, env_action: Any) -> Observation:
        raise NotImplementedError

    def ee_position(self) -> tuple[float, float, float]:
        raise NotImplementedError

    def is_success(self) -> bool:
        raise NotImplementedError