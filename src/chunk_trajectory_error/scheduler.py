"""Shared chunk handover logic for closed-loop evaluation.

The first chunk is synchronous.  Afterwards, for a takeover boundary ``b`` the
prediction is requested from the observation available at step ``b - r``; the
old chunk keeps executing during those ``r`` steps and the new chunk takes over
at ``b`` with ``A[r : r + e]``.  ``r`` is an injected delay, not a measured one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ExecutedStep:
    step: int  # environment step k; the action is applied to the state observed at k
    chunk_id: int
    chunk_index: int  # index inside the chunk that produced this command
    observed_at: int  # step whose observation produced the chunk
    is_boundary: bool  # first command of a new chunk (never true for step 0)


class ChunkScheduler:
    def __init__(self, horizon: int, execute_steps: int, delay: int) -> None:
        if min(horizon, execute_steps) <= 0 or delay < 0:
            raise ValueError("horizon and execute_steps must be positive, delay non-negative")
        if delay > execute_steps:
            raise ValueError("delay must not exceed execute_steps (the request must fall inside the old segment)")
        if horizon < delay + execute_steps:
            raise ValueError("horizon must cover delay + execute_steps")
        self.horizon, self.execute_steps, self.delay = horizon, execute_steps, delay
        self.reset()

    def reset(self) -> None:
        self._chunk: Any = None
        self._chunk_id = -1
        self._index = 0
        self._observed_at = 0
        self._next_boundary = 0
        self._pending: tuple[Any, int] | None = None

    def step(self, k: int, predict: Callable[[], Any]) -> tuple[Any, ExecutedStep]:
        """Return the command for step ``k``.  ``predict()`` must use the observation at ``k``."""
        boundary = False
        if k == 0:
            self._chunk, self._chunk_id, self._index, self._observed_at = predict(), 0, 0, 0
            self._next_boundary = self.execute_steps
        else:
            if self.delay > 0 and k == self._next_boundary - self.delay:
                self._pending = (predict(), k)  # computed now, usable only from the boundary on
            if k == self._next_boundary:
                if self.delay == 0:
                    self._pending = (predict(), k)
                if self._pending is None:
                    raise RuntimeError("no prediction available at the takeover boundary")
                self._chunk, self._observed_at = self._pending
                self._pending = None
                self._chunk_id += 1
                self._index = self.delay
                self._next_boundary = k + self.execute_steps
                boundary = True
        command = self._chunk[self._index]
        record = ExecutedStep(k, self._chunk_id, self._index, self._observed_at, boundary)
        self._index += 1
        return command, record
