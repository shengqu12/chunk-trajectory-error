"""Test-time chunk selection with the frozen critic.

The critic ranks candidate chunks well on held-out labels but fails as a training signal.
This module uses it only to *choose*: at every decision the flow policy samples ``num_samples``
chunks for the current observation, the critic scores each tail ``A[r : r + K]`` with the same
inputs it was trained on (end-effector history, robot state, delay), and the chunk with the
lowest (``best``) or highest (``worst``) predicted cost is executed.  No parameter is updated.

The history is rebuilt online from the observation stream (the end-effector position is part
of the state observation), so the selector needs ``observe(observation)`` at every control
step; ``closed_loop_episode`` calls it.  Until ``history_steps + 1`` positions have been seen
the first sample is executed and the decision is logged as unselected, because the critic was
never trained on partial histories.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from chunk_trajectory_error.pipeline import ErrorBundle, PolicyBundle


class ChunkSelector:
    def __init__(
        self, policy: PolicyBundle, error: ErrorBundle, num_samples: int, delay: int, history_steps: int,
        tcp_slice: slice, robot_slice: slice, mode: str = "best",
    ) -> None:
        if not policy.is_flow:
            raise ValueError("chunk selection needs a stochastic (flow) policy to sample from")
        if mode not in ("best", "worst", "first"):
            raise ValueError(f"unknown mode {mode!r}")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.policy, self.error = policy, error
        self.num_samples, self.delay, self.history_steps, self.mode = num_samples, delay, history_steps, mode
        self.tcp_slice, self.robot_slice = tcp_slice, robot_slice
        self.meta = policy.meta
        self.K = error.meta["evaluation_steps"]
        self.state_source = error.meta.get("state_source", "robot")
        self.positions: list[np.ndarray] = []
        self.stats: list[dict[str, Any]] = []

    # -- plumbing used by the evaluation loop -------------------------------- #
    @property
    def is_flow(self) -> bool:
        return True

    def reseed(self, seed: int | None = None) -> None:
        self.policy.reseed(seed)

    def reset_episode(self) -> None:
        self.positions = []

    def observe(self, observation: np.ndarray) -> None:
        self.positions.append(np.asarray(observation, dtype=np.float32)[self.tcp_slice].copy())

    def current_history(self) -> np.ndarray | None:
        if len(self.positions) < self.history_steps + 1:
            return None
        stack = np.stack(self.positions[-(self.history_steps + 1) :])
        return stack[:-1] - stack[-1]

    # -- selection ----------------------------------------------------------- #
    @torch.no_grad()
    def act_with_history(self, observation: np.ndarray, history: np.ndarray | None) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32)[None].expand(self.num_samples, -1)
        chunks = self.policy(obs).clamp(-1.0, 1.0)  # N samples, noise from the policy's own generator
        if history is None or self.mode == "first" or self.num_samples == 1:
            self.stats.append({"selected": False})
            return chunks[0].numpy()
        hist = torch.as_tensor(history, dtype=torch.float32)[None].expand(self.num_samples, -1, -1)
        state = observation if self.state_source == "full_obs" else np.asarray(observation, dtype=np.float32)[self.robot_slice]
        state = torch.as_tensor(state, dtype=torch.float32)[None].expand(self.num_samples, -1)
        tails = chunks[:, self.delay : self.delay + self.K]
        scores = self.error.normalized_cost(hist, state, tails, self.delay)
        index = int(scores.argmin() if self.mode == "best" else scores.argmax())
        self.stats.append({
            "selected": True, "chosen": index, "score_chosen": float(scores[index]),
            "score_min": float(scores.min()), "score_median": float(scores.median()), "score_max": float(scores.max()),
        })
        return chunks[index].numpy()

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.act_with_history(observation, self.current_history())

    def summary(self) -> dict[str, Any]:
        selected = [s for s in self.stats if s["selected"]]
        out = {"decisions": len(self.stats), "selected_decisions": len(selected), "num_samples": self.num_samples, "mode": self.mode}
        if selected:
            for key in ("score_chosen", "score_min", "score_median", "score_max"):
                out[f"mean_{key}"] = float(np.mean([s[key] for s in selected]))
        return out
