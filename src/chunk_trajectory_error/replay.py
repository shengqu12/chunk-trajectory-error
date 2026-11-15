from collections.abc import Sequence
import math


def position_errors(
    candidate_positions: Sequence[Sequence[float]],
    expert_positions: Sequence[Sequence[float]],
) -> list[float]:
    if len(candidate_positions) != len(expert_positions):
        raise ValueError("Candidate and expert trajectories must have equal lengths")
    if not candidate_positions:
        raise ValueError("Trajectories must contain at least one position")

    errors = []
    for candidate, expert in zip(candidate_positions, expert_positions):
        if len(candidate) != 3 or len(expert) != 3:
            raise ValueError("End-effector positions must be XYZ triples")
        errors.append(math.dist(candidate, expert))
    return errors


def trajectory_cost_m2(position_distance_m: Sequence[float]) -> float:
    if not position_distance_m:
        raise ValueError("At least one position distance is required")
    return sum(distance * distance for distance in position_distance_m) / len(position_distance_m)