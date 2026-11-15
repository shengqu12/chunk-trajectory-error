"""Physics shifts for the sim-to-sim pilot (docs/SIM2SIM_PILOT_SPEC.md, section 2).

A shift is applied **once** to a freshly created environment, after its first reset,
relative to the nominal values read at that moment.  PickCube-v1 does not rebuild the
scene on later resets (``reconfiguration_freq = 0``), so the change persists; applying it
again after every reset would compound it.  ``apply_shift`` refuses a second application
and ``verify_shift`` checks after a reset that the applied values are still in place.

Families
  actuation  scale stiffness and damping of the 7 arm joint drives by ``s`` (fingers untouched)
  latency    the command sent at step k is applied at step k + d (``LatencyBackend``)
  contact    cube mass * m, cube friction * f

Contact material modes
  ``cube_only`` (default)  the cube's collision shapes get their own material with the scaled
                            friction.  In SAPIEN 3.0.3 the cube shares the scene default material
                            with the table, the ground and the arm links, so this is the only way to
                            shift the cube alone.
  ``shared_in_place``       the spec's literal recipe: scale ``shape.get_physical_material()`` in
                            place.  This mutates the process-wide default material and therefore also
                            the table, ground and arm-link friction (and every environment created
                            later in the same process).  Kept as a diagnostic.
``body.set_mass`` does not change the inertia tensor (checked on sapien 3.0.3); the inertia is
recorded, not scaled, because the spec asks for a mass change only.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FAMILIES = ("nominal", "actuation", "latency", "contact")
ARM_JOINTS = 7
CONTACT_MODES = ("cube_only", "shared_in_place")


@dataclass(frozen=True)
class ShiftSpec:
    family: str
    stiffness_scale: float = 1.0  # actuation s
    delay_steps: int = 0  # latency d
    friction_scale: float = 1.0  # contact f
    mass_scale: float = 1.0  # contact m
    contact_mode: str = "cube_only"
    name: str = field(default="")

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown shift family {self.family!r}")
        if self.contact_mode not in CONTACT_MODES:
            raise ValueError(f"unknown contact mode {self.contact_mode!r}")
        if self.delay_steps < 0 or self.stiffness_scale <= 0 or self.friction_scale < 0 or self.mass_scale <= 0:
            raise ValueError("shift levels out of range")
        if not self.name:
            object.__setattr__(self, "name", self.default_name())

    def default_name(self) -> str:
        if self.family == "actuation":
            return f"actuation_s{self.stiffness_scale:g}"
        if self.family == "latency":
            return f"latency_d{self.delay_steps}"
        if self.family == "contact":
            suffix = "" if self.contact_mode == "cube_only" else "_shared"
            return f"contact{suffix}_f{self.friction_scale:g}_m{self.mass_scale:g}"
        return "nominal"

    def level(self) -> dict[str, Any]:
        if self.family == "actuation":
            return {"s": self.stiffness_scale}
        if self.family == "latency":
            return {"d": self.delay_steps}
        if self.family == "contact":
            return {"f": self.friction_scale, "m": self.mass_scale, "contact_mode": self.contact_mode}
        return {}


# --------------------------------------------------------------------------- #
# reading physical values
# --------------------------------------------------------------------------- #
def _scalar(value: Any) -> float:
    array = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float64).reshape(-1)
    return float(array[0])


def arm_joints(base: Any) -> list:
    return list(base.agent.robot.get_active_joints()[:ARM_JOINTS])


def cube_shapes(base: Any) -> list:
    return [shape for body in base.cube._bodies for shape in body.get_collision_shapes()]


def read_physics(base: Any) -> dict[str, Any]:
    """Current values of every quantity a shift can touch (unbatched floats)."""
    import sapien

    joints = list(base.agent.robot.get_active_joints())
    default = sapien.physx.get_default_material()
    table_shapes = [s for b in base.scene.actors["table-workspace"]._bodies for s in b.get_collision_shapes()]
    return {
        "joints": {
            joint.name: {"stiffness": _scalar(joint.stiffness), "damping": _scalar(joint.damping), "force_limit": _scalar(joint.force_limit)}
            for joint in joints
        },
        "cube_mass_kg": [float(body.get_mass()) for body in base.cube._bodies],
        "cube_inertia": [np.asarray(body.get_inertia(), dtype=np.float64).tolist() for body in base.cube._bodies],
        "cube_friction": [
            {"static": float(s.get_physical_material().static_friction), "dynamic": float(s.get_physical_material().dynamic_friction)}
            for s in cube_shapes(base)
        ],
        "table_friction": [
            {"static": float(s.get_physical_material().static_friction), "dynamic": float(s.get_physical_material().dynamic_friction)}
            for s in table_shapes
        ],
        "default_material_friction": {"static": float(default.static_friction), "dynamic": float(default.dynamic_friction)},
        "cube_material_is_default": bool(all(s.get_physical_material() is default or id(s.get_physical_material()) == id(default) for s in cube_shapes(base))),
    }


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
class LatencyBackend:
    """Backend wrapper: the command sent at step k is applied at step k + d.

    For the first d steps of an episode the applied command is a hold: zero translation
    with the gripper channel of the first command of the episode (the current gripper
    command at that moment).  The queue of pending (not yet applied) commands is part of
    the target state: ``pending()`` exposes it, ``snapshot``/``restore`` carry it, and
    ``last_applied`` holds the command that actually reached the simulator.
    """

    def __init__(self, backend: Any, delay_steps: int, gripper_dims: list[int]) -> None:
        if delay_steps < 0:
            raise ValueError("delay must be non-negative")
        self.inner = backend
        self.delay_steps = int(delay_steps)
        self.gripper_dims = list(gripper_dims)
        self._queue: deque = deque()
        self.last_applied: np.ndarray | None = None
        self._action_dim = 0

    def __getattr__(self, name: str) -> Any:  # everything else is the wrapped backend
        return getattr(self.inner, name)

    def reset(self, seed: int | None = None, init_state: Any | None = None):
        self._queue.clear()
        self.last_applied = None
        return self.inner.reset(seed=seed, init_state=init_state)

    def pending(self) -> np.ndarray:
        """Commands sent but not yet applied, oldest first, [n, action_dim]."""
        return np.array(list(self._queue), dtype=np.float32).reshape(len(self._queue), self._action_dim)

    def in_flight(self, next_command: Any) -> np.ndarray:
        """The d commands that will be applied before ``next_command`` (the pending buffer at this step,
        including the start-of-episode hold that ``step`` creates on the first call), [d, action_dim]."""
        command = np.asarray(next_command, dtype=np.float32)
        if self.last_applied is None and not self._queue and self.delay_steps > 0:
            hold = np.zeros_like(command)
            hold[self.gripper_dims] = command[self.gripper_dims]
            return np.repeat(hold[None], self.delay_steps, axis=0)
        return np.array(list(self._queue), dtype=np.float32).reshape(len(self._queue), command.shape[-1])

    def step(self, env_action: Any):
        command = np.asarray(env_action, dtype=np.float32).copy()
        self._action_dim = int(command.shape[-1])
        if self.last_applied is None and not self._queue and self.delay_steps > 0:
            hold = np.zeros_like(command)
            hold[self.gripper_dims] = command[self.gripper_dims]
            self._queue.extend(hold.copy() for _ in range(self.delay_steps))
        self._queue.append(command)
        applied = self._queue.popleft()
        self.last_applied = applied
        return self.inner.step(applied)

    def snapshot(self) -> Any:
        return {"inner": self.inner.snapshot(), "pending": self.pending().copy(),
                "last_applied": None if self.last_applied is None else self.last_applied.copy()}

    def restore(self, snapshot: Any) -> None:  # sim-to-sim diagnostics only
        self.inner.restore(snapshot["inner"])
        pending = np.asarray(snapshot["pending"], dtype=np.float32)
        self._action_dim = int(pending.shape[-1]) if pending.ndim == 2 else self._action_dim
        self._queue = deque(c.copy() for c in pending)
        self.last_applied = None if snapshot["last_applied"] is None else np.asarray(snapshot["last_applied"], dtype=np.float32).copy()


# --------------------------------------------------------------------------- #
# applying and verifying
# --------------------------------------------------------------------------- #
def apply_shift(backend: Any, shift: ShiftSpec, gripper_dims: list[int]) -> tuple[Any, dict[str, Any]]:
    """Apply ``shift`` once to ``backend`` (a ``ManiSkillBackend`` that has been reset).

    Returns the backend to use (wrapped for latency) and a record with the nominal values
    read just before the change and the values right after it.
    """
    base = backend.base
    if getattr(base, "_sim2sim_shift_applied", None) is not None:
        raise RuntimeError(f"a shift ({base._sim2sim_shift_applied}) was already applied to this environment")
    nominal = read_physics(base)
    target_backend = backend
    if shift.family == "actuation":
        for joint in arm_joints(base):
            joint.set_drive_properties(
                _scalar(joint.stiffness) * shift.stiffness_scale,
                _scalar(joint.damping) * shift.stiffness_scale,
                force_limit=_scalar(joint.force_limit),
            )
    elif shift.family == "contact":
        import sapien

        for body in base.cube._bodies:
            body.set_mass(body.get_mass() * shift.mass_scale)
        for shape in cube_shapes(base):
            material = shape.get_physical_material()
            if shift.contact_mode == "shared_in_place":
                material.set_static_friction(material.static_friction * shift.friction_scale)
                material.set_dynamic_friction(material.dynamic_friction * shift.friction_scale)
            else:
                shape.set_physical_material(
                    sapien.physx.PhysxMaterial(
                        material.static_friction * shift.friction_scale,
                        material.dynamic_friction * shift.friction_scale,
                        material.restitution,
                    )
                )
    elif shift.family == "latency":
        target_backend = LatencyBackend(backend, shift.delay_steps, gripper_dims)
    base._sim2sim_shift_applied = shift.name
    applied = read_physics(base)
    expected = expected_physics(nominal, shift)
    check_physics(applied, expected)
    return target_backend, {"shift": shift.name, "family": shift.family, "level": shift.level(),
                            "nominal": nominal, "applied": applied, "expected": expected}


def expected_physics(nominal: dict[str, Any], shift: ShiftSpec) -> dict[str, Any]:
    expected = copy.deepcopy(nominal)
    if shift.family == "actuation":
        for index, (name, values) in enumerate(expected["joints"].items()):
            if index < ARM_JOINTS:
                values["stiffness"] *= shift.stiffness_scale
                values["damping"] *= shift.stiffness_scale
    elif shift.family == "contact":
        expected["cube_mass_kg"] = [m * shift.mass_scale for m in expected["cube_mass_kg"]]
        for entry in expected["cube_friction"]:
            entry["static"] *= shift.friction_scale
            entry["dynamic"] *= shift.friction_scale
        if shift.contact_mode == "shared_in_place" and nominal["cube_material_is_default"]:
            for entry in expected["table_friction"]:
                entry["static"] *= shift.friction_scale
                entry["dynamic"] *= shift.friction_scale
            expected["default_material_friction"] = {k: v * shift.friction_scale for k, v in expected["default_material_friction"].items()}
        if shift.contact_mode == "cube_only":
            expected["cube_material_is_default"] = False
    return expected


def check_physics(actual: dict[str, Any], expected: dict[str, Any], rtol: float = 1e-5) -> None:
    """Raise if any value differs from the expected one (float32 storage -> relative tolerance)."""

    def walk(a: Any, e: Any, path: str) -> None:
        if isinstance(e, dict):
            if set(a) != set(e):
                raise AssertionError(f"{path}: keys differ")
            for key in e:
                walk(a[key], e[key], f"{path}.{key}")
        elif isinstance(e, list):
            if len(a) != len(e):
                raise AssertionError(f"{path}: lengths differ")
            for index, (x, y) in enumerate(zip(a, e)):
                walk(x, y, f"{path}[{index}]")
        elif isinstance(e, bool):
            if a != e:
                raise AssertionError(f"{path}: {a} != {e}")
        elif not np.isclose(float(a), float(e), rtol=rtol, atol=0.0):
            raise AssertionError(f"{path}: {a} != expected {e}")

    walk(actual, expected, "physics")


def verify_shift(backend: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Assert the applied values are still in place (call after a reset); returns the read-back."""
    base = backend.base if not isinstance(backend, LatencyBackend) else backend.inner.base
    current = read_physics(base)
    check_physics(current, record["expected"])
    if base._sim2sim_shift_applied != record["shift"]:
        raise AssertionError("shift marker changed")
    if record["family"] == "latency" and backend.delay_steps != record["level"]["d"]:
        raise AssertionError("latency changed")
    return current


def make_target_backend(config: dict[str, Any], shift: ShiftSpec, gripper_dims: list[int], first_seed: int = 0,
                        render: bool = False) -> tuple[Any, dict[str, Any]]:
    """Fresh environment -> first reset -> apply the shift once -> reset -> verify it persisted."""
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    backend = ManiSkillBackend(config, render=render)
    backend.reset(seed=first_seed)
    target, record = apply_shift(backend, shift, gripper_dims)
    target.reset(seed=first_seed + 1)
    record["after_reset"] = verify_shift(target, record)
    record["persisted_after_reset"] = True
    return target, record
