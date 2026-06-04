from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ROBOT_MODEL = "alohamini2pro"
STATE_DIM = 18

ACTION_NAMES: tuple[str, ...] = (
    "arm_left_shoulder_pan.pos",
    "arm_left_shoulder_lift.pos",
    "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos",
    "arm_left_wrist_yaw.pos",
    "arm_left_wrist_roll.pos",
    "arm_left_gripper.pos",
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_yaw.pos",
    "arm_right_wrist_roll.pos",
    "arm_right_gripper.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
    "lift_axis.height_mm",
)

REQUIRED_IMAGE_FEATURES: tuple[str, ...] = (
    "observation.images.chest",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)

CAMERA_MAP: dict[str, str] = {
    "observation.images.chest": "chest",
    "observation.images.wrist_left": "wrist_left",
    "observation.images.wrist_right": "wrist_right",
}


@dataclass(frozen=True)
class DatasetCheck:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = value.get("shape") if isinstance(value, dict) else getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _names_tuple(value: Any) -> tuple[str, ...]:
    names = value.get("names") if isinstance(value, dict) else getattr(value, "names", None)
    if names is None:
        return ()
    return tuple(str(v) for v in names)


def validate_lerobot_features(features: dict[str, Any]) -> DatasetCheck:
    errors: list[str] = []
    warnings: list[str] = []

    action = features.get("action")
    state = features.get("observation.state")

    if _shape_tuple(action) != (STATE_DIM,):
        errors.append(f"action.shape must be ({STATE_DIM},), got {_shape_tuple(action)}")
    if _shape_tuple(state) != (STATE_DIM,):
        errors.append(f"observation.state.shape must be ({STATE_DIM},), got {_shape_tuple(state)}")

    action_names = _names_tuple(action)
    state_names = _names_tuple(state)
    if action_names != ACTION_NAMES:
        errors.append(f"action.names do not match {ROBOT_MODEL} order")
    if state_names != ACTION_NAMES:
        errors.append(f"observation.state.names do not match {ROBOT_MODEL} order")

    for key in REQUIRED_IMAGE_FEATURES:
        if key not in features:
            errors.append(f"missing required image feature: {key}")

    optional_images = {
        "observation.images.forward",
        "observation.images.backward",
    }
    missing_optional = sorted(optional_images.difference(features))
    if missing_optional:
        warnings.append(f"optional navigation cameras not present: {missing_optional}")

    return DatasetCheck(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def make_state_vector(observation: dict[str, Any]) -> np.ndarray:
    import numpy as np

    return np.asarray([observation.get(key, 0.0) for key in ACTION_NAMES], dtype=np.float32)


def action_vector_to_dict(action: np.ndarray) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}-dim action, got shape {arr.shape}")
    return {key: float(arr[i]) for i, key in enumerate(ACTION_NAMES)}


def clamp_action(
    action: np.ndarray,
    *,
    max_xy_vel: float = 0.25,
    max_theta_vel: float = 75.0,
    min_lift_mm: float = 0.0,
    max_lift_mm: float = 600.0,
) -> np.ndarray:
    import numpy as np

    arr = np.asarray(action, dtype=np.float32).copy().reshape(-1)
    if arr.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}-dim action, got shape {arr.shape}")
    arr[14] = np.clip(arr[14], -max_xy_vel, max_xy_vel)
    arr[15] = np.clip(arr[15], -max_xy_vel, max_xy_vel)
    arr[16] = np.clip(arr[16], -max_theta_vel, max_theta_vel)
    arr[17] = np.clip(arr[17], min_lift_mm, max_lift_mm)
    return arr


def make_lingbot_observation(
    observation: dict[str, Any],
    *,
    task: str,
    include_optional_cameras: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation.state": make_state_vector(observation),
        "task": task,
        "prompt": [task],
    }

    for lingbot_key, local_key in CAMERA_MAP.items():
        if local_key not in observation:
            raise KeyError(f"Robot observation is missing camera '{local_key}'")
        payload[lingbot_key] = observation[local_key]

    if include_optional_cameras:
        for local_key in ("forward", "backward"):
            if local_key in observation:
                payload[f"observation.images.{local_key}"] = observation[local_key]

    return payload

