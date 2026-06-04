from __future__ import annotations

import argparse
import os
import time
from collections import deque


from .policy_client import LingBotPolicyClient
from .schema import ROBOT_MODEL, action_vector_to_dict, clamp_action, make_lingbot_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LingBot-VLA policy on AlohaMini2Pro.")
    parser.add_argument("--remote_ip", required=True, help="Raspberry Pi AlohaMini host IP.")
    parser.add_argument("--robot_id", default="alohamini2pro")
    parser.add_argument("--robot_model", default=ROBOT_MODEL, choices=["alohamini2pro"])
    parser.add_argument(
        "--policy_host",
        default=os.environ.get("LINGBOT_POLICY_HOST", "127.0.0.1"),
        help=(
            "LingBot websocket server host. Use 127.0.0.1 when using an SSH local tunnel, "
            "or set this to the GPU server IP/domain. Can also be set by LINGBOT_POLICY_HOST."
        ),
    )
    parser.add_argument(
        "--policy_port",
        type=int,
        default=int(os.environ.get("LINGBOT_POLICY_PORT", "8006")),
        help="LingBot websocket server port. Can also be set by LINGBOT_POLICY_PORT.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--max_xy_vel", type=float, default=0.25)
    parser.add_argument("--max_theta_vel", type=float, default=75.0)
    parser.add_argument("--min_lift_mm", type=float, default=0.0)
    parser.add_argument("--max_lift_mm", type=float, default=600.0)
    parser.add_argument("--robo_name", default=ROBOT_MODEL, help="LingBot robot config name used by server reset.")
    args = parser.parse_args()

    import numpy as np

    from lerobot.processor import make_default_processors
    from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
    from lerobot.utils.robot_utils import precise_sleep

    robot = LeKiwiClient(
        LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id, robot_model=args.robot_model)
    )
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    policy = LingBotPolicyClient(host=args.policy_host, port=args.policy_port)

    action_queue: deque[np.ndarray] = deque()

    robot.connect()
    policy.connect()
    try:
        policy.reset(args.robo_name)
        control_interval = 1.0 / args.fps
        started_at = time.perf_counter()

        while time.perf_counter() - started_at < args.duration:
            loop_started = time.perf_counter()
            obs_raw = robot.get_observation()
            obs_processed = robot_observation_processor(obs_raw)

            if not action_queue:
                payload = make_lingbot_observation(obs_processed, task=args.task)
                response = policy.infer(payload)
                chunk = _extract_action_chunk(response)
                for row in chunk:
                    action_queue.append(np.asarray(row, dtype=np.float32))

            action = action_queue.popleft()
            action = clamp_action(
                action,
                max_xy_vel=args.max_xy_vel,
                max_theta_vel=args.max_theta_vel,
                min_lift_mm=args.min_lift_mm,
                max_lift_mm=args.max_lift_mm,
            )
            action_dict = action_vector_to_dict(action)
            robot.send_action(robot_action_processor((action_dict, obs_raw)))

            dt = time.perf_counter() - loop_started
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
    finally:
        policy.close()
        robot.disconnect()


def _extract_action_chunk(response: object) -> np.ndarray:
    import numpy as np

    if isinstance(response, dict):
        if "action" in response:
            chunk = response["action"]
        elif "actions" in response:
            chunk = response["actions"]
        else:
            raise KeyError(f"LingBot response does not contain action/actions keys: {response.keys()}")
    else:
        chunk = response

    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 18:
        raise ValueError(f"Expected action chunk with shape [T, 18], got {arr.shape}")
    return arr


if __name__ == "__main__":
    main()

