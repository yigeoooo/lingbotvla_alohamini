from __future__ import annotations

import argparse

from .schema import validate_lerobot_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a LeRobot dataset is AlohaMini2Pro 18-dim format.")
    parser.add_argument("--repo_id", required=True, help="HF repo id or local LeRobot dataset repo id.")
    parser.add_argument("--root", default=None, help="Optional local dataset root.")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    check = validate_lerobot_features(dataset.meta.features)
    errors = list(check.errors)
    warnings = list(check.warnings)
    task_names = _task_names(getattr(dataset.meta, "tasks", None))

    if not task_names:
        errors.append("dataset must contain at least one natural-language task in meta/tasks")
    for task in task_names:
        normalized = task.strip().lower()
        if not normalized:
            errors.append("dataset contains an empty task name")
        elif normalized in {"robot task", "my task description", "my task description4", "task"}:
            warnings.append(f"task name looks generic; use a concrete instruction instead: {task!r}")

    print(f"repo_id: {dataset.repo_id}")
    print(f"fps: {dataset.meta.fps}")
    print(f"robot_type: {dataset.meta.robot_type}")
    print(f"total_episodes: {dataset.meta.total_episodes}")
    print(f"total_frames: {dataset.meta.total_frames}")
    print(f"total_tasks: {len(task_names)}")
    print(f"action.shape: {dataset.meta.features['action']['shape']}")
    print(f"observation.state.shape: {dataset.meta.features['observation.state']['shape']}")
    print("tasks:")
    for i, task in enumerate(task_names):
        print(f"  {i:02d}: {task}")
    print("action.names:")
    for i, name in enumerate(dataset.meta.features["action"]["names"]):
        print(f"  {i:02d}: {name}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("OK: dataset matches AlohaMini2Pro 18-dim schema and contains task prompts.")


def _task_names(tasks: object) -> list[str]:
    if tasks is None:
        return []
    if hasattr(tasks, "index"):
        return [str(task) for task in tasks.index.tolist()]
    if isinstance(tasks, dict):
        return [str(task) for task in tasks.keys()]
    return [str(task) for task in tasks]


if __name__ == "__main__":
    main()

