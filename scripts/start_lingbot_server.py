#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINGBOT_REPO = PROJECT_ROOT / "third_party" / "lingbot-vla"
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_configs" / "alohamini2pro.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start upstream LingBot-VLA websocket policy server.")
    parser.add_argument("--lingbot_repo", default=str(DEFAULT_LINGBOT_REPO))
    parser.add_argument("--ckpt", required=True, help="Post-training checkpoint directory.")
    parser.add_argument("--qwen25_path", required=True)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--use_length", type=int, default=1)
    parser.add_argument("--norm_path", default=None)
    parser.add_argument("--num_denoising_step", type=int, default=10)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--robot_config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--no_install_robot_config", action="store_true")
    parser.add_argument("--overwrite_robot_config", action="store_true")
    parser.add_argument("--entrypoint", default="deploy/lingbot_vla_policy.py")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.lingbot_repo).expanduser().resolve()
    entrypoint = repo / args.entrypoint
    ckpt = Path(args.ckpt).expanduser().resolve()

    if not entrypoint.exists():
        raise FileNotFoundError(entrypoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not (ckpt / "lingbotvla_cli.yaml").exists():
        raise FileNotFoundError(f"{ckpt}/lingbotvla_cli.yaml")

    if not args.no_install_robot_config:
        src_robot_config = Path(args.robot_config).expanduser().resolve()
        dst_robot_config = repo / "configs" / "robot_configs" / src_robot_config.name
        if not src_robot_config.exists():
            raise FileNotFoundError(src_robot_config)
        if dst_robot_config.exists() and not args.overwrite_robot_config:
            print(f"Robot config already exists: {dst_robot_config}")
        else:
            dst_robot_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_robot_config, dst_robot_config)
            print(f"Installed robot config: {dst_robot_config}")

    cmd = [
        "python",
        str(entrypoint),
        "--model_path",
        str(ckpt),
        "--use_length",
        str(args.use_length),
        "--port",
        str(args.port),
        "--num_denoising_step",
        str(args.num_denoising_step),
    ]
    if args.norm_path:
        cmd.extend(["--norm_path", str(Path(args.norm_path).expanduser().resolve())])
    if args.use_compile:
        cmd.append("--use_compile")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}:{env.get("PYTHONPATH", "")}"
    env["QWEN25_PATH"] = args.qwen25_path

    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
