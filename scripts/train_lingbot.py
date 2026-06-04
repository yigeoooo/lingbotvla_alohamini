#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINGBOT_REPO = PROJECT_ROOT / "third_party" / "lingbot-vla"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vla" / "alohamini2pro_real_load20000h.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch LingBot-VLA post-training for AlohaMini2Pro.")
    parser.add_argument("--lingbot_repo", default=str(DEFAULT_LINGBOT_REPO))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--entrypoint", default="tasks/vla/train_lingbotvla.py")
    parser.add_argument("--train_sh", default="train.sh")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="LingBot CLI overrides after --")
    args = parser.parse_args()

    repo = Path(args.lingbot_repo).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    train_sh = repo / args.train_sh
    entrypoint = repo / args.entrypoint

    if not repo.exists():
        raise FileNotFoundError(repo)
    if not train_sh.exists():
        raise FileNotFoundError(train_sh)
    if not entrypoint.exists():
        raise FileNotFoundError(entrypoint)
    if not config.exists():
        raise FileNotFoundError(config)

    overrides = [item for item in args.overrides if item != "--"]
    cmd = ["bash", str(train_sh), str(entrypoint), str(config), *overrides]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}:{env.get("PYTHONPATH", "")}"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
