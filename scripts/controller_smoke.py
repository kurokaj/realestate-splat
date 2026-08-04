#!/usr/bin/env python3
"""Convenience smoke helpers for the terminal-first controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller_api.main import (  # noqa: E402
    PreprocessQueueRequest,
    ProjectCreate,
    StageActionRequest,
    StageRunCreate,
    approve_stage_run,
    create_project,
    create_stage_run,
    queue_preprocess,
    startup,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small controller smoke setup actions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fake = subparsers.add_parser("fake-chain", help="Create a project and queue fake preprocess/COLMAP/training flow.")
    fake.add_argument("--project-id", default="smoke_8a")
    fake.add_argument("--name", default="Smoke 8A")
    fake.add_argument("--raw-uri", default="r2://buildvision3d-pipeline/projects/smoke_8a/raw")

    preprocess = subparsers.add_parser("queue-preprocess", help="Create a project and queue real local preprocessing.")
    preprocess.add_argument("--project-id", required=True)
    preprocess.add_argument("--name")
    preprocess.add_argument("--raw-uri", required=True)
    preprocess.add_argument("--output-uri")
    preprocess.add_argument("--profile", default="indoor_room")
    preprocess.add_argument("--python-bin")
    preprocess.add_argument("--dry-run", action="store_true")
    preprocess.add_argument("--preprocess-arg", action="append", default=[])

    approve_parser = subparsers.add_parser("approve", help="Approve a stage and enqueue its next stage.")
    approve_parser.add_argument("stage_run_id")
    return parser.parse_args(argv)


def fake_chain(args: argparse.Namespace) -> None:
    startup()
    project = create_project(ProjectCreate(id=args.project_id, name=args.name, raw_uri=args.raw_uri))
    preprocess = create_stage_run(
        args.project_id,
        StageRunCreate(stage="preprocess", provider="local_fake", output_uri=f"fake://projects/{args.project_id}/preprocess/current"),
    )
    print(f"project {project['id']} {project['status']}")
    print(f"queued {preprocess['id']} {preprocess['status']}")
    print("Run `python -m controller_worker --once`, then approve the preprocess stage to enqueue COLMAP:")
    print(f"python scripts/controller_smoke.py approve {preprocess['id']}")


def queue_local_preprocess(args: argparse.Namespace) -> None:
    startup()
    create_project(ProjectCreate(id=args.project_id, name=args.name or args.project_id, raw_uri=args.raw_uri))
    stage = queue_preprocess(
        args.project_id,
        PreprocessQueueRequest(
            raw_uri=args.raw_uri,
            output_uri=args.output_uri,
            profile=args.profile,
            python_bin=args.python_bin,
            preprocess_args=args.preprocess_arg,
            dry_run=args.dry_run,
        ),
    )
    print(f"queued {stage['id']} {stage['status']}")


def approve(stage_run_id: str) -> None:
    startup()
    next_stage = approve_stage_run(stage_run_id, StageActionRequest())
    print(f"queued {next_stage['id']} {next_stage['status']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "fake-chain":
        fake_chain(args)
    elif args.command == "queue-preprocess":
        queue_local_preprocess(args)
    elif args.command == "approve":
        approve(args.stage_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
