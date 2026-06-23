"""Shared CLI helpers for pipeline scripts."""

from __future__ import annotations

import datetime as dt
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class CommandResult:
    name: str
    command: List[str]
    log_path: str
    returncode: int
    started_at: str
    finished_at: str
    duration_seconds: float
    cwd: Optional[str] = None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_under_run(run_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return run_dir / path


def relative_to(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_pairs(raw_pairs: Sequence[str], option_name: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw_pair in raw_pairs:
        if "=" not in raw_pair:
            raise SystemExit(f"{option_name} expects KEY=VALUE, got: {raw_pair}")
        key, value = raw_pair.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"{option_name} has an empty key: {raw_pair}")
        parsed[key] = parse_scalar(value.strip())
    return parsed


def option_map_to_cli(options: Mapping[str, Any]) -> List[str]:
    cli_args: List[str] = []
    for key, value in sorted(options.items()):
        if value is None:
            continue
        option_name = str(key)
        if not option_name.startswith("--"):
            option_name = f"--{option_name}"
        if isinstance(value, bool):
            cli_args.append(f"{option_name}={str(value)}")
        else:
            cli_args.append(f"{option_name}={value}")
    return cli_args


def resolve_executable(name_or_path: Optional[Path], fallback_name: str, dry_run: bool) -> str:
    if name_or_path is not None:
        expanded = name_or_path.expanduser()
        if expanded.exists() or dry_run:
            return str(expanded)

    path_hit = shutil.which(fallback_name)
    if path_hit:
        return path_hit
    if dry_run:
        return fallback_name
    raise SystemExit(f"Could not find {fallback_name}. Pass an explicit executable path or add it to PATH.")


def run_logged_command(
    name: str,
    command: Sequence[str],
    logs_dir: Path,
    cwd: Optional[Path] = None,
) -> CommandResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    started_at = utc_now()
    start_time = time.monotonic()

    print(f"\n$ {shlex.join(command)}", flush=True)
    if cwd is not None:
        print(f"(cwd: {cwd})", flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n")
        if cwd is not None:
            log_file.write(f"cwd: {cwd}\n")
        log_file.write("\n")
        log_file.flush()

        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = process.wait()

    finished_at = utc_now()
    duration = round(time.monotonic() - start_time, 3)
    result = CommandResult(
        name=name,
        command=list(command),
        log_path=str(log_path),
        returncode=returncode,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        cwd=str(cwd) if cwd is not None else None,
    )
    if returncode != 0:
        raise RuntimeError(f"Command failed ({name}) with exit code {returncode}. See {log_path}")
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def command_results_to_json(results: Iterable[CommandResult]) -> List[Dict[str, Any]]:
    return [asdict(result) for result in results]


def environment_report(executables: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executables": dict(executables),
    }


def latest_existing_file(candidates: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in candidates if path.exists() and path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)

