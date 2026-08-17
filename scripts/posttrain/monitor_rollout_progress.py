"""Display one aggregate tqdm bar for parallel RoboDojo rollout workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from progress import BAR_FORMAT, progress_enabled
from tqdm.auto import tqdm


def _recorded_episodes(root: Path) -> int:
    return sum(1 for _ in (root / "episodes").glob("*/manifest.json"))


def _alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # A completed child remains visible to kill(pid, 0) until its parent
        # reaps it. Treat that zombie state as finished so a short or failed
        # rollout cannot leave the monitor waiting forever.
        if stat_path.read_text(encoding="utf-8").split()[2] == "Z":
            return False
    except (FileNotFoundError, IndexError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    current = min(_recorded_episodes(root), args.total)
    with tqdm(
        total=args.total,
        initial=current,
        desc=args.desc,
        unit="episode",
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
        disable=not progress_enabled(),
    ) as progress:
        while current < args.total:
            alive = sum(_alive(pid) for pid in args.worker_pid)
            observed = min(_recorded_episodes(root), args.total)
            if observed > current:
                progress.update(observed - current)
                current = observed
            progress.set_postfix(workers=alive, refresh=False)
            if alive == 0:
                break
            time.sleep(args.poll_seconds)
        observed = min(_recorded_episodes(root), args.total)
        if observed > current:
            progress.update(observed - current)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--worker-pid", type=int, action="append", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--desc", default="Rollout")
    main(parser.parse_args())
