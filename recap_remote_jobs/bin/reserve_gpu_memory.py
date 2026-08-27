"""Reserve otherwise-idle training GPUs until a model is ready to allocate them."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import time

import torch

PID_ENV = "ROBODOJO_GPU_RESERVATION_PID"
READY_FILE_ENV = "ROBODOJO_GPU_RESERVATION_READY_FILE"


def _process_holds_resources(pid: int) -> bool:
    """Return false for a missing or zombie process.

    A zombie has already closed its CUDA contexts; only its exit status remains
    for the launcher shell to reap.
    """

    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    if len(fields) >= 3:
        return fields[2] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def release_gpu_reservation_from_environment(timeout: float = 30.0) -> None:
    """Ask the launcher-owned reservation process to release all CUDA memory."""

    pid_text = os.environ.get(PID_ENV, "").strip()
    ready_text = os.environ.get(READY_FILE_ENV, "").strip()
    if not pid_text:
        return
    ready_file = Path(ready_text) if ready_text else None
    if ready_file is not None and not ready_file.exists():
        return
    pid = int(pid_text)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        return
    graceful_deadline = time.monotonic() + min(5.0, timeout)
    while _process_holds_resources(pid) and time.monotonic() < graceful_deadline:
        time.sleep(0.05)
    if _process_holds_resources(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while _process_holds_resources(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_holds_resources(pid):
        raise TimeoutError(
            f"GPU reservation process {pid} did not exit within {timeout:g}s."
        )
    if ready_file is not None:
        ready_file.unlink(missing_ok=True)


def _allocate_device(device_index: int, leave_free_bytes: int) -> tuple[list[torch.Tensor], int]:
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    target = max(0, free_bytes - leave_free_bytes)
    chunks: list[torch.Tensor] = []
    allocated = 0
    next_chunk = min(target, 256 * 1024 * 1024)
    while allocated < target and next_chunk >= 1024 * 1024:
        request = min(next_chunk, target - allocated)
        try:
            chunks.append(torch.empty(request, dtype=torch.uint8, device=device))
            allocated += request
            next_chunk = min(256 * 1024 * 1024, target - allocated)
        except torch.OutOfMemoryError:
            next_chunk = request // 2
            torch.cuda.empty_cache()
    return chunks, allocated


def main(args: argparse.Namespace) -> None:
    if args.device_count < 1:
        raise ValueError("--device-count must be positive.")
    if args.leave_free_mib < 256:
        raise ValueError("--leave-free-mib must be at least 256.")
    if args.max_hold_seconds < 0:
        raise ValueError("--max-hold-seconds cannot be negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU reservation requested, but CUDA is unavailable.")
    if torch.cuda.device_count() != args.device_count:
        raise RuntimeError(
            f"Expected {args.device_count} visible GPUs, found {torch.cuda.device_count()}."
        )

    ready_file = Path(args.ready_file).resolve()
    allocations: list[list[torch.Tensor]] = []
    allocated_bytes: list[int] = []
    leave_free_bytes = args.leave_free_mib * 1024 * 1024
    for device_index in range(args.device_count):
        chunks, allocated = _allocate_device(device_index, leave_free_bytes)
        allocations.append(chunks)
        allocated_bytes.append(allocated)
        del chunks
    unavailable = [index for index, value in enumerate(allocated_bytes) if value == 0]
    if unavailable:
        raise RuntimeError(
            "Could not reserve any memory on visible GPU(s) "
            f"{unavailable}; each has no free memory beyond the "
            f"{args.leave_free_mib} MiB safety margin."
        )
    ready_file.write_text(
        ",".join(str(value) for value in allocated_bytes) + "\n",
        encoding="utf-8",
    )
    print(
        f"[GPU reservation] {args.label}: reserved "
        + ", ".join(f"cuda:{index}={value / 2**30:.2f} GiB" for index, value in enumerate(allocated_bytes))
        + f"; leaving approximately {args.leave_free_mib} MiB free per GPU",
        flush=True,
    )
    if args.max_hold_seconds > 0:
        signal.alarm(args.max_hold_seconds)
    # SIGTERM uses the operating-system default action. Process teardown is
    # the most reliable way to release every CUDA context and allocation; the
    # training ranks wait for a missing/zombie PID before proceeding.
    signal.pause()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-count", type=int, required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--leave-free-mib", type=int, default=2048)
    parser.add_argument("--max-hold-seconds", type=int, default=0)
    parser.add_argument("--label", default="data preparation")
    main(parser.parse_args())
