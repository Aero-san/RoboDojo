"""Progress-bar helpers for the official WCM train/evaluation loops.

The upstream WCM code intentionally keeps its JSON/timing output simple.  The
adapter wraps its DataLoaders instead of replacing those loops, so the
original stdout logs and checkpoint behavior remain unchanged while rank zero
gets an interactive tqdm bar.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from contextlib import contextmanager
import os
import sys
from typing import Any

from torch.utils.data import DataLoader as TorchDataLoader
from tqdm.auto import tqdm

BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def progress_enabled() -> bool:
    """Return whether the current process should render progress bars."""

    if os.environ.get("ROBODOJO_DISABLE_PROGRESS", "0") == "1":
        return False
    if os.environ.get("WCM_DISABLE_PROGRESS", "0") == "1":
        return False
    # DDP workers must not write competing carriage-return streams.  The rank
    # zero bar still represents the global training/evaluation progress.
    return int(os.environ.get("RANK", "0")) == 0


@contextmanager
def tqdm_print_bridge():
    """Route normal ``print`` calls through tqdm while bars are active.

    ``tqdm.write`` clears active bars before writing and redraws them after the
    message.  The WCM trainer uses ordinary ``print`` for JSON metrics and
    timing lines, so a temporary builtins bridge prevents those lines from
    leaving stale copies of the bar in the terminal.  Calls targeting other
    file objects keep their original behavior.
    """

    if not progress_enabled():
        yield
        return

    original_print = builtins.print

    def print_with_tqdm(*args: Any, **kwargs: Any) -> None:
        target = kwargs.get("file")
        if target is not None and target not in (sys.stdout, sys.stderr):
            original_print(*args, **kwargs)
            return

        sep = kwargs.get("sep", " ")
        if sep is None:
            sep = " "
        end = kwargs.get("end", "\n")
        flush = kwargs.get("flush", False)
        message = sep.join(str(value) for value in args)
        target = sys.stdout if target is None else target
        tqdm.write(message, file=target, end=end)
        if flush:
            target.flush()

    builtins.print = print_with_tqdm
    try:
        yield
    finally:
        builtins.print = original_print


def progress_iter(iterable: Any, *, desc: str, total: int | None = None, unit: str = "item") -> Iterator[Any]:
    """Yield an iterable with a rank-zero progress bar."""

    with tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
        disable=not progress_enabled(),
    ) as progress:
        yield from progress


class ProgressDataLoader:
    """Delegate to a DataLoader and show progress whenever it is iterated."""

    def __init__(
        self,
        loader: TorchDataLoader,
        *,
        desc_factory,
    ) -> None:
        self._loader = loader
        self._desc_factory = desc_factory
        self._iteration_count = 0

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __iter__(self) -> Iterator[Any]:
        self._iteration_count += 1
        desc = self._desc_factory(self._iteration_count)
        with tqdm(
            total=len(self._loader),
            desc=desc,
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format=BAR_FORMAT,
            disable=not progress_enabled(),
        ) as progress:
            for batch in self._loader:
                progress.update(1)
                yield batch


class ProgressDataLoaderFactory:
    """Drop-in callable replacement for torch.utils.data.DataLoader.

    The official training module creates the train loader first and the
    validation loader second.  Keeping that distinction here avoids changing
    the upstream training loop or its original output.
    """

    def __init__(self, *, train_epochs: int | None = None, standalone_eval: bool = False) -> None:
        self.train_epochs = train_epochs
        self.standalone_eval = standalone_eval
        self._created = 0

    def __call__(self, *args: Any, **kwargs: Any) -> ProgressDataLoader:
        loader = TorchDataLoader(*args, **kwargs)
        if self.standalone_eval:
            def desc_factory(iteration: int) -> str:
                del iteration
                return "Evaluation"
        elif self._created == 0:
            total = "?" if self.train_epochs is None else str(self.train_epochs)

            def desc_factory(iteration: int) -> str:
                return f"Epoch: {iteration}/{total}"
        else:
            def desc_factory(iteration: int) -> str:
                del iteration
                return "Validation"
        self._created += 1
        return ProgressDataLoader(loader, desc_factory=desc_factory)


def install_train_progress(train_module: Any, *, train_epochs: int | None) -> None:
    """Wrap official WCM training DataLoaders with rank-zero progress bars."""

    train_module.DataLoader = ProgressDataLoaderFactory(train_epochs=train_epochs)


def install_eval_progress(eval_module: Any) -> None:
    """Wrap official WCM standalone evaluation DataLoader."""

    eval_module.DataLoader = ProgressDataLoaderFactory(standalone_eval=True)
