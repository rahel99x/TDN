"""Slurm stop handlers only set flags; checkpointing happens between updates."""
from __future__ import annotations

import signal
import os
from dataclasses import dataclass, field


@dataclass
class StopRequest:
    requested: bool = False
    signal_number: int | None = None
    previous: dict = field(default_factory=dict, repr=False)

    def _handler(self, number: int, _frame) -> None:
        self.requested = True
        self.signal_number = number

    def install(self) -> "StopRequest":
        names = ["SIGUSR1", "SIGTERM"]
        if os.environ.get("TDN_EXECUTION_MODE") == "desktop":
            names.extend(("SIGINT", "SIGBREAK"))
        for name in names:
            number = getattr(signal, name, None)
            if number is not None:
                self.previous[number] = signal.getsignal(number)
                signal.signal(number, self._handler)
        return self

    def restore(self) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        self.previous.clear()

    def __enter__(self) -> "StopRequest":
        return self.install()

    def __exit__(self, *_args) -> None:
        self.restore()


def ignore_worker_warning(_worker_id: int = 0) -> None:
    """For an explicitly enabled worker pool; the reference loop uses none."""
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    if os.environ.get("TDN_EXECUTION_MODE") == "desktop":
        # The parent owns safe-boundary checkpointing after a console stop.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
