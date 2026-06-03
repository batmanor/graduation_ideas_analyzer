from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import threading
import time
from typing import Iterator


@dataclass
class TimingStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None
    last_ms: float | None = None

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def record(self, elapsed_s: float) -> None:
        with self._lock:
            elapsed_ms = elapsed_s * 1000
            self.count += 1
            self.total_ms += elapsed_ms
            self.last_ms = elapsed_ms
            self.min_ms = (
                elapsed_ms if self.min_ms is None else min(self.min_ms, elapsed_ms)
            )
            self.max_ms = (
                elapsed_ms if self.max_ms is None else max(self.max_ms, elapsed_ms)
            )

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            avg_ms = self.total_ms / self.count if self.count else None
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(avg_ms, 3) if avg_ms is not None else None,
            "min_ms": round(self.min_ms, 3) if self.min_ms is not None else None,
            "max_ms": round(self.max_ms, 3) if self.max_ms is not None else None,
            "last_ms": round(self.last_ms, 3) if self.last_ms is not None else None,
        }

    def clear(self) -> None:
        with self._lock:
            self.count = 0
            self.total_ms = 0.0
            self.min_ms = None
            self.max_ms = None
            self.last_ms = None


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._timings: dict[str, TimingStats] = {}

    def record_timing(self, name: str, elapsed_s: float) -> None:
        stats = self._timings.get(name)

        if stats is None:
            with self._lock:
                stats = self._timings.setdefault(name, TimingStats())

        stats.record(elapsed_s)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            timings = {
                name: stats.snapshot() for name, stats in sorted(self._timings.items())
            }

        return {
            "process": {
                "pid": os.getpid(),
                "rss_mb": get_rss_mb(),
            },
            "timings": timings,
        }

    def clear(self):
        self._timings.clear()


def get_rss_mb() -> float | None:
    psutil_rss = _get_psutil_rss_mb()
    if psutil_rss is not None:
        return psutil_rss

    if os.name == "nt":
        return _get_windows_rss_mb()

    return _get_resource_rss_mb()


def _get_psutil_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None

    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)


def _get_windows_rss_mb() -> float | None:
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return round(counters.WorkingSetSize / (1024 * 1024), 3)
    except Exception:
        return None


def _get_resource_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        return None

    if os.uname().sysname == "Darwin":
        return round(rss / (1024 * 1024), 3)
    return round(rss / 1024, 3)


@contextmanager
def track_timing(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.record_timing(name, time.perf_counter() - start)


metrics = MetricsRegistry()
