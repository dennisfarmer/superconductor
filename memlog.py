"""Periodic memory log for mrt2_server.py: process RAM and model (GPU) memory.

One line every `interval` seconds, from a daemon thread, e.g.
  [mem] ram 2.31G (peak 2.40G) | gpu 5.12G (peak 6.03G) of 59.4G reserved

  ram   process resident set size (psutil, if installed) and its peak
        (getrusage, always available)
  gpu   jax: bytes the model's buffers use on the device, against the pool
        XLA reserved up front (by default 75% of the card, so nvidia-smi
        shows the reservation, not what the model needs; `peak` is the need)
        mlx: Metal buffers in use, their peak, and the allocator's free cache;
        on Apple Silicon this is unified memory, i.e. part of the same RAM
"""

import resource
import sys
import threading
import time

_GiB = 1024**3


def _ram():
    # ru_maxrss is bytes on macOS, KiB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    try:
        import psutil
        return f"ram {psutil.Process().memory_info().rss / _GiB:.2f}G (peak {peak / _GiB:.2f}G)"
    except ImportError:
        return f"ram peak {peak / _GiB:.2f}G"


def _gpu(backend):
    if backend == "jax":
        import jax
        parts = []
        for d in jax.local_devices():
            s = d.memory_stats()
            if not s:  # CPU devices have no stats
                continue
            parts.append(
                f"{d.platform}{d.id} {s['bytes_in_use'] / _GiB:.2f}G "
                f"(peak {s['peak_bytes_in_use'] / _GiB:.2f}G) "
                f"of {s.get('bytes_limit', 0) / _GiB:.1f}G reserved"
            )
        return " | ".join(parts) or "gpu n/a"

    import mlx.core as mx
    return (
        f"gpu {mx.get_active_memory() / _GiB:.2f}G "
        f"(peak {mx.get_peak_memory() / _GiB:.2f}G, cache {mx.get_cache_memory() / _GiB:.2f}G)"
    )


def line(backend):
    try:
        gpu = _gpu(backend)
    except Exception as e:  # never let logging take the server down
        gpu = f"gpu error: {e}"
    return f"[mem] {_ram()} | {gpu}"


def start(backend, interval):
    """Log memory every `interval` seconds; interval <= 0 disables."""
    if interval <= 0:
        return

    def run():
        while True:
            print(line(backend), flush=True)
            time.sleep(interval)

    threading.Thread(target=run, daemon=True, name="memlog").start()
