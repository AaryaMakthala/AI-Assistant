"""One-time CPU/thread runtime configuration for the local ML stack.

The retrieval models run through PyTorch on the CPU: the Render free instance
has no GPU, and probing for CUDA every request is wasted work (and wastes
memory while the CUDA loader runs).  Before the first model use this module
pins torch to ``device="cpu"`` inference and caps thread usage to
``TORCH_NUM_THREADS`` (default 2) so a ~0.1 CPU instance is not
oversubscribed by OpenMP/MKL/OpenBLAS pools.  All side effects happen
exactly once per process; later calls return immediately.
"""

from __future__ import annotations

import os
import threading

_TORCH_NUM_THREADS_DEFAULT = 2

_configured = False
_config_lock = threading.Lock()


def apply_torch_runtime_config() -> None:
    """Pin torch CPU threading before inference, exactly once per process.

    The thread cap is read from ``TORCH_NUM_THREADS`` once and applied to torch
    and to the BLAS/OpenMP thread pools via environment variables, which must be
    in place before those pools initialize.  Idempotent and lock-guarded so two
    concurrent first-use requests cannot configure it twice or race the import.
    """
    global _configured
    if _configured:
        return

    threads = os.environ.get("TORCH_NUM_THREADS") or str(_TORCH_NUM_THREADS_DEFAULT)

    # Cap the underlying thread pools via env before torch initializes them.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, threads)

    with _config_lock:
        if _configured:
            return
        import torch

        torch.set_num_threads(int(threads))
        _configured = True


__all__ = ["apply_torch_runtime_config"]