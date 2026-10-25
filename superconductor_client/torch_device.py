"""Which torch device this process can allocate.

Several processes can share an L40S when its compute mode is Default.
`is_available()` stays true when the driver still will not open a context,
so the check is allocating a tensor. `cudaErrorDevicesUnavailable` is that
refusal: the card is in Exclusive_Process, and the first process to use it
keeps it. This falls back to CPU instead of crashing.
"""

_told = set()


def cuda_unavailable(exc):
    """The driver sees a GPU and will not give it to this process."""
    text = str(exc).lower()
    return ("busy or unavailable" in text or "devicesunavailable" in text
            or "cuda-capable device" in text)


def tell(what):
    if what in _told:
        return
    _told.add(what)
    print(f"The GPU is in exclusive mode, so this process cannot open it. {what} will run on CPU. "
          "nvidia-smi shows the compute mode; Default lets the L40S be shared.",
          flush=True)


DESCRIBE_NEEDS_GPU = (
    "The describe server needs a GPU, and none is available. "
    "The L40S is in Exclusive_Process and already taken, or this machine has no CUDA device."
)


def cuda_state_detached():
    """Whether a new process can allocate on CUDA.

    The check exits with that process, so a success here does not leave this
    process holding the L40S under Exclusive_Process.
    """
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-c", "import torch; torch.ones(1, device='cuda').add_(1).item()"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return "ok"
    if cuda_unavailable(f"{proc.stderr}\n{proc.stdout}"):
        return "busy"
    return "absent"


def cuda_state():
    """'ok' if a tensor can be allocated, 'busy' if the driver refuses, 'absent' otherwise."""
    import torch
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        return "absent"
    if not available:
        return "absent"
    try:
        torch.ones(1, device="cuda").add_(1).item()
    except Exception as exc:
        if cuda_unavailable(exc):
            return "busy"
        raise
    return "ok"


def _mps_available():
    import torch
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False


def resolve_device(device=None, what="This model"):
    """cpu stays cpu. Otherwise cuda, then mps, then cpu.

    Exclusive_Process counts as unavailable: the first process already has the L40S.
    """
    if device == "cpu":
        return "cpu"
    has_mps = _mps_available()
    cuda = cuda_state()
    if device == "cuda" and cuda == "ok":
        return "cuda"
    if device == "mps" and has_mps:
        return "mps"
    if cuda == "ok":
        return "cuda"
    if has_mps:
        return "mps"
    if cuda == "busy":
        tell(what)
    return "cpu"


def move_module(module, device, what):
    """`module.to(device)`, or CPU when Exclusive_Process refuses this process."""
    try:
        module.to(device)
        return device
    except Exception as exc:
        if device == "cpu" or not cuda_unavailable(exc):
            raise
        tell(what)
        module.to("cpu")
        return "cpu"
