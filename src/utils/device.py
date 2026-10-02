"""Choose a CPU or CUDA device consistently across experiments."""

from importlib import import_module
from typing import Literal


def get_device(preference: str = "auto") -> Literal["cpu", "cuda"]:
    """Return a device string usable by ``model.to(device)``.

    ``auto`` prefers CUDA when PyTorch reports it available, otherwise CPU.
    ``cpu`` never imports PyTorch. An absent optional PyTorch installation
    means CPU in auto mode; an explicit CUDA request raises RuntimeError.
    Broken PyTorch installations and CUDA probe errors propagate to callers.
    GPU indices and backends other than CPU/CUDA are intentionally unsupported.
    """
    if not isinstance(preference, str):
        raise TypeError("device preference must be a string")
    preference = preference.strip().lower()
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device preference must be 'auto', 'cpu', or 'cuda'")
    if preference == "cpu":
        return "cpu"

    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        if preference == "cuda":
            raise RuntimeError(
                "CUDA requested but PyTorch is not installed; install the ml extra."
            ) from exc
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError(
            "CUDA requested but unavailable; check the GPU, driver, and PyTorch build."
        )
    return "cpu"
