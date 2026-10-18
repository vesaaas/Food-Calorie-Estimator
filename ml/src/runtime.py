"""Centralized device policy and training reproducibility."""

import torch

if __package__:
    from .dataset import DataConfig, seed_random_generators
else:
    from dataset import DataConfig, seed_random_generators


def select_device(requested: str = "auto") -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU; fail on unavailable explicit devices."""
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        requested = next(name for name, supported in available.items() if supported)
    if not available[requested]:
        raise RuntimeError(f"Requested device '{requested}' is unavailable in this PyTorch runtime")
    return torch.device(requested)


def seed_training(random_seed: int = DataConfig().random_seed) -> None:
    """Reuse Phase 2 seeding (Python, NumPy, and PyTorch including accelerators).

    Disable cuDNN autotuning for repeatability. Bitwise identical results across
    different backends/versions and exact interrupted-run resumption are not promised.
    """
    if not 0 <= random_seed < 2**32:
        raise ValueError("random_seed must be in [0, 2**32)")
    seed_random_generators(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
