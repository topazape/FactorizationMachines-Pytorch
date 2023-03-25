import torch
import torch.backends.mps


def get_device() -> str:
    """Get the device to use."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device
