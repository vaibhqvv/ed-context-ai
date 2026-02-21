import torch
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEVICE = None


def get_device() -> torch.device:
    """Return the best available torch device (CUDA GPU if available, else CPU).

    The result is cached after the first call so the log message only fires once.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        _DEVICE = torch.device("cpu")
        log.info("CUDA not available — using CPU")
    return _DEVICE
