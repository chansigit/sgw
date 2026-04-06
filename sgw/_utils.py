import gc
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_gc(do_cuda: bool = True):
    gc.collect()
    if do_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
