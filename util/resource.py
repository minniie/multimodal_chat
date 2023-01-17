import torch


GB_FACTOR = 1024 * 1024 * 1024


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_device_util():
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        print (
            f"> device\n{torch.cuda.get_device_name(gpu_idx)}\n"
            f"> total memory\n{torch.cuda.get_device_properties(gpu_idx).total_memory / GB_FACTOR:.3f}\n"
            f"> allocated memory\n{torch.cuda.memory_allocated() / GB_FACTOR:.3f}"
        )