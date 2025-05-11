from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

def print_gpu_utilization(device):
    if device != 'cpu':
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        memory_used = info.used / 1024**3
        memory_total = info.total / 1024**3
        print(f"GPU memory occupied: {memory_used:.3f}/{memory_total:.3f} GB.")
        return memory_used, memory_total


def print_summary(result, device):
    if device != 'cpu':
        print(f"Time: {result.metrics['train_runtime']:.3f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()