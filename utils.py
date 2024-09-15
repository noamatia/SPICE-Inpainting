import py3nvml.py3nvml as nvml


def find_free_gpu():
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    max_free_memory = 0
    best_gpu_index = 0

    for i in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = memory_info.free

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu_index = i

    nvml.nvmlShutdown()
    return str(best_gpu_index)
