import torch
import threading
import time

def keep_gpu_busy():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting thread.")
        return

    # Adjust the following parameters based on your GPU's capacity
    tensor_size = (4000, 4000)  # Start with a medium-sized tensor
    num_operations = 75  # Start with a moderate number of operations

    while True:
        tensor = torch.randn(*tensor_size, device="cuda")
        for _ in range(num_operations):
            tensor = tensor * torch.randn(*tensor_size, device="cuda")
        
        # Short sleep to allow for some idle time
        time.sleep(0.1)

# Create a new thread that runs the GPU-busy function
thread = threading.Thread(target=keep_gpu_busy)
thread.start()
