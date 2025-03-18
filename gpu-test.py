import torch

print(torch.cuda.is_available())  # Returns True if CUDA is available (GPU is accessible)
print(torch.cuda.current_device())  # Displays the current GPU device id
print(torch.cuda.get_device_name(0))  # Displays the name of the GPU device
