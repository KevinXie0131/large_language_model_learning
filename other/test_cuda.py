import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Print the number of available GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # Print the name of the current GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Print the CUDA version PyTorch was compiled with
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
else:
    print("PyTorch is running on CPU.")