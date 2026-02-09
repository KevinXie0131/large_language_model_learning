import torch
import time

def main():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create ~1GB of float32 data (1GB / 4 bytes per float32 = 256M elements)
    num_elements = 256 * 1024 * 1024  # 256M floats = 1GB
    print(f"\nAllocating {num_elements * 4 / 1e9:.2f} GB of float32 data on CPU...")

    cpu_data = torch.randn(num_elements)
    print(f"CPU tensor size: {cpu_data.nelement() * cpu_data.element_size() / 1e9:.2f} GB")

    # Transfer to GPU
    print("\nTransferring data to GPU...")
    start = time.perf_counter()
    gpu_data = cpu_data.to(device)
    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - start
    print(f"Transfer time: {transfer_time:.3f}s")
    print(f"Transfer speed: {1.0 / transfer_time:.2f} GB/s")

    # Run computations on GPU
    print("\n--- GPU Computations ---")

    # 1. Element-wise multiply
    start = time.perf_counter()
    result = gpu_data * 2.0
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Element-wise multiply: {elapsed:.4f}s")

    # 2. Sum reduction
    start = time.perf_counter()
    total = gpu_data.sum()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Sum reduction:         {elapsed:.4f}s  (result: {total.item():.2f})")

    # 3. Sort
    start = time.perf_counter()
    sorted_data, _ = gpu_data.sort()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Sort:                  {elapsed:.4f}s")

    # 4. Matrix multiply (reshape to 2D)
    side = int(num_elements ** 0.5)  # ~16K x 16K
    mat = gpu_data[:side * side].reshape(side, side)
    start = time.perf_counter()
    mat_result = torch.mm(mat, mat)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Matrix multiply ({side}x{side}): {elapsed:.4f}s")

    # 5. FFT
    start = time.perf_counter()
    fft_result = torch.fft.fft(gpu_data)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"FFT:                   {elapsed:.4f}s")

    # Memory usage
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Cleanup
    del gpu_data, result, sorted_data, mat, mat_result, fft_result
    torch.cuda.empty_cache()
    print("\nGPU memory freed.")


if __name__ == "__main__":
    main()
