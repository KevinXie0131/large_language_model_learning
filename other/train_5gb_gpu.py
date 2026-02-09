import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import math


class DeepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def generate_data(total_bytes=5 * 1024 ** 3, input_dim=512, num_classes=10):
    """Generate ~5GB of synthetic classification data."""
    bytes_per_sample = input_dim * 4 + 8  # float32 features + int64 label
    num_samples = total_bytes // bytes_per_sample
    print(f"Generating {num_samples:,} samples ({input_dim} features each)")
    print(f"Total data size: {num_samples * bytes_per_sample / 1e9:.2f} GB")

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train(epochs=10, batch_size=8192, input_dim=512, hidden_dim=1024, num_classes=10, lr=1e-3):
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device = torch.device("cuda")
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = gpu_props.total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {gpu_mem_gb:.2f} GB")

    # Generate 5GB data on CPU
    X, y = generate_data(total_bytes=5 * 1024 ** 3, input_dim=input_dim, num_classes=num_classes)

    # Move entire dataset to GPU
    print("\nMoving data to GPU...")
    transfer_start = time.perf_counter()
    X = X.to(device)
    y = y.to(device)
    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - transfer_start
    print(f"Data transferred to GPU in {transfer_time:.2f}s")
    print(f"GPU memory after data load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Batches per epoch: {len(loader):,}  |  Batch size: {batch_size:,}")

    # Move model to GPU
    print("\nMoving model to GPU...")
    model = DeepNet(input_dim, hidden_dim, num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}")

    best_loss = float("inf")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.perf_counter()

        for batch_idx, (features, labels) in enumerate(loader):
            # Data is already on GPU, no transfer needed
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"GPU Mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        scheduler.step()

        epoch_loss /= total
        accuracy = 100.0 * correct / total
        epoch_time = time.perf_counter() - epoch_start
        samples_per_sec = total / epoch_time

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_model_5gb.pth")

        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s | {samples_per_sec:,.0f} samples/s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    total_time = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to best_model_5gb.pth")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    train()
