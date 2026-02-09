import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    """A feedforward neural network for image classification."""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], output_size=10):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.network(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 0.001
    USE_FASHION_MNIST = True  # Set to False for regular MNIST

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    dataset_class = datasets.FashionMNIST if USE_FASHION_MNIST else datasets.MNIST
    dataset_name = "Fashion-MNIST" if USE_FASHION_MNIST else "MNIST"

    print(f"Loading {dataset_name} dataset...")

    train_dataset = dataset_class(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = dataset_class(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class labels
    if USE_FASHION_MNIST:
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        classes = [str(i) for i in range(10)]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = NeuralNetwork(
        input_size=784,      # 28x28 images
        hidden_sizes=[512, 256, 128],
        output_size=10       # 10 classes
    ).to(device)

    print("\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    best_accuracy = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("-" * 60)
    print(f"Best test accuracy: {best_accuracy:.2f}%")

    # Load best model and show sample predictions
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    print("\nSample predictions:")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images[:10])
        _, predictions = outputs.max(1)

    for i in range(10):
        actual = classes[labels[i].item()]
        predicted = classes[predictions[i].item()]
        status = "✓" if labels[i] == predictions[i] else "✗"
        print(f"  {status} Actual: {actual:12s} | Predicted: {predicted}")


if __name__ == "__main__":
    main()
