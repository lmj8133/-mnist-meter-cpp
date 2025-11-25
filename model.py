"""
TinyCNN model architecture for MNIST digit classification.

This module defines the neural network architecture that matches
the saved state_dict in tinycnn_mnist.pt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """
    Lightweight CNN for MNIST digit classification.

    Architecture:
        Input: 1×28×28 grayscale image

        Features (Convolutional layers):
        - Conv2d(1→8, kernel=3×3, padding=1) → ReLU → MaxPool2d(2×2)
          Output: 8×14×14
        - Conv2d(8→16, kernel=3×3, padding=1) → ReLU → MaxPool2d(2×2)
          Output: 16×7×7

        Classifier (Fully connected layers):
        - Flatten: 16×7×7 = 784
        - Linear(784→64) → ReLU → Dropout(0.5)
        - Linear(64→10)

        Output: 10 class logits (digits 0-9)

    Total parameters: ~51,000
    """

    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize TinyCNN model.

        Args:
            dropout_rate: Dropout probability for regularization (default: 0.5)
        """
        super(TinyCNN, self).__init__()

        # Feature extraction (convolutional layers)
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1×28×28 → 8×28×28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 8×28×28 → 8×14×14

            # Second convolutional block
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # 8×14×14 → 16×14×14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 16×14×14 → 16×7×7
        )

        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),                    # 784 → 64
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 10),                            # 64 → 10
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Extract features
        x = self.features(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 16, 7, 7) → (batch, 784)

        # Classify
        x = self.classifier(x)

        return x

    def predict(self, x):
        """
        Predict class probabilities.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        return predicted_classes, probabilities


def create_model(pretrained_path: str = None) -> TinyCNN:
    """
    Create TinyCNN model instance.

    Args:
        pretrained_path: Path to pretrained state_dict file (optional)

    Returns:
        TinyCNN model instance
    """
    model = TinyCNN()

    if pretrained_path:
        print(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("✓ Weights loaded successfully")

    return model


if __name__ == "__main__":
    # Test model architecture
    print("Testing TinyCNN architecture...")
    print("=" * 80)

    model = TinyCNN()
    model.eval()

    # Print model structure
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           {total_params * 4 / 1024:.2f} KB (float32)")

    # Test forward pass
    print(f"\nTesting forward pass:")
    dummy_input = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test prediction method
    predicted_classes, probabilities = model.predict(dummy_input)
    print(f"\nTesting predict method:")
    print(f"  Predicted class: {predicted_classes.item()}")
    print(f"  Class probabilities: {probabilities.squeeze().numpy()}")

    print("\n✅ Architecture test passed!")
