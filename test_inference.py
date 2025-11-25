#!/usr/bin/env python3
"""
Test TinyCNN MNIST model inference in Python.

This script validates the model by running inference on MNIST test data
and saves sample images for C++ testing.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np
from model import TinyCNN


def load_model(model_path: str):
    """Load the PyTorch model."""
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if it's a state dict or full model
    if isinstance(checkpoint, dict):
        print("  ✓ Detected state_dict format")
        model = TinyCNN()
        model.load_state_dict(checkpoint)
        print("  ✓ Weights loaded into TinyCNN architecture")
    else:
        model = checkpoint
        print("  ✓ Loaded full model object")

    model.eval()
    return model


def prepare_mnist_testset(data_dir: str = './data'):
    """
    Download and prepare MNIST test dataset.

    Args:
        data_dir: Directory to store MNIST data

    Returns:
        DataLoader for test set
    """
    print(f"Preparing MNIST test dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    print(f"  ✓ Loaded {len(test_dataset)} test images")
    return test_loader, test_dataset


def evaluate_model(model, test_loader):
    """
    Evaluate model accuracy on test set.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test data

    Returns:
        Accuracy percentage
    """
    print("\nEvaluating model on test set...")

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"  ✓ Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return accuracy


def test_single_inference(model, image, label):
    """
    Test inference on a single image.

    Args:
        model: PyTorch model
        image: Input image tensor [1, 28, 28]
        label: Ground truth label

    Returns:
        Predicted label and confidence scores
    """
    with torch.no_grad():
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item(), probabilities.squeeze()


def save_sample_images(test_dataset, output_dir: str = './tests/sample_images', num_samples: int = 10):
    """
    Save sample images for C++ testing.

    Args:
        test_dataset: MNIST test dataset
        output_dir: Directory to save images
        num_samples: Number of samples to save
    """
    print(f"\nSaving sample images to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save samples for each digit (0-9)
    digit_counts = {i: 0 for i in range(10)}
    max_per_digit = 2

    for idx, (image, label) in enumerate(test_dataset):
        label_int = label if isinstance(label, int) else label.item()

        if digit_counts[label_int] < max_per_digit:
            # Convert normalized tensor back to image
            # Denormalize: img = (img * std) + mean
            img_denorm = image.squeeze() * 0.3081 + 0.1307
            img_denorm = torch.clamp(img_denorm, 0, 1)

            # Convert to PIL Image
            img_array = (img_denorm.numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array, mode='L')

            # Save image
            filename = f"digit_{label_int}_sample_{digit_counts[label_int]}.png"
            filepath = os.path.join(output_dir, filename)
            pil_image.save(filepath)

            digit_counts[label_int] += 1
            print(f"  ✓ Saved: {filename}")

        # Stop if we have enough samples
        if all(count >= max_per_digit for count in digit_counts.values()):
            break

    print(f"  ✓ Total samples saved: {sum(digit_counts.values())}")


def demonstrate_inference(model, test_dataset, num_examples: int = 5):
    """
    Demonstrate inference on sample images with detailed output.

    Args:
        model: PyTorch model
        test_dataset: MNIST test dataset
        num_examples: Number of examples to show
    """
    print("\n" + "=" * 80)
    print("Demonstration: Single Image Inference")
    print("=" * 80)

    for i in range(num_examples):
        image, label = test_dataset[i]
        predicted, confidence, probs = test_single_inference(model, image, label)

        print(f"\nExample {i + 1}:")
        print(f"  Ground Truth:    {label}")
        print(f"  Predicted:       {predicted} {'✓' if predicted == label else '✗'}")
        print(f"  Confidence:      {confidence * 100:.2f}%")
        print(f"  Top-3 predictions:")

        top3_probs, top3_indices = torch.topk(probs, 3)
        for rank, (prob, idx) in enumerate(zip(top3_probs, top3_indices), 1):
            print(f"    {rank}. Digit {idx.item()}: {prob.item() * 100:.2f}%")


def main():
    model_path = "tinycnn_mnist.pt"

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Make sure 'tinycnn_mnist.pt' exists and is a valid PyTorch model.")
        return

    # Prepare test data
    try:
        test_loader, test_dataset = prepare_mnist_testset()
    except Exception as e:
        print(f"❌ Error loading MNIST data: {e}")
        return

    # Evaluate on full test set
    accuracy = evaluate_model(model, test_loader)

    # Demonstrate inference on samples
    demonstrate_inference(model, test_dataset)

    # Save sample images for C++ testing
    save_sample_images(test_dataset)

    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"  • Model accuracy: {accuracy:.2f}%")
    print(f"  • Sample images saved to: ./tests/sample_images/")
    print(f"\n💡 Next steps:")
    print(f"  1. Review sample images in ./tests/sample_images/")
    print(f"  2. Build C++ application with LibTorch")
    print(f"  3. Test C++ inference using saved images")


if __name__ == "__main__":
    main()
