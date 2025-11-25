#!/usr/bin/env python3
"""
Analyze TinyCNN MNIST model architecture and export for C++ inference.

This script loads the PyTorch model, displays its architecture,
and exports it to TorchScript format for LibTorch usage.
"""

import torch
import torch.nn as nn
from model import TinyCNN


def analyze_model(model_path: str):
    """
    Load and analyze the PyTorch model.

    Args:
        model_path: Path to the .pt model file
    """
    print(f"Loading model from: {model_path}")
    print("=" * 80)

    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict):
            print("\n✓ Detected state_dict format")
            print("\nState dict keys:")
            for key, value in checkpoint.items():
                print(f"  {key:40s} {str(value.shape):20s}")

            print("\nLoading weights into TinyCNN architecture...")
            model = TinyCNN()
            model.load_state_dict(checkpoint)
            print("✓ Weights loaded successfully")
        else:
            # Full model object
            model = checkpoint
            print("\n✓ Loaded full model object")

        # Set to evaluation mode
        model.eval()

        # Print model architecture
        print("\n📋 Model Architecture:")
        print("-" * 80)
        print(model)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n📊 Model Statistics:")
        print("-" * 80)
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size:           {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

        # Print detailed layer information
        print(f"\n🔍 Layer Details:")
        print("-" * 80)
        for name, param in model.named_parameters():
            print(f"  {name:40s} {str(param.shape):20s} {param.numel():>10,} params")

        # Test forward pass with dummy input
        print(f"\n🧪 Testing Forward Pass:")
        print("-" * 80)
        dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input: [batch, channels, height, width]

        with torch.no_grad():
            output = model(dummy_input)

        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

        # Export to TorchScript
        print(f"\n💾 Exporting to TorchScript:")
        print("-" * 80)

        traced_model = torch.jit.trace(model, dummy_input)
        output_path = model_path.replace('.pt', '_traced.pt')
        traced_model.save(output_path)

        print(f"  ✓ Saved TorchScript model to: {output_path}")
        print(f"  This file can be loaded by LibTorch in C++")

        return model

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return None


def main():
    model_path = "best_model_mnist_and_water_meter.pt"
    model = analyze_model(model_path)

    if model is not None:
        print(f"\n✅ Analysis complete!")
        print(f"\n💡 Next steps:")
        print(f"  1. Use the *_traced.pt model for C++ inference")
        print(f"  2. Run test_inference.py to validate Python inference")
        print(f"  3. Build C++ application with LibTorch")
    else:
        print(f"\n💡 If model is a state_dict, you need to:")
        print(f"  1. Define the model architecture class")
        print(f"  2. Instantiate the model")
        print(f"  3. Load state_dict with model.load_state_dict()")


if __name__ == "__main__":
    main()
