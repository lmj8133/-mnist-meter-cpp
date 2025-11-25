#!/usr/bin/env python3
"""
Generate simple test images for C++ inference testing.

This script creates synthetic digit images without requiring
the full MNIST dataset download.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_simple_digit_image(digit: int, output_path: str):
    """
    Create a simple 28x28 grayscale image with a digit.

    Args:
        digit: Digit to render (0-9)
        output_path: Path to save the image
    """
    # Create white background
    img = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fallback to default if not available
    try:
        # Try to find a suitable font
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)
                break

        if font is None:
            font = ImageFont.load_default()

    except Exception:
        font = ImageFont.load_default()

    # Draw the digit in black, centered
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (28 - text_width) // 2
    y = (28 - text_height) // 2

    draw.text((x, y), text, fill=0, font=font)

    # Save image
    img.save(output_path)


def main():
    output_dir = "./tests/sample_images"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating simple test images...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Generate one image for each digit
    for digit in range(10):
        filename = f"digit_{digit}_simple.png"
        filepath = os.path.join(output_dir, filename)
        create_simple_digit_image(digit, filepath)
        print(f"  ✓ Created: {filename}")

    print("-" * 60)
    print(f"✅ Generated {10} test images")
    print(f"\n💡 Note: These are simple synthetic images.")
    print(f"   For real MNIST test images, run: uv run python test_inference.py")


if __name__ == "__main__":
    main()
