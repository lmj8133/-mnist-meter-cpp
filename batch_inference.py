#!/usr/bin/env python3
"""
Batch inference on categorized test images.

This script performs inference on images organized in digit-labeled folders
(tests/0/, tests/1/, ..., tests/9/) and reports per-class accuracy.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import time
from model import TinyCNN


class BatchInference:
    """Batch inference engine with statistics tracking."""

    def __init__(self, model_path: str):
        """
        Initialize batch inference engine.

        Args:
            model_path: Path to model file (.pt)
        """
        self.model_path = model_path
        self.model = None
        self.results = {
            'total': 0,
            'correct': 0,
            'per_class': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'confusion_matrix': np.zeros((10, 10), dtype=int),
            'errors': [],
            'inference_times': []
        }

    def load_model(self):
        """Load PyTorch model."""
        print(f"Loading model from: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint, dict):
            print("  ✓ Detected state_dict format")
            self.model = TinyCNN()
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model.eval()
        print("  ✓ Model loaded and ready for inference\n")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor [1, 1, 28, 28]
        """
        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale

        # Resize to 28x28 if needed
        if img.size != (28, 28):
            img = img.resize((28, 28), Image.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Apply MNIST normalization
        #mean = 0.1307
        #std = 0.3081
        mean = 0.5
        std = 0.5
        img_array = (img_array - mean) / std

        # Convert to tensor
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

        return tensor

    def predict(self, image_path: str) -> tuple:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (predicted_digit, confidence, inference_time)
        """
        start_time = time.time()

        # Preprocess
        input_tensor = self.preprocess_image(image_path)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = time.time() - start_time

        return predicted.item(), confidence.item(), inference_time

    def process_directory(self, test_dir: str, show_progress: bool = True):
        """
        Process all images in categorized directories.

        Args:
            test_dir: Root directory containing digit folders (0-9)
            show_progress: Show progress during processing
        """
        test_path = Path(test_dir)

        if not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        print(f"Processing test images from: {test_dir}")
        print("=" * 80)

        # Process each digit folder (0-9)
        for digit in range(10):
            digit_dir = test_path / str(digit)

            if not digit_dir.exists():
                print(f"⚠️  Warning: Directory {digit_dir} not found, skipping...")
                continue

            # Get all image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_files.extend(digit_dir.glob(ext))

            if not image_files:
                print(f"⚠️  Warning: No images found in {digit_dir}, skipping...")
                continue

            # Process images
            for idx, img_path in enumerate(image_files):
                try:
                    predicted, confidence, inf_time = self.predict(str(img_path))

                    # Update statistics
                    self.results['total'] += 1
                    self.results['per_class'][digit]['total'] += 1
                    self.results['confusion_matrix'][digit][predicted] += 1
                    self.results['inference_times'].append(inf_time)

                    if predicted == digit:
                        self.results['correct'] += 1
                        self.results['per_class'][digit]['correct'] += 1
                    else:
                        # Record error
                        self.results['errors'].append({
                            'image': str(img_path),
                            'true_label': digit,
                            'predicted': predicted,
                            'confidence': confidence
                        })

                    # Show progress
                    if show_progress and (idx + 1) % 50 == 0:
                        print(f"  Digit {digit}: Processed {idx + 1}/{len(image_files)} images", end='\r')

                except Exception as e:
                    print(f"\n⚠️  Error processing {img_path}: {e}")

            if show_progress:
                total_digit = self.results['per_class'][digit]['total']
                correct_digit = self.results['per_class'][digit]['correct']
                accuracy_digit = 100.0 * correct_digit / total_digit if total_digit > 0 else 0
                print(f"  Digit {digit}: {correct_digit}/{total_digit} "
                      f"({accuracy_digit:.2f}%) ✓")

        print("\n" + "=" * 80)

    def print_report(self):
        """Print detailed inference report."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 25 + "BATCH INFERENCE REPORT" + " " * 31 + "║")
        print("╚" + "=" * 78 + "╝\n")

        # Overall accuracy
        overall_acc = 100.0 * self.results['correct'] / self.results['total']
        print(f"📊 Overall Statistics:")
        print(f"   Total images:     {self.results['total']:,}")
        print(f"   Correct:          {self.results['correct']:,}")
        print(f"   Incorrect:        {self.results['total'] - self.results['correct']:,}")
        print(f"   Accuracy:         {overall_acc:.2f}%")

        # Performance
        avg_time = np.mean(self.results['inference_times']) * 1000
        print(f"\n⚡ Performance:")
        print(f"   Avg inference:    {avg_time:.2f} ms/image")
        print(f"   Throughput:       {1000.0 / avg_time:.1f} images/sec")

        # Per-class accuracy
        print(f"\n📈 Per-Class Accuracy:")
        print("   " + "─" * 60)
        print(f"   {'Digit':<8} {'Correct/Total':<20} {'Accuracy':<15} {'Bar'}")
        print("   " + "─" * 60)

        for digit in range(10):
            stats = self.results['per_class'][digit]
            if stats['total'] > 0:
                acc = 100.0 * stats['correct'] / stats['total']
                bar_len = int(acc / 2)  # Scale to 50 chars max
                bar = "█" * bar_len + "░" * (50 - bar_len)
                print(f"   {digit:<8} {stats['correct']:>5}/{stats['total']:<5}        "
                      f"{acc:>6.2f}%        [{bar}]")
            else:
                print(f"   {digit:<8} {'N/A':<20} {'N/A':<15}")

        # Confusion Matrix
        print(f"\n🔄 Confusion Matrix:")
        print("   (Rows: True labels, Columns: Predicted labels)")
        print("   " + "─" * 65)
        print("      ", end="")
        for i in range(10):
            print(f"{i:>5}", end=" ")
        print()
        print("   " + "─" * 65)

        for i in range(10):
            print(f"   {i:>2} ", end="")
            for j in range(10):
                count = self.results['confusion_matrix'][i][j]
                if i == j:
                    # Diagonal (correct predictions) - highlighted
                    print(f"\033[1;32m{count:>5}\033[0m", end=" ")
                elif count > 0:
                    # Errors - highlighted in red
                    print(f"\033[1;31m{count:>5}\033[0m", end=" ")
                else:
                    print(f"{count:>5}", end=" ")
            print()

        # Error analysis
        if self.results['errors']:
            print(f"\n❌ Error Analysis (Total: {len(self.results['errors'])}):")
            print("   " + "─" * 60)

            # Group errors by true label
            errors_by_class = defaultdict(list)
            for error in self.results['errors']:
                errors_by_class[error['true_label']].append(error)

            for digit in range(10):
                if digit in errors_by_class:
                    errors = errors_by_class[digit]
                    # Count most common misclassifications
                    misclass_counts = defaultdict(int)
                    for err in errors:
                        misclass_counts[err['predicted']] += 1

                    top_misclass = sorted(misclass_counts.items(),
                                          key=lambda x: x[1], reverse=True)[:3]

                    print(f"   Digit {digit}: {len(errors)} errors")
                    for pred_digit, count in top_misclass:
                        print(f"      → Often misclassified as {pred_digit}: {count} times")

            # Show some error examples
            print(f"\n   Top 5 Error Examples:")
            sorted_errors = sorted(self.results['errors'],
                                   key=lambda x: x['confidence'], reverse=True)[:5]

            for i, error in enumerate(sorted_errors, 1):
                print(f"   {i}. {Path(error['image']).name}")
                print(f"      True: {error['true_label']}, "
                      f"Predicted: {error['predicted']} "
                      f"(confidence: {error['confidence'] * 100:.2f}%)")

        print("\n" + "=" * 80)

    def save_results(self, output_path: str = "inference_results.csv"):
        """
        Save detailed results to CSV.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'True Label', 'Predicted', 'Confidence', 'Correct'])

            # Write all predictions
            for digit in range(10):
                digit_dir = Path(f"tests/{digit}")
                if digit_dir.exists():
                    for img_path in digit_dir.glob("*.png"):
                        predicted, confidence, _ = self.predict(str(img_path))
                        correct = 'Yes' if predicted == digit else 'No'
                        writer.writerow([str(img_path), digit, predicted,
                                         f"{confidence:.4f}", correct])

        print(f"✓ Results saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch inference on categorized test images')
    parser.add_argument('--model', type=str, default='best_model_mnist_and_water_meter.pt',
                        help='Path to model file (.pt)')
    parser.add_argument('--test-dir', type=str, default='./tests',
                        help='Root directory containing digit folders (0-9)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save detailed results to CSV')
    parser.add_argument('--csv-output', type=str, default='inference_results.csv',
                        help='Output CSV filename')

    args = parser.parse_args()

    # Initialize batch inference
    batch_inf = BatchInference(args.model)

    # Load model
    batch_inf.load_model()

    # Process all images
    batch_inf.process_directory(args.test_dir)

    # Print report
    batch_inf.print_report()

    # Save results if requested
    if args.save_csv:
        print(f"\nSaving detailed results to CSV...")
        batch_inf.save_results(args.csv_output)

    print(f"\n✅ Batch inference complete!\n")


if __name__ == "__main__":
    main()
