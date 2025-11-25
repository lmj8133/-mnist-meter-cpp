#include "inference.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> <image_path>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path  Path to TorchScript model (.pt file)\n";
    std::cout << "  image_path  Path to input image (PNG, JPG, etc.)\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " tinycnn_mnist_traced.pt tests/sample_images/digit_5_sample_0.png\n";
}

void printResult(const MNISTInference::Result& result) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Prediction Result\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "  Predicted Digit: " << result.predicted_digit << "\n";
    std::cout << "  Confidence:      " << std::fixed << std::setprecision(2)
              << (result.confidence * 100.0f) << "%\n\n";

    std::cout << "  Probability Distribution:\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    // Find top 3 predictions
    std::vector<std::pair<int, float>> indexed_probs;
    for (int i = 0; i < static_cast<int>(result.probabilities.size()); ++i) {
        indexed_probs.push_back({i, result.probabilities[i]});
    }

    std::sort(indexed_probs.begin(), indexed_probs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Print all probabilities with visual bars
    const int bar_width = 30;
    for (int digit = 0; digit < 10; ++digit) {
        float prob = result.probabilities[digit];
        int bar_len = static_cast<int>(prob * bar_width);

        std::cout << "  Digit " << digit << ": "
                  << std::fixed << std::setprecision(4) << (prob * 100.0f) << "% ";

        // Add visual bar
        std::cout << "[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < bar_len) {
                std::cout << "█";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "]";

        // Mark top prediction
        if (digit == result.predicted_digit) {
            std::cout << " ← PREDICTED";
        }

        std::cout << "\n";
    }

    std::cout << "\n  Top 3 Predictions:\n";
    std::cout << "  " << std::string(60, '-') << "\n";
    for (int i = 0; i < 3 && i < static_cast<int>(indexed_probs.size()); ++i) {
        int digit = indexed_probs[i].first;
        float prob = indexed_probs[i].second;
        std::cout << "  " << (i + 1) << ". Digit " << digit << ": "
                  << std::fixed << std::setprecision(2) << (prob * 100.0f) << "%\n";
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       MNIST Digit Classification - C++ Inference          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Parse command line arguments
    if (argc != 3) {
        std::cerr << "✗ Error: Invalid number of arguments\n\n";
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    std::cout << "Configuration:\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  Image: " << image_path << "\n\n";

    // Initialize inference engine
    MNISTInference inference;

    // Load model
    std::cout << std::string(80, '-') << "\n";
    if (!inference.loadModel(model_path)) {
        std::cerr << "\n✗ Failed to load model\n";
        return 1;
    }

    // Run inference
    std::cout << std::string(80, '-') << "\n";
    std::cout << "Running inference...\n";

    MNISTInference::Result result;
    if (!inference.predict(image_path, result)) {
        std::cerr << "\n✗ Inference failed\n";
        return 1;
    }

    std::cout << "✓ Inference completed successfully\n";

    // Print results
    printResult(result);

    std::cout << "\n✅ Program completed successfully\n\n";

    return 0;
}
