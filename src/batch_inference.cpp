#include "inference.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

// ANSI color codes for terminal output
#define RESET   "\033[0m"
#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define YELLOW  "\033[1;33m"
#define BLUE    "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN    "\033[1;36m"

struct Statistics {
    int total = 0;
    int correct = 0;
    std::vector<std::vector<int>> confusion_matrix;
    std::map<int, int> per_class_total;
    std::map<int, int> per_class_correct;
    std::vector<double> inference_times;

    Statistics() {
        // Initialize 10x10 confusion matrix
        confusion_matrix.resize(10, std::vector<int>(10, 0));
        for (int i = 0; i < 10; ++i) {
            per_class_total[i] = 0;
            per_class_correct[i] = 0;
        }
    }

    void addPrediction(int true_label, int predicted, double inference_time) {
        total++;
        per_class_total[true_label]++;
        inference_times.push_back(inference_time);

        confusion_matrix[true_label][predicted]++;

        if (true_label == predicted) {
            correct++;
            per_class_correct[true_label]++;
        }
    }

    double getOverallAccuracy() const {
        return total > 0 ? (100.0 * correct / total) : 0.0;
    }

    double getClassAccuracy(int digit) const {
        int class_total = per_class_total.at(digit);
        int class_correct = per_class_correct.at(digit);
        return class_total > 0 ? (100.0 * class_correct / class_total) : 0.0;
    }

    double getAverageInferenceTime() const {
        if (inference_times.empty()) return 0.0;
        double sum = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
        return sum / inference_times.size();
    }
};

void printProgressBar(int current, int total, int true_label) {
    const int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r  Digit " << true_label << ": [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "█";
        else std::cout << "░";
    }
    std::cout << "] " << current << "/" << total << " images" << std::flush;
}

void printHeader() {
    std::cout << "\n╔" << std::string(78, '=') << "╗\n";
    std::cout << "║" << std::string(20, ' ') << "C++ BATCH INFERENCE REPORT"
              << std::string(32, ' ') << "║\n";
    std::cout << "╚" << std::string(78, '=') << "╝\n\n";
}

void printOverallStats(const Statistics& stats) {
    std::cout << "📊 Overall Statistics:\n";
    std::cout << "   Total images:     " << stats.total << "\n";
    std::cout << "   Correct:          " << stats.correct << "\n";
    std::cout << "   Incorrect:        " << (stats.total - stats.correct) << "\n";
    std::cout << "   Accuracy:         " << std::fixed << std::setprecision(2)
              << stats.getOverallAccuracy() << "%\n";
}

void printPerformance(const Statistics& stats) {
    double avg_time = stats.getAverageInferenceTime() * 1000.0; // Convert to ms
    std::cout << "\n⚡ Performance:\n";
    std::cout << "   Avg inference:    " << std::fixed << std::setprecision(2)
              << avg_time << " ms/image\n";
    std::cout << "   Throughput:       " << std::fixed << std::setprecision(1)
              << (1000.0 / avg_time) << " images/sec\n";
}

void printPerClassAccuracy(const Statistics& stats) {
    std::cout << "\n📈 Per-Class Accuracy:\n";
    std::cout << "   " << std::string(60, '─') << "\n";
    std::cout << "   " << std::left << std::setw(8) << "Digit"
              << std::setw(20) << "Correct/Total"
              << std::setw(15) << "Accuracy"
              << "Bar\n";
    std::cout << "   " << std::string(60, '─') << "\n";

    for (int digit = 0; digit < 10; ++digit) {
        int class_total = stats.per_class_total.at(digit);
        int class_correct = stats.per_class_correct.at(digit);

        if (class_total > 0) {
            double acc = stats.getClassAccuracy(digit);
            int bar_len = static_cast<int>(acc / 2); // Scale to 50 chars max

            std::cout << "   " << std::left << std::setw(8) << digit
                      << std::right << std::setw(5) << class_correct << "/"
                      << std::left << std::setw(5) << class_total
                      << std::string(8, ' ')
                      << std::right << std::setw(6) << std::fixed
                      << std::setprecision(2) << acc << "%"
                      << std::string(8, ' ') << "[";

            for (int i = 0; i < 50; ++i) {
                if (i < bar_len) std::cout << "█";
                else std::cout << "░";
            }
            std::cout << "]\n";
        }
    }
}

void printConfusionMatrix(const Statistics& stats) {
    std::cout << "\n🔄 Confusion Matrix:\n";
    std::cout << "   (Rows: True labels, Columns: Predicted labels)\n";
    std::cout << "   " << std::string(65, '─') << "\n";

    // Header
    std::cout << "      ";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(5) << i << " ";
    }
    std::cout << "\n   " << std::string(65, '─') << "\n";

    // Matrix rows
    for (int i = 0; i < 10; ++i) {
        std::cout << "   " << std::setw(2) << i << " ";
        for (int j = 0; j < 10; ++j) {
            int count = stats.confusion_matrix[i][j];

            if (i == j && count > 0) {
                // Diagonal (correct predictions) - green
                std::cout << GREEN << std::setw(5) << count << RESET << " ";
            } else if (count > 0) {
                // Errors - red
                std::cout << RED << std::setw(5) << count << RESET << " ";
            } else {
                std::cout << std::setw(5) << count << " ";
            }
        }
        std::cout << "\n";
    }
}

void printErrorAnalysis(const Statistics& stats) {
    std::cout << "\n❌ Error Analysis:\n";
    std::cout << "   " << std::string(60, '─') << "\n";

    int total_errors = stats.total - stats.correct;
    if (total_errors == 0) {
        std::cout << "   " << GREEN << "No errors! Perfect accuracy!" << RESET << "\n";
        return;
    }

    std::cout << "   Total errors: " << total_errors << "\n\n";

    for (int digit = 0; digit < 10; ++digit) {
        int class_total = stats.per_class_total.at(digit);
        int class_correct = stats.per_class_correct.at(digit);
        int errors = class_total - class_correct;

        if (errors > 0) {
            std::cout << "   Digit " << digit << ": " << errors << " error(s)\n";

            // Find top misclassifications
            std::vector<std::pair<int, int>> misclass; // (predicted_digit, count)
            for (int j = 0; j < 10; ++j) {
                if (j != digit && stats.confusion_matrix[digit][j] > 0) {
                    misclass.push_back({j, stats.confusion_matrix[digit][j]});
                }
            }

            // Sort by count (descending)
            std::sort(misclass.begin(), misclass.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            // Show top 3 misclassifications
            int show_count = std::min(3, static_cast<int>(misclass.size()));
            for (int i = 0; i < show_count; ++i) {
                std::cout << "      → Often misclassified as " << misclass[i].first
                          << ": " << misclass[i].second << " times\n";
            }
        }
    }
}

void processBatchInference(const std::string& model_path, const std::string& test_dir) {
    std::cout << "Loading model from: " << model_path << "\n";
    std::cout << std::string(80, '=') << "\n";

    // Initialize inference engine
    MNISTInference inference;
    if (!inference.loadModel(model_path)) {
        std::cerr << RED << "✗ Failed to load model" << RESET << "\n";
        return;
    }

    std::cout << "\nProcessing test images from: " << test_dir << "\n";
    std::cout << std::string(80, '=') << "\n\n";

    Statistics stats;

    // Process each digit folder (0-9)
    for (int digit = 0; digit < 10; ++digit) {
        fs::path digit_dir = fs::path(test_dir) / std::to_string(digit);

        if (!fs::exists(digit_dir) || !fs::is_directory(digit_dir)) {
            std::cout << YELLOW << "⚠️  Warning: Directory " << digit_dir
                      << " not found, skipping..." << RESET << "\n";
            continue;
        }

        // Collect all image files
        std::vector<fs::path> image_files;
        for (const auto& entry : fs::directory_iterator(digit_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                    image_files.push_back(entry.path());
                }
            }
        }

        if (image_files.empty()) {
            std::cout << YELLOW << "⚠️  Warning: No images found in " << digit_dir
                      << ", skipping..." << RESET << "\n";
            continue;
        }

        // Process images
        for (size_t i = 0; i < image_files.size(); ++i) {
            try {
                auto start_time = std::chrono::high_resolution_clock::now();

                MNISTInference::Result result;
                if (inference.predict(image_files[i].string(), result)) {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_time - start_time;

                    stats.addPrediction(digit, result.predicted_digit, elapsed.count());
                } else {
                    std::cerr << "\n" << RED << "⚠️  Error processing: "
                              << image_files[i].filename() << RESET << "\n";
                }

                // Update progress bar every 50 images
                if ((i + 1) % 50 == 0 || (i + 1) == image_files.size()) {
                    printProgressBar(i + 1, image_files.size(), digit);
                }

            } catch (const std::exception& e) {
                std::cerr << "\n" << RED << "⚠️  Exception: " << e.what() << RESET << "\n";
            }
        }

        // Print summary for this digit
        int class_total = stats.per_class_total[digit];
        int class_correct = stats.per_class_correct[digit];
        double class_acc = stats.getClassAccuracy(digit);

        std::cout << "\r  Digit " << digit << ": " << class_correct << "/"
                  << class_total << " (" << std::fixed << std::setprecision(2)
                  << class_acc << "%) " << GREEN << "✓" << RESET << "\n";
    }

    std::cout << "\n" << std::string(80, '=') << "\n";

    // Print detailed report
    printHeader();
    printOverallStats(stats);
    printPerformance(stats);
    printPerClassAccuracy(stats);
    printConfusionMatrix(stats);
    printErrorAnalysis(stats);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << GREEN << "\n✅ Batch inference complete!\n" << RESET << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "\n╔" << std::string(60, '=') << "╗\n";
    std::cout << "║" << std::string(10, ' ') << "MNIST Batch Inference - C++" << std::string(22, ' ') << "║\n";
    std::cout << "╚" << std::string(60, '=') << "╝\n\n";

    // Parse command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <test_dir>\n\n";
        std::cerr << "Arguments:\n";
        std::cerr << "  model_path  Path to TorchScript model (.pt file)\n";
        std::cerr << "  test_dir    Root directory containing digit folders (0-9)\n\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " tinycnn_mnist_traced.pt ./tests\n\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_dir = argv[2];

    std::cout << "Configuration:\n";
    std::cout << "  Model:     " << model_path << "\n";
    std::cout << "  Test dir:  " << test_dir << "\n\n";

    try {
        processBatchInference(model_path, test_dir);
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ Fatal error: " << e.what() << RESET << "\n";
        return 1;
    }

    return 0;
}
