#include "inference.h"
#include <iostream>
#include <algorithm>
#include <numeric>

MNISTInference::MNISTInference() : model_loaded_(false) {}

MNISTInference::~MNISTInference() {}

bool MNISTInference::loadModel(const std::string& model_path) {
    try {
        std::cout << "Loading model from: " << model_path << std::endl;

        // Deserialize the ScriptModule from file
        module_ = torch::jit::load(model_path);
        module_.eval();  // Set to evaluation mode

        model_loaded_ = true;
        std::cout << "✓ Model loaded successfully" << std::endl;

        return true;
    } catch (const c10::Error& e) {
        std::cerr << "✗ Error loading model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

bool MNISTInference::predict(const std::string& image_path, Result& result) {
    if (!model_loaded_) {
        std::cerr << "✗ Model not loaded. Call loadModel() first." << std::endl;
        return false;
    }

    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "✗ Failed to load image: " << image_path << std::endl;
        return false;
    }

    return predict(image, result);
}

bool MNISTInference::predict(const cv::Mat& image, Result& result) {
    if (!model_loaded_) {
        std::cerr << "✗ Model not loaded. Call loadModel() first." << std::endl;
        return false;
    }

    try {
        // Preprocess image
        torch::Tensor input_tensor = preprocessImage(image);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        torch::Tensor output = module_.forward(inputs).toTensor();

        // Postprocess output
        postprocess(output, result);

        return true;
    } catch (const c10::Error& e) {
        std::cerr << "✗ Inference error: " << e.what() << std::endl;
        return false;
    }
}

bool MNISTInference::isModelLoaded() const {
    return model_loaded_;
}

torch::Tensor MNISTInference::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;

    // Convert to grayscale if needed
    if (image.channels() == 3) {
        cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = image.clone();
    }

    // Resize to 28x28 if needed
    if (processed.rows != 28 || processed.cols != 28) {
        cv::resize(processed, processed, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);
    }

    // Convert to float and normalize to [0, 1]
    processed.convertTo(processed, CV_32FC1, 1.0 / 255.0);

    // Apply MNIST normalization: (x - mean) / std
    processed = (processed - MNIST_MEAN) / MNIST_STD;

    // Convert to tensor with shape [1, 1, 28, 28]
    // OpenCV Mat format: [H, W] -> Torch tensor format: [B, C, H, W]
    torch::Tensor tensor = torch::from_blob(
        processed.data,
        {1, 1, 28, 28},
        torch::kFloat32
    ).clone();  // Clone to ensure memory is owned by tensor

    return tensor;
}

void MNISTInference::postprocess(const torch::Tensor& output, Result& result) {
    // Apply softmax to get probabilities
    torch::Tensor probabilities = torch::softmax(output, /*dim=*/1);

    // Get predicted class and confidence
    auto max_result = probabilities.max(1);
    torch::Tensor max_prob = std::get<0>(max_result);
    torch::Tensor max_index = std::get<1>(max_result);

    result.predicted_digit = max_index.item<int>();
    result.confidence = max_prob.item<float>();

    // Extract all probabilities
    result.probabilities.clear();
    result.probabilities.reserve(10);

    auto prob_accessor = probabilities.accessor<float, 2>();
    for (int i = 0; i < 10; ++i) {
        result.probabilities.push_back(prob_accessor[0][i]);
    }
}
