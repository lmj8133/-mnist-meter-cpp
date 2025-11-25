#ifndef MNIST_INFERENCE_H
#define MNIST_INFERENCE_H

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

/**
 * MNIST digit classification inference using LibTorch.
 *
 * This class provides an interface for loading a TorchScript model
 * and performing inference on MNIST-style images (28x28 grayscale).
 */
class MNISTInference {
public:
    /**
     * Result structure containing prediction and confidence scores.
     */
    struct Result {
        int predicted_digit;           // Predicted digit (0-9)
        float confidence;              // Confidence score [0, 1]
        std::vector<float> probabilities;  // Probability for each digit (0-9)
    };

    /**
     * Constructor.
     */
    MNISTInference();

    /**
     * Destructor.
     */
    ~MNISTInference();

    /**
     * Load TorchScript model from file.
     *
     * @param model_path Path to the .pt TorchScript model file
     * @return true if model loaded successfully, false otherwise
     */
    bool loadModel(const std::string& model_path);

    /**
     * Perform inference on an image file.
     *
     * @param image_path Path to input image file
     * @param result Output result structure
     * @return true if inference succeeded, false otherwise
     */
    bool predict(const std::string& image_path, Result& result);

    /**
     * Perform inference on an OpenCV Mat image.
     *
     * @param image Input image (grayscale or color, will be converted)
     * @param result Output result structure
     * @return true if inference succeeded, false otherwise
     */
    bool predict(const cv::Mat& image, Result& result);

    /**
     * Check if model is loaded.
     *
     * @return true if model is ready for inference
     */
    bool isModelLoaded() const;

private:
    /**
     * Preprocess image to model input format.
     *
     * Converts image to 28x28 grayscale, normalizes pixel values,
     * and creates tensor with shape [1, 1, 28, 28].
     *
     * @param image Input image
     * @return Preprocessed tensor ready for model input
     */
    torch::Tensor preprocessImage(const cv::Mat& image);

    /**
     * Apply softmax and extract prediction result.
     *
     * @param output Raw model output tensor
     * @param result Output result structure
     */
    void postprocess(const torch::Tensor& output, Result& result);

    torch::jit::script::Module module_;  // TorchScript module
    bool model_loaded_;                   // Model loading status

    // MNIST normalization parameters
    // Updated to match batch_inference.py settings
    static constexpr float MNIST_MEAN = 0.5f;
    static constexpr float MNIST_STD = 0.5f;
};

#endif // MNIST_INFERENCE_H
