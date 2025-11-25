# MNIST Digit Classification - C++ Inference with LibTorch

C++ inference for MNIST digit recognition using LibTorch (PyTorch C++ API).

## Requirements

- CMake >= 3.18
- C++17 compiler (GCC >= 7 / Clang >= 5 / MSVC 2019+)
- LibTorch 2.x
- OpenCV >= 4.0
- Python >= 3.8 (for model export)

## Installation

### LibTorch (Linux)

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-*.zip && sudo mv libtorch /usr/local/
```

### OpenCV

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# macOS
brew install opencv
```

## Quick Start

```bash
# 1. Export TorchScript model
uv run python analyze_model.py

# 2. Build C++ application
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
cmake --build . --config Release

# 3. Run inference
./mnist_inference ../tinycnn_mnist_traced.pt ../tests/0/image.jpg
```

## Batch Inference

Run inference on all test images (organized in `tests/{0-9}/` folders):

```bash
# Python (with statistics & confusion matrix)
uv run python batch_inference.py --model tinycnn_mnist.pt --test-dir ./tests
uv run python batch_inference.py --save-csv --csv-output results.csv  # export CSV

# C++
./build/batch_inference_cpp tinycnn_mnist_traced.pt ./tests
```

## Project Structure

```
├── model.py              # TinyCNN architecture
├── analyze_model.py      # Model analyzer & TorchScript export
├── batch_inference.py    # Python batch inference
├── src/                  # C++ source (inference.cpp, main.cpp, batch_inference.cpp)
├── include/              # C++ headers
├── tests/{0-9}/          # Test images (500 per digit)
└── CMakeLists.txt        # Build configuration
```
