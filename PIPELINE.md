# PyTorch 模型到 C++ 推論完整 Pipeline

> 從拿到 `.pt` 模型檔案開始，到完成 C++ 推論和批次準確率分析的完整流程指南

---

## 📑 目錄

1. [概述](#概述)
2. [前置需求](#前置需求)
3. [完整流程](#完整流程)
4. [快速參考](#快速參考)
5. [檔案結構](#檔案結構)
6. [故障排除](#故障排除)

---

## 概述

### 適用場景

- ✅ 拿到 PyTorch 訓練好的 `.pt` 模型檔案
- ✅ 需要在 C++ 環境中部署推論
- ✅ 需要批次處理並統計準確率
- ✅ MNIST 或類似的圖像分類任務

### 整體流程

```
原始模型 (.pt)
    ↓
[分析與匯出] → TorchScript 模型 (.pt)
    ↓
[環境建置] → C++ 可執行檔
    ↓
[執行推論] → 結果與統計報告
```

---

## 前置需求

### 必要軟體

| 軟體 | 最低版本 | 檢查命令 |
|------|---------|---------|
| Python | 3.8+ | `python --version` |
| CMake | 3.18+ | `cmake --version` |
| C++ 編譯器 | GCC 7+/Clang 5+ | `gcc --version` |
| uv | - | `uv --version` |

### Python 依賴

```bash
# 使用 uv 安裝
uv pip install torch torchvision pillow numpy opencv-python
```

---

## 完整流程

### 階段一：模型分析與驗證

#### Step 1: 檢查原始模型

```bash
# 確認模型檔案存在
ls -lh tinycnn_mnist.pt

# 預期輸出：約 200+ KB 的 .pt 檔案
```

#### Step 2: 分析模型架構並匯出 TorchScript

```bash
uv run python analyze_model.py
```

**這個步驟會：**
- 自動偵測模型格式（完整模型 or state_dict）
- 載入權重到 TinyCNN 架構
- 顯示模型結構和參數統計
- **匯出 TorchScript 格式**：`tinycnn_mnist_traced.pt` ⭐

**預期輸出：**
```
Loading model from: tinycnn_mnist.pt
================================================================================
✓ Detected state_dict format
✓ Weights loaded successfully

📋 Model Architecture:
TinyCNN(
  (features): Sequential(...)
  (classifier): Sequential(...)
)

📊 Model Statistics:
  Total parameters:     52,138

💾 Exporting to TorchScript:
  ✓ Saved TorchScript model to: tinycnn_mnist_traced.pt
```

**關鍵產出：**
- ✅ `tinycnn_mnist_traced.pt` - C++ 推論必需的檔案

#### Step 3: 驗證 Python 推論（可選）

```bash
# 完整驗證（會下載 MNIST 測試集）
uv run python test_inference.py

# 或快速生成測試圖像
uv run python generate_test_images.py
```

---

### 階段二：C++ 環境建置

#### Step 4: 安裝 LibTorch

**下載並安裝：**
```bash
cd /tmp

# 下載 LibTorch (CPU 版本)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip

# 解壓縮
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# 移動到系統目錄
sudo mv libtorch /usr/local/
```

**設置環境變數：**
```bash
# 加到 ~/.bashrc 或 ~/.zshrc
export CMAKE_PREFIX_PATH=/usr/local/libtorch:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH

# 立即生效
source ~/.bashrc
```

#### Step 5: 安裝 OpenCV

**Ubuntu/Debian：**
```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev
```

**驗證安裝：**
```bash
pkg-config --modversion opencv4
# 預期輸出：4.x.x
```

#### Step 6: 建置 C++ 專案

```bash
# 建立 build 目錄
mkdir build
cd build

# 配置 CMake
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..

# 編譯（Release 模式）
cmake --build . --config Release
```

**預期輸出：**
```
-- Found Torch: /usr/local/libtorch/lib/libtorch.so
-- Found LibTorch: /usr/local/libtorch
-- Found OpenCV: 4.x.x
-- Build type: Release
...
[100%] Built target mnist_inference
```

**驗證可執行檔：**
```bash
# 回到專案根目錄
cd ..

# 檢查可執行檔
ls -lh build/mnist_inference
```

---

### 階段三：推論執行

#### Step 7: C++ 單張圖像推論

```bash
# 基本用法
./build/mnist_inference tinycnn_mnist_traced.pt tests/0/image.png

# 注意：必須使用 TorchScript 模型 (*_traced.pt)
```

**輸出範例：**
```
╔════════════════════════════════════════════════╗
║   MNIST Digit Classification - C++ Inference   ║
╚════════════════════════════════════════════════╝

Configuration:
  Model: tinycnn_mnist_traced.pt
  Image: tests/0/image.png

Loading model from: tinycnn_mnist_traced.pt
✓ Model loaded successfully
Running inference...
✓ Inference completed successfully

Predicted Digit: 0
Confidence:      98.45%

Probability Distribution:
  Digit 0: 98.4567% [█████████████████████████████] ← PREDICTED
  Digit 1:  0.1234% [                              ]
  ...
```

#### Step 8: Python 批次推論與準確率統計

```bash
# 基本批次推論
uv run python batch_inference.py

# 指定測試目錄
uv run python batch_inference.py --test-dir ./tests

# 儲存詳細結果到 CSV
uv run python batch_inference.py --save-csv --csv-output results.csv
```

**輸出報告：**
```
📊 Overall Statistics:
   Total images:     5,000
   Correct:          4,950
   Accuracy:         99.00%

📈 Per-Class Accuracy:
   Digit 0: 495/500 (99.00%)
   Digit 1: 498/500 (99.60%)
   ...

🔄 Confusion Matrix:
   (顯示各類別間的混淆情況)

❌ Error Analysis:
   (詳細的錯誤案例分析)
```

---

## 快速參考

### 命令速查表

| 步驟 | 命令 | 用途 |
|:----:|------|------|
| 1 | `ls -lh tinycnn_mnist.pt` | 檢查原始模型 |
| 2 | `uv run python analyze_model.py` | 分析並匯出 TorchScript |
| 3 | `uv run python test_inference.py` | Python 推論驗證 |
| 4 | `sudo mv libtorch /usr/local/` | 安裝 LibTorch |
| 5 | `sudo apt-get install libopencv-dev` | 安裝 OpenCV |
| 6 | `cmake .. && cmake --build .` | 建置 C++ 專案 |
| 7 | `./build/mnist_inference model.pt img.png` | C++ 單張推論 |
| 8 | `uv run python batch_inference.py` | 批次推論與統計 |

### 關鍵檔案說明

| 檔案 | 類型 | 用途 | 何時使用 |
|------|------|------|---------|
| `tinycnn_mnist.pt` | PyTorch State Dict | 原始訓練權重 | Python 載入時需要 model.py |
| `tinycnn_mnist_traced.pt` | TorchScript | C++ 推論模型 | **C++ 推論必需** ⭐ |
| `model.py` | Python 腳本 | TinyCNN 架構定義 | 載入 state_dict 時需要 |
| `analyze_model.py` | Python 腳本 | 模型分析工具 | 匯出 TorchScript |
| `batch_inference.py` | Python 腳本 | 批次推論工具 | 統計準確率 |
| `build/mnist_inference` | 可執行檔 | C++ 推論程式 | 生產環境部署 |

### 模型檔案對照

```
Python 推論:
  tinycnn_mnist.pt (state_dict) + model.py → 推論

C++ 推論:
  tinycnn_mnist_traced.pt (TorchScript) → LibTorch → 推論

關鍵轉換:
  analyze_model.py: .pt (state_dict) → .pt (TorchScript)
```

---

## 檔案結構

執行完整 pipeline 後的專案結構：

```
water_meter/
├── 📄 模型檔案
│   ├── tinycnn_mnist.pt              # 原始模型（state_dict）
│   └── tinycnn_mnist_traced.pt       # TorchScript 模型（C++ 用）✨
│
├── 🐍 Python 工具
│   ├── model.py                      # TinyCNN 架構定義
│   ├── analyze_model.py              # 模型分析與匯出工具
│   ├── test_inference.py             # Python 推論驗證
│   ├── batch_inference.py            # 批次推論與統計 ⭐
│   └── generate_test_images.py       # 測試圖像生成器
│
├── 💻 C++ 程式碼
│   ├── include/
│   │   └── inference.h               # C++ API 標頭檔
│   ├── src/
│   │   ├── inference.cpp             # 推論實作
│   │   └── main.cpp                  # CLI 主程式
│   └── CMakeLists.txt                # 建置配置
│
├── 🏗️ 建置產物
│   └── build/
│       └── mnist_inference           # C++ 可執行檔 ✨
│
├── 🧪 測試資料
│   └── tests/
│       ├── 0/  (500 images)
│       ├── 1/  (500 images)
│       ├── ...
│       └── 9/  (500 images)
│
├── 📚 文檔
│   ├── README.md                     # 專案說明
│   ├── PIPELINE.md                   # 本文檔
│   └── .gitignore
│
└── 📊 結果檔案（可選）
    └── inference_results.csv         # 批次推論詳細結果
```

---

## 故障排除

### CMake 找不到 LibTorch

**錯誤訊息：**
```
CMake Error: Could not find a package configuration file provided by "Torch"
```

**解決方案：**
```bash
# 明確指定 LibTorch 路徑
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
```

---

### CMake 找不到 OpenCV

**錯誤訊息：**
```
CMake Error: Could not find a package configuration file provided by "OpenCV"
```

**解決方案：**
```bash
# 安裝 OpenCV 開發庫
sudo apt-get update
sudo apt-get install -y libopencv-dev

# 驗證安裝
pkg-config --modversion opencv4

# 如果 pkg-config 找不到，手動指定路徑
cmake -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 ..
```

---

### 執行時找不到共享庫

**錯誤訊息：**
```
error while loading shared libraries: libtorch.so: cannot open shared object file
```

**解決方案：**
```bash
# 臨時設置
export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH

# 永久設置（加到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### Python 載入模型失敗（state_dict）

**錯誤訊息：**
```
RuntimeError: Error(s) in loading state_dict...
```

**原因：**
模型是 state_dict 格式，但缺少架構定義。

**解決方案：**
確保有 `model.py` 且已正確 import：
```python
from model import TinyCNN

model = TinyCNN()
model.load_state_dict(torch.load('tinycnn_mnist.pt'))
```

---

### C++ 載入模型失敗

**錯誤訊息：**
```
Error loading model: [model loading error]
```

**常見原因：**
1. 使用了 state_dict 而不是 TorchScript 模型
2. 模型路徑錯誤

**解決方案：**
```bash
# 確保使用 TorchScript 模型
./build/mnist_inference tinycnn_mnist_traced.pt image.png
#                        ^^^^^^^^^^^^^^^^^^^^^^^^
#                        必須是 *_traced.pt 檔案！

# 檢查檔案是否存在
ls -lh tinycnn_mnist_traced.pt
```

---

### 批次推論準確率異常低

**可能原因：**
1. 測試圖像與訓練資料分佈不同（域偏移）
2. 圖像前處理不一致
3. 圖像格式或品質問題

**診斷步驟：**
```bash
# 1. 查看幾張測試圖像
ls tests/0/ | head -5

# 2. 檢查圖像內容和品質
# （使用圖像查看器確認）

# 3. 與標準 MNIST 比較
uv run python test_inference.py  # 會下載真實 MNIST 測試
```

**調整建議：**
- 檢查測試圖像的來源和品質
- 可能需要調整 `batch_inference.py` 中的圖像前處理
- 考慮使用數據增強或微調模型

---

## 工作流程圖

```
┌─────────────────────────────────────────┐
│  起點: tinycnn_mnist.pt (state_dict)    │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Step 1-2: 模型分析與匯出                 │
│  $ uv run python analyze_model.py        │
│  → 偵測 state_dict 格式                  │
│  → 載入 TinyCNN 架構                     │
│  → 匯出 tinycnn_mnist_traced.pt ✨       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Step 3: (可選) Python 推論驗證           │
│  $ uv run python test_inference.py       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Step 4-5: 安裝 C++ 依賴                 │
│  • LibTorch → /usr/local/libtorch        │
│  • OpenCV   → apt-get install            │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Step 6: CMake 建置                      │
│  $ mkdir build && cd build               │
│  $ cmake -DCMAKE_PREFIX_PATH=... ..      │
│  $ cmake --build . --config Release      │
│  → 生成 mnist_inference 可執行檔 ✨      │
└──────────────┬───────────────────────────┘
               ↓
       ┌───────┴───────┐
       ↓               ↓
┌─────────────┐  ┌────────────────┐
│ Step 7:     │  │ Step 8:        │
│ C++ 單張    │  │ Python 批次    │
│ 推論        │  │ 推論與統計      │
└──────┬──────┘  └───────┬────────┘
       ↓                 ↓
       └────────┬────────┘
                ↓
    ┌───────────────────────┐
    │  完成: 推論結果        │
    │  • 預測類別           │
    │  • 信心分數           │
    │  • 準確率統計         │
    │  • 混淆矩陣           │
    └───────────────────────┘
```

---

## 最佳實踐

### 開發階段

1. **先用 Python 驗證**
   ```bash
   uv run python analyze_model.py    # 確認模型可載入
   uv run python test_inference.py   # 驗證推論正確性
   ```

2. **逐步建置 C++**
   ```bash
   # 先確認依賴都安裝正確
   pkg-config --modversion opencv4
   ls /usr/local/libtorch/lib/libtorch.so

   # 再進行建置
   cmake .. && cmake --build .
   ```

3. **小規模測試後再批次**
   ```bash
   # 先測試單張
   ./build/mnist_inference model.pt tests/0/test.png

   # 確認無誤後再批次
   uv run python batch_inference.py
   ```

### 生產部署

1. **只需要這些檔案**
   - `tinycnn_mnist_traced.pt` (模型)
   - `mnist_inference` (可執行檔)
   - LibTorch 動態庫
   - OpenCV 動態庫

2. **環境變數設置**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH
   ```

3. **效能優化**
   - 使用 Release 建置模式
   - 考慮 GPU 版本 LibTorch（如果需要）
   - 批次處理多張圖像

---

## 總結

完整 pipeline 三階段：

1. **分析 → 匯出** (`analyze_model.py`)
   - 輸入：`tinycnn_mnist.pt` (state_dict)
   - 輸出：`tinycnn_mnist_traced.pt` (TorchScript)

2. **建置 → 編譯** (CMake)
   - 依賴：LibTorch + OpenCV
   - 輸出：`mnist_inference` (可執行檔)

3. **推論 → 統計** (C++/Python)
   - C++：單張高效推論
   - Python：批次統計分析

**核心產出檔案：**
- ✨ `tinycnn_mnist_traced.pt` - C++ 推論必需
- ✨ `mnist_inference` - 可執行推論程式
- ✨ `inference_results.csv` - 詳細結果分析

---

## 相關文檔

- [README.md](README.md) - 專案完整說明
- [LibTorch 官方文檔](https://pytorch.org/cppdocs/)
- [OpenCV 官方文檔](https://docs.opencv.org/)

---

**最後更新**: 2025-01-21
**維護者**: Claude Code
