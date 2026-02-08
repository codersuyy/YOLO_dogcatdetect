# 🐶🐱 YOLO Dog & Cat Classification

A dog and cat image classification project using **YOLOv8** (Ultralytics) - a state-of-the-art deep learning model for image classification tasks.

## 📋 Table of Contents

- [Introduction](#introduction)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage Guide](#usage-guide)
- [Results](#results)
- [License](#license)

## 🎯 Introduction

This project uses **YOLOv8n-cls** (YOLO version 8 nano classification) to train an image classification model for dogs and cats. The model is designed to run on **Google Colab** with free GPU access.

### Key Features:
- ✅ Train YOLOv8 model for image classification
- ✅ Model evaluation with metrics: Accuracy, Precision, Recall, F1-Score
- ✅ Display Confusion Matrix and misclassified images
- ✅ Predict on new images

## 💻 System Requirements

- Python 3.8+
- Google Colab (recommended) or a computer with GPU
- Google Drive for storing dataset and model

### Required Libraries:
- `ultralytics` - YOLOv8 Framework
- `torch` - PyTorch deep learning
- `scikit-learn` - Model evaluation
- `matplotlib` - Visualization
- `Pillow` - Image processing
- `gdown` - Download files from Google Drive

## 🚀 Installation

### 1. Clone repository
```bash
git clone https://github.com/yourusername/YOLO_dogcatdetect.git
cd YOLO_dogcatdetect
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Google Colab
1. Upload `dogcatdectect.ipynb` to Google Colab
2. Connect to GPU Runtime: `Runtime` → `Change runtime type` → `GPU`
3. Run each cell in order

## 📁 Dataset Structure

The dataset should be organized as follows:

```
CatandDogDataset/
├── train/
│   ├── cats/
│   │   ├── cat001.jpg
│   │   ├── cat002.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog001.jpg
│       ├── dog002.jpg
│       └── ...
├── val/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

## 📖 Usage Guide

### Step 1: Connect Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Train the Model
```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
model.train(
    data="/path/to/CatandDogDataset",
    epochs=10,
    imgsz=224,
    batch=16
)
```

### Step 3: Evaluate the Model
```python
model = YOLO("runs/classify/train/weights/best.pt")
results = model(test_images)
```

### Step 4: Predict on New Images
```python
from PIL import Image

img_path = "path/to/your/image.jpg"
result = model(img_path)
label = result[0].names[result[0].probs.top1]
confidence = result[0].probs.top1conf
print(f"Prediction: {label} ({confidence*100:.2f}%)")
```

## 📊 Results

The model is evaluated with the following metrics:

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~95% |
| Recall | ~95% |
| F1-Score | ~95% |

*Results may vary depending on the dataset and number of training epochs.*

## 🔧 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | YOLOv8n-cls | Nano classification model |
| Epochs | 10 | Number of training epochs |
| Image Size | 224x224 | Input image size |
| Batch Size | 16 | Images per batch |
| Workers | 2 | Number of data loading workers |

## 📂 Project Structure

```
YOLO_dogcatdetect/
├── dogcatdectect.ipynb    # Main notebook
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── LICENSE               # MIT License
└── .gitignore            # Ignore files
```

## 🤝 Contributing

All contributions are welcome! Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📄 License

This project is distributed under the MIT License. See [LICENSE](LICENSE) file for more details.

## 👤 Author

- **Vuong** - [GitHub](https://github.com/codersuyy)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Google Colab](https://colab.research.google.com/)
- Dataset: Cat and Dog Classification Dataset

---

⭐ If you find this helpful, please give this project a star!
