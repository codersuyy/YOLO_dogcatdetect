# 🐶🐱 YOLO Dog & Cat Classification

Dự án phân loại ảnh chó và mèo sử dụng **YOLOv8** (Ultralytics) - mô hình deep learning hiện đại cho bài toán image classification.

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kết quả](#kết-quả)
- [License](#license)

## 🎯 Giới thiệu

Dự án này sử dụng **YOLOv8n-cls** (YOLO version 8 nano classification) để huấn luyện mô hình phân loại ảnh chó và mèo. Mô hình được thiết kế để chạy trên **Google Colab** với GPU miễn phí.

### Các tính năng chính:
- ✅ Huấn luyện mô hình YOLOv8 cho phân loại ảnh
- ✅ Đánh giá mô hình với các metrics: Accuracy, Precision, Recall, F1-Score
- ✅ Hiển thị Confusion Matrix và các ảnh dự đoán sai
- ✅ Dự đoán trên ảnh mới

## 💻 Yêu cầu hệ thống

- Python 3.8+
- Google Colab (khuyến nghị) hoặc máy tính có GPU
- Google Drive để lưu trữ dataset và mô hình

### Thư viện cần thiết:
- `ultralytics` - Framework YOLOv8
- `torch` - PyTorch deep learning
- `scikit-learn` - Đánh giá mô hình
- `matplotlib` - Visualization
- `Pillow` - Xử lý ảnh
- `gdown` - Tải file từ Google Drive

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/yourusername/YOLO_dogcatdetect.git
cd YOLO_dogcatdetect
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy trên Google Colab
1. Upload file `dogcatdectect.ipynb` lên Google Colab
2. Kết nối với GPU Runtime: `Runtime` → `Change runtime type` → `GPU`
3. Chạy từng cell theo thứ tự

## 📁 Cấu trúc dữ liệu

Dataset cần được tổ chức theo cấu trúc sau:

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

## 📖 Hướng dẫn sử dụng

### Bước 1: Kết nối Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 2: Huấn luyện mô hình
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

### Bước 3: Đánh giá mô hình
```python
model = YOLO("runs/classify/train/weights/best.pt")
results = model(test_images)
```

### Bước 4: Dự đoán trên ảnh mới
```python
from PIL import Image

img_path = "path/to/your/image.jpg"
result = model(img_path)
label = result[0].names[result[0].probs.top1]
confidence = result[0].probs.top1conf
print(f"Dự đoán: {label} ({confidence*100:.2f}%)")
```

## 📊 Kết quả

Mô hình được đánh giá với các metrics sau:

| Metric | Giá trị |
|--------|---------|
| Accuracy | ~95% |
| Precision | ~95% |
| Recall | ~95% |
| F1-Score | ~95% |

*Kết quả có thể thay đổi tùy thuộc vào dataset và số epochs huấn luyện.*

## 🔧 Cấu hình huấn luyện

| Parameter | Giá trị | Mô tả |
|-----------|---------|-------|
| Model | YOLOv8n-cls | Nano classification model |
| Epochs | 10 | Số vòng huấn luyện |
| Image Size | 224x224 | Kích thước ảnh đầu vào |
| Batch Size | 16 | Số ảnh mỗi batch |
| Workers | 2 | Số luồng xử lý dữ liệu |

## 📂 Cấu trúc project

```
YOLO_dogcatdetect/
├── dogcatdectect.ipynb    # Notebook chính
├── README.md              # Tài liệu hướng dẫn
├── requirements.txt       # Dependencies
├── LICENSE               # Giấy phép MIT
└── .gitignore            # Ignore files
```

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Dự án này được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👤 Tác giả

- **Vuong** - [GitHub](https://github.com/codersuyy)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Google Colab](https://colab.research.google.com/)
- Dataset: Cat and Dog Classification Dataset

---

⭐ Nếu thấy hữu ích, hãy cho project một star nhé!
