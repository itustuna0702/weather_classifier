# 🌦️ Weather Classifier (PyTorch)

Dự án deep learning phân loại ảnh thời tiết sử dụng PyTorch. Hỗ trợ mô hình ResNet18 (pretrained) hoặc CNN tùy chỉnh, áp dụng cho 11 loại thời tiết như mưa, tuyết, sương mù, cầu vồng, v.v.

---

## 📁 Cấu trúc dự án

```
weather_classifier/
├── model/
│   ├── __init__.py
│   ├── models.py           # ResNet18 hoặc CNN tự định nghĩa
│   ├── losses.py           # CrossEntropyLoss
│   └── modeling_output.py  # Chuyển logits → nhãn
├── weather_dataset/        # 11 thư mục tương ứng 11 loại thời tiết
├── config.yaml             # File cấu hình (batch_size, epoch, ...)
├── data_utils.py           # Load dữ liệu, chia train/val/test
├── trainer.py              # Vòng huấn luyện chính
├── main.py                 # Điểm bắt đầu huấn luyện
├── utils.py                # Metrics và vẽ confusion matrix
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Tài liệu mô tả dự án
```

---

## 🔧 Tính năng nổi bật

- ✅ Load ảnh với `torchvision.datasets.ImageFolder`
- ✅ Chia dữ liệu theo class: **Train 70%**, **Val 15%**, **Test 15%**
- ✅ Resize ảnh về **224x224**, augment đơn giản
- ✅ Hỗ trợ chọn giữa **ResNet18 pretrained** hoặc **CNN tự thiết kế**
- ✅ Tính **Accuracy**, **Precision**, **Recall**, **F1-score** mỗi epoch
- ✅ Vẽ và lưu **confusion matrix** dưới dạng ảnh
- ✅ Lưu mô hình mỗi epoch dưới dạng checkpoint
- ✅ Tách riêng config qua file `config.yaml` dễ điều chỉnh
- ✅ Mã nguồn chia module rõ ràng, dễ mở rộng

---

## 📦 Cài đặt

### Cách 1: Dùng pip

```bash
pip install -r requirements.txt
```

### Cách 2: Dùng conda (khuyên dùng)

```bash
conda create -n weather-env python=3.10 -y
conda activate weather-env
pip install -r requirements.txt
```

---

## ⚙️ Cấu hình (`config.yaml`)

```yaml
batch_size: 32
learning_rate: 0.001
epochs: 10
num_classes: 11
train_ratio: 0.7
val_ratio: 0.15
img_size: 224
model_name: "resnet18"
checkpoint_dir: "checkpoints"
```

---

## 🚀 Huấn luyện mô hình

Chạy lệnh sau từ thư mục gốc:

```bash
python main.py
```

- Mô hình sẽ được huấn luyện theo cấu hình trong `config.yaml`
- Mỗi epoch sẽ in kết quả và lưu checkpoint + confusion matrix

---

## 📊 Đánh giá và trực quan hóa

- Sử dụng các hàm trong `utils.py` để tính:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Vẽ biểu đồ confusion matrix (sau mỗi epoch) và lưu vào thư mục `checkpoints/`.

---

## 🧩 Mở rộng dễ dàng

- Thêm mô hình khác → `models.py`
- Đổi loss function → `losses.py`
- Thêm tính năng: EarlyStopping, TensorBoard, GradCAM, Scheduler...
- Viết API dự đoán → thêm `inference.py` hoặc REST API bằng FastAPI

---

## 🧠 Nhãn phân loại (11 class)

- `dew`, `fogsmog`, `frost`, `glaze`, `hail`, `lightning`,  
  `rain`, `rainbow`, `rime`, `sandstorm`, `snow`

---

## 📌 Ghi chú

- Dự án tối ưu cho chạy trên CPU (hỗ trợ cả GPU NVIDIA nếu có).
- Nếu bạn dùng GPU AMD (như Radeon 780M), chương trình sẽ tự động fallback về CPU.
