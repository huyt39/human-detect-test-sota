# P-DESTRE Dataset Processor for YOLO Training

Tool để xử lý dataset P-DESTRE cho tác vụ huấn luyện YOLO cho human detection.

## 📋 Mô tả

Script này chuyển đổi dataset P-DESTRE từ format video + annotation sang format YOLO chuẩn, bao gồm:
- Trích xuất frames từ video theo tỷ lệ frame rate
- Chuyển đổi annotation từ format P-DESTRE sang YOLO format
- Chia dataset thành train/validation/test splits
- Tạo file cấu hình YOLO
- Lọc annotation theo confidence và kích thước bounding box

## 📁 Cấu trúc Dataset

```
P-DESTRE/
├── videos/
│   ├── 08-11-2019-1-1.MP4
│   ├── 08-11-2019-1-2.MP4
│   └── ...
└── annotation/
    ├── 08-11-2019-1-1.txt
    ├── 08-11-2019-1-2.txt
    └── ...
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install opencv-python pandas numpy tqdm pyyaml
```

### 2. Chạy với tham số mặc định

```bash
python run_data_processing.py
```

### 3. Chạy với tham số tùy chỉnh

```bash
python data_processor.py \
    --dataset_path P-DESTRE \
    --output_path processed_data \
    --frame_rate 5 \
    --min_confidence 0.5 \
    --min_box_size 20 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1
```

## ⚙️ Tham số

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|------------------|
| `--dataset_path` | Đường dẫn đến thư mục P-DESTRE | Bắt buộc |
| `--output_path` | Thư mục lưu dữ liệu đã xử lý | Bắt buộc |
| `--frame_rate` | Tỷ lệ trích xuất frame (1=every frame, 5=every 5th frame) | 1 |
| `--min_confidence` | Ngưỡng confidence tối thiểu (0-1) | 0.5 |
| `--min_box_size` | Kích thước bounding box tối thiểu (pixel) | 20 |
| `--train_ratio` | Tỷ lệ dữ liệu training | 0.7 |
| `--val_ratio` | Tỷ lệ dữ liệu validation | 0.2 |
| `--test_ratio` | Tỷ lệ dữ liệu testing | 0.1 |
| `--seed` | Random seed cho reproducibility | 42 |

## 📊 Output Structure

Sau khi xử lý, cấu trúc thư mục output sẽ như sau:

```
processed_data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── config/
│   └── dataset.yaml
└── dataset_info.json
```

### Files được tạo:

1. **`images/{split}/`**: Frames được trích xuất từ video
2. **`labels/{split}/`**: Annotation files theo format YOLO
3. **`config/dataset.yaml`**: File cấu hình cho YOLO training
4. **`dataset_info.json`**: Thông tin chi tiết về dataset đã xử lý

## 📝 Format Annotation

### Input (P-DESTRE format):
```
frame_id, track_id, x, y, width, height, confidence, ...
```

### Output (YOLO format):
```
class_id x_center y_center width height
```

Trong đó:
- `class_id`: 0 (person)
- `x_center, y_center`: Tọa độ tâm bounding box (normalized 0-1)
- `width, height`: Kích thước bounding box (normalized 0-1)

## 🔧 YOLO Configuration

File `config/dataset.yaml` được tạo tự động:

```yaml
path: /path/to/processed_data
train: images/train
val: images/val
test: images/test
nc: 1
names: ['person']
```

## 📈 Thống kê

Script sẽ hiển thị thống kê chi tiết sau khi xử lý:

```
==================================================
PROCESSING SUMMARY
==================================================
Total frames extracted: 12345
Frame extraction rate: 5
Minimum confidence: 0.5
Minimum box size: 20px

TRAIN SPLIT:
  Videos: 45/50
  Frames: 8641

VAL SPLIT:
  Videos: 13/14
  Frames: 2469

TEST SPLIT:
  Videos: 7/8
  Frames: 1235

Output directory: processed_data
==================================================
```

## ⚠️ Lưu ý

1. **Frame Rate**: Sử dụng `frame_rate > 1` để giảm kích thước dataset
2. **Confidence**: Tăng `min_confidence` để lọc bỏ detection kém chất lượng
3. **Box Size**: Tăng `min_box_size` để lọc bỏ object quá nhỏ
4. **Memory**: Xử lý video lớn có thể tốn nhiều RAM

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **"Videos directory not found"**
   - Kiểm tra đường dẫn dataset_path
   - Đảm bảo thư mục videos/ tồn tại

2. **"Annotation file not found"**
   - Kiểm tra tên file annotation có khớp với video không
   - Đảm bảo thư mục annotation/ tồn tại

3. **"No video files found"**
   - Kiểm tra định dạng file video (.MP4, .mp4, .avi, .mov)
   - Đảm bảo file video không bị hỏng

## 📄 License

MIT License

## 🤝 Contributing

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request. 