# Image Captioning Pipeline

Đây là codebase cho dự án Image Captioning, hỗ trợ trích xuất đặc trưng bottom-up, huấn luyện mô hình và đánh giá kết quả.

## Cấu trúc thư mục

*   `data/`: Chứa dữ liệu ảnh và nhãn (chia thành `train`, `val`, `test`).
*   `models/`: Chứa mã nguồn định nghĩa các mô hình (`.py`).
*   `utils/`: Chứa các hàm hỗ trợ đọc dữ liệu, đánh giá, v.v.
*   `result/`: Chứa kết quả đầu ra (checkpoints, logs, features, eval results).
*   `scripts/`: Chứa các script tiền xử lý dữ liệu.
*   `train.py`, `eval.py`, `complete_image_captioning_pipeline.py`: Các file chạy chính.

## 1. Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

Đảm bảo dữ liệu ảnh đã được đặt trong `data/train`, `data/val`, và `data/test`.

### Bước 1: Trích xuất đặc trưng (Feature Extraction)

Trích xuất đặc trưng bottom-up từ ảnh sử dụng Faster R-CNN.

*   **Lệnh chạy:**
    ```bash
    python scripts/generate_tsv.py --cuda --cfg models/cfgs/res101.yml --net res101
    ```
    *(Lưu ý: Cần cấu hình đúng đường dẫn ảnh và output trong script hoặc qua tham số nếu có)*

*   **Input:** Ảnh trong `data/train`, `data/val`, `data/test`.
*   **Output:** Các file `.tsv` được lưu trong `result/features_tsv/`.

### Bước 2: Định dạng đặc trưng (Feature Formatting)

Chuyển đổi file TSV sang định dạng `.npy` để huấn luyện nhanh hơn và chia về đúng thư mục.

*   **Lệnh chạy:**
    ```bash
    python scripts/make_bu_data.py
    ```
*   **Input:** Các file `.tsv` trong `result/features_tsv/`.
*   **Output:** Các thư mục `att`, `fc`, `box` bên trong `data/train/`, `data/val/`, `data/test/`.

### Bước 3: Chuẩn bị dữ liệu huấn luyện (Data Preparation)

Gộp và xử lý file annotation từ các tập train/val thành một file JSON thống nhất cho huấn luyện.

*   **Lệnh chạy:**
    ```bash
    python scripts/prepare_data_merged.py
    ```
    *(Lưu ý: Kiểm tra và sửa đường dẫn `train_path`, `val_path`, `output_path` trong file này nếu cần)*

*   **Input:** `data/train/train_data.json`, `data/val/val_data.json`.
*   **Output:** `data/LSTM/data.json`.

### Bước 4: Tạo file tham chiếu đánh giá (Evaluation Reference)

Tạo file JSON tham chiếu chuẩn cho việc đánh giá (nếu cần thiết cho `coco-caption`).

*   **Lệnh chạy:**
    ```bash
    python scripts/prepro_reference_json.py --input_json data/val/val_data.json --output_json data/val/val_reference.json
    ```

### Bước 5: Tiền xử lý nhãn (Label Preprocessing)

Tạo bộ từ điển (vocabulary) và file H5 chứa nhãn đã mã hóa cho mô hình.

*   **Lệnh chạy:**
    ```bash
    python scripts/prepro_labels.py --input_json data/LSTM/data.json --output_json data/LSTM/coco_label.json --output_h5 data/LSTM/coco_label
    ```
*   **Input:** `data/LSTM/data.json`.
*   **Output:** `data/LSTM/coco_label.h5` và `data/LSTM/coco_label.json`.

### Bước 6: Tính toán N-gram (N-gram Precomputation)

Tính toán trước tần suất n-gram để tăng tốc độ tính toán điểm CIDEr trong quá trình huấn luyện.

*   **Lệnh chạy:**
    ```bash
    python scripts/prepro_ngrams.py --input_json data/LSTM/data.json --dict_json data/LSTM/coco_label.json --output_pkl data/LSTM/coco-train --split train
    ```

## 3. Huấn luyện (Training)

Huấn luyện mô hình Image Captioning.

*   **Lệnh chạy:**
    ```bash
    python train.py --id my_run --caption_model updown --input_json data/LSTM/coco_label.json --input_label_h5 data/LSTM/coco_label.h5 --batch_size 10 --learning_rate 5e-4 --checkpoint_path result/checkpoints
    ```
*   **Tham số quan trọng:**
    *   `--id`: Tên định danh cho lần chạy này.
    *   `--caption_model`: Loại mô hình (ví dụ: `updown`, `att2in`, `transformer`).
    *   `--checkpoint_path`: Thư mục lưu checkpoint.
*   **Output:**
    *   Checkpoint mô hình: `result/checkpoints/model-best.pth`.
    *   File thông tin huấn luyện: `result/checkpoints/infos_my_run-best.pkl`.
    *   Log huấn luyện: `result/training_history_data.csv`.

## 4. Đánh giá (Evaluation)

Đánh giá mô hình đã huấn luyện trên tập test và sinh caption.

*   **Lệnh chạy:**
    ```bash
    python complete_image_captioning_pipeline.py caption --dataset_json data/test/test_data.json --feature_path data/test/att --model result/checkpoints/model-best.pth --infos result/checkpoints/infos_my_run-best.pkl
    ```
    *(Lưu ý: `feature_path` trỏ đến thư mục chứa đặc trưng attention của tập test)*

*   **Output:**
    *   Kết quả đánh giá chi tiết (JSON): `result/eval_results/`.
    *   File dự đoán caption: `result/caption_predictions.csv`.

## 5. Full Pipeline

Chạy toàn bộ quy trình từ trích xuất đặc trưng đến sinh caption cho một thư mục ảnh mới.

*   **Lệnh chạy:**
    ```bash
    python complete_image_captioning_pipeline.py full --images_dir data/my_new_images --output_root result/my_output --model result/checkpoints/model-best.pth --infos result/checkpoints/infos_my_run-best.pkl
    ```

## Ghi chú

*   Để sử dụng CPU thay vì GPU, thêm cờ `--force_cpu 1` (tuy nhiên huấn luyện sẽ rất chậm).
*   Kết quả đánh giá và log sẽ được lưu tập trung trong thư mục `result/` để dễ quản lý.