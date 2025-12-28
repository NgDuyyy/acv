# Image Captioning Pipeline

Đây là codebase toàn diện cho dự án **Image Captioning**, kết hợp sức mạnh của thị giác máy tính và xử lý ngôn ngữ tự nhiên. Dự án triển khai quy trình trích xuất đặc trưng kép: sử dụng **Faster R-CNN** cho đặc trưng vùng (Bottom-Up Attention) và **RelTR (Relation Transformer)** để xây dựng biểu đồ ngữ cảnh (Scene Graph). Các đặc trưng này được tích hợp vào mô hình giải mã **LSTM** (với cơ chế Attention) để sinh ra các câu mô tả ảnh chính xác, tự nhiên và giàu ngữ nghĩa hơn.

## Cấu trúc thư mục

```text
├── configs
│   └── lstm_train.yml
├── data
│   ├── features
│   ├── LSTM
│   ├── RelTR_ckpt
│   ├── test
│   ├── train
│   └── val
├── models
│   ├── feature_extracting
│   ├── LSTM
│   └── RelTR
├── scripts
│   ├── create_reltr_hdf5.py
│   ├── extract_reltr_features.py
│   ├── generate_tsv.py
│   ├── make_bu_data.py
│   ├── prepare_data_merged.py
│   ├── prepro_labels.py
│   └── prepro_reference_json.py
├── utils
│   ├── feature_extracting
│   ├── LSTM
│   └── RelTR
├── complete_image_captioning_pipeline.py
├── eval.py
├── infer.py
└── train.py
```

## Kết quả thực nghiệm (Experimental Results)

| Method | BLEU-1 | BLEU-4 | METEOR | ROUGE_L | CIDEr | SPICE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RelTR (Scene Graph) | 72.1 | 39.1 | 34.7 | 55.9 | 123.3 | 8.7 |

## 1. Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

Đảm bảo dữ liệu ảnh đã được đặt trong `data/train`, `data/val`, và `data/test`.

### Bước 1: Trích xuất đặc trưng hình ảnh (Visual Features)

Bắt buộc cho mọi mô hình. Sử dụng Faster R-CNN để trích xuất đặc trưng vùng (ROI).

*   **Lệnh chạy:**
    ```bash
    python scripts/generate_tsv.py --cuda --cfg models/cfgs/res101.yml --net res101
    ```
    *(Lưu ý: Cần cấu hình đúng đường dẫn ảnh và output trong script hoặc qua tham số nếu có)*

*   **Input:** Ảnh trong `data/train`, `data/val`, `data/test`.
*   **Output:** Các file `.tsv` được lưu trong `data/features/features_tsv/`.

### Bước 2: Trích xuất đặc trưng Scene Graph (RelTR)

Bắt buộc nếu bạn dùng mô hình có RelTR. Bước này tạo ra file `.h5` chứa thông tin quan hệ giữa các vật thể.

*   **Lệnh chạy:**
    ```bash
    python scripts/create_reltr_hdf5.py --json_path data/LSTM/data_merged.json --data_root data --output_h5 data/features/reltr_features.h5
    ```
*   **Input:** File JSON đã gộp (`data/LSTM/data_merged.json`) và thư mục ảnh gốc.
*   **Output:** File `data/features/reltr_features.h5` chứa đặc trưng quan hệ (Subject-Predicate-Object).

### Bước 3: Định dạng đặc trưng (Feature Formatting)

Chuyển đổi file TSV sang định dạng `.npy` để huấn luyện nhanh hơn và chia về đúng thư mục.

*   **Lệnh chạy:**
    ```bash
    python scripts/make_bu_data.py
    ```
*   **Input:** Các file `.tsv` trong `data/features/features_tsv/`.
*   **Output:** Các thư mục `features_extracted_att`, `features_extracted_fc`, `features_extracted_box` bên trong `data/features`.

### Bước 4: Chuẩn bị dữ liệu huấn luyện (Data Preparation)

Gộp và xử lý file annotation từ các tập train/val thành một file JSON thống nhất cho huấn luyện.

*   **Lệnh chạy:**
    ```bash
    python scripts/prepare_data_merged.py
    ```
    *(Lưu ý: Kiểm tra và sửa đường dẫn `train_path`, `val_path`, `output_path` trong file này nếu cần)*

*   **Input:** `data/train/train_data.json`, `data/val/val_data.json`.
*   **Output:** `data/LSTM/data_merged.json`.

### Bước 5: Tạo file tham chiếu đánh giá (Evaluation Reference)

Tạo file JSON tham chiếu chuẩn cho việc đánh giá (nếu cần thiết cho `coco-caption`).

*   **Lệnh chạy:**
    ```bash
    python scripts/prepro_reference_json.py --input_json data/val/val_data.json --output_json data/LSTM/val_reference.json
    ```

### Bước 6: Tiền xử lý nhãn (Label Preprocessing)

Tạo bộ từ điển (vocabulary) và file H5 chứa nhãn đã mã hóa cho mô hình.

*   **Lệnh chạy:**
    ```bash
    python scripts/prepro_labels.py --input_json data/LSTM/data_merged.json --output_json data/LSTM/data_label.json --output_h5 data/LSTM/data
    ```
*   **Input:** `data/LSTM/data_merged.json`.
*   **Output:** `data/LSTM/data_label.h5` và `data/LSTM/data_label.json`.


## 3. Huấn luyện (Training)

Huấn luyện mô hình Image Captioning.

*   **Lệnh chạy:**
    ```bash
    python train.py --cfg configs/lstm_train.yml
    ```
*   **Tham số quan trọng:**
    *   `--id`: Tên định danh cho lần chạy này.
    *   `--caption_model`: Loại mô hình (ví dụ: `updown`, `att2in`, `transformer`).
    *   `--checkpoint_path`: Thư mục lưu checkpoint.
    *   `--checkpoint_path`: Thư mục lưu checkpoint.
    *   **Lưu ý cho RelTR:** Để bật tính năng Scene Graph, hãy đảm bảo config `configs/lstm_train.yml` có dòng `input_rel_dir: data/features/reltr_features.h5`. Nếu muốn chạy Baseline (không Graph), hãy để trống dòng này (`input_rel_dir: ""`).
    *   Các tham số được chỉnh trong `configs/lstm_train.yml`.
*   **Output:**
    *   Checkpoint mô hình: `result/checkpoints/model-best.pth`.
    *   File thông tin huấn luyện: `result/checkpoints/infos_my_run-best.pkl`.
    *   Log huấn luyện: `result/training_history_data.csv`.

## 4. Đánh giá (Evaluation)

Đánh giá mô hình đã huấn luyện trên tập val/test, sinh caption và tính toán các chỉ số (BLEU, CIDEr, SPICE...).

*   **Lệnh chạy:**
    ```bash
    python eval.py \
      --model result/log_lstm/model-best.pth \
      --infos_path result/log_lstm/infos_-best.pkl \
      --input_json data/LSTM/val_reference.json \
      --language_eval_json data/LSTM/val_reference.json \
      --input_att_dir data/features/features_extracted_att \
      --split val \
      --language_eval 1 \
      --save_csv_results 1 \
      --predictions_csv result/predictions_val.csv \
      --metrics_csv result/scores_val.csv
    ```

*   **Giải thích tham số:**
    *   `--model`, `--infos_path`: Đường dẫn đến checkpoint mô hình và file info.
    *   `--input_json` & `--language_eval_json`: File JSON chứa danh sách ảnh và caption gốc (Ground Truth).
    *   `--input_att_dir`: Thư mục chứa features (attention) tương ứng.
    *   `--language_eval 1`: Bật tính năng chấm điểm ngôn ngữ.
    *   `--save_csv_results 1`: Bật tính năng lưu kết quả ra file CSV.

*   **Output:**
    *   **`result/predictions_val.csv`**: Chứa filename, caption gốc và caption dự đoán.
    *   **`result/scores_val.csv`**: Chứa bảng điểm chi tiết (BLEU-1..4, METEOR, ROUGE_L, CIDEr, SPICE).

## 5. Full Pipeline

Chạy toàn bộ quy trình từ trích xuất đặc trưng đến sinh caption cho một thư mục ảnh mới.

*   **Lệnh chạy:**
    ```bash
    python complete_image_captioning_pipeline.py full --images_dir data/my_new_images --output_root result/my_output --model result/checkpoints/model-best.pth --infos result/checkpoints/infos_my_run-best.pkl
    ```



## 6. Inference đơn giản

Chạy inference trên một ảnh bất kỳ để kiểm tra nhanh kết quả.

*   **Lệnh chạy:**
    ```bash
    python infer.py --image test_image/my_image.jpg --frcnn_model models/feature_extracting/pretrained_model/faster_rcnn_res101_vg.pth --caption_model result/log_lstm/model-best.pth --infos_path result/log_lstm/infos_-best.pkl --gpu
    ```

*   **Tham số:**
    *   `--image`: Đường dẫn ảnh cần đặt caption.
    *   `--frcnn_model`: Đường dẫn đến checkpoint Faster R-CNN (dùng để trích xuất đặc trưng).
    *   `--caption_model`: Đường dẫn đến checkpoint mô hình Captioning đã train.
    *   `--infos_path`: Đường dẫn đến file infos.pkl tương ứng với checkpoint.
    *   `--gpu`: Thêm cờ này để chạy trên GPU (mặc định chạy CPU nếu không có cờ này).

*   **Output:** In trực tiếp caption dự đoán ra màn hình.

### Chạy Inference với RelTR (Scene Graph)

Để sử dụng mô hình có tích hợp Scene Graph:

*   **Lệnh chạy:**
    ```bash
    python infer.py --use_reltr --image test_image/my_image.jpg ... (các tham số khác như trên)
    ```
*   **Tham số thêm:**
    *   `--use_reltr`: Bật tính năng sử dụng RelTR.
    *   `--reltr_model_path`: Đường dẫn đến checkpoint RelTR (mặc định: `data/RelTR_ckpt/checkpoint0149.pth`).

## Ghi chú

*   Để sử dụng CPU thay vì GPU, thêm cờ `--force_cpu 1` (tuy nhiên huấn luyện sẽ rất chậm).
*   Kết quả đánh giá và log sẽ được lưu tập trung trong thư mục `result/` để dễ quản lý.