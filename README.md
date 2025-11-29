# Hướng dẫn chạy Image-Captioning

## 1. Tổng quan
Dự án triển khai mô hình sinh mô tả ảnh (image captioning) sử dụng encoder ResNet101 và decoder LSTM với beam search cho suy luận. Cấu trúc chính:

- `config.py`: nơi tập trung mọi đường dẫn và siêu tham số.
- `prepare_data.py`: chuyển đổi dữ liệu gốc (JSON + ảnh) sang định dạng HDF5/JSON xử lý sẵn.
- `train_and_evaluate.py`: huấn luyện mô hình, lưu checkpoint tốt nhất và ghi lịch sử.
- `eval.py`: chạy beam search trên một tập (train/val/test/…​) và ghi BLEU/METEOR/ROUGE/CIDEr.
- `run_single_inference.py`: caption một ảnh đơn.
- `scripts/visualize_predictions.py`: caption tối đa 3 ảnh và sinh PDF hiển thị GT vs. dự đoán.
- `plot_training_history.py`: vẽ biểu đồ loss/BLEU từ file lịch sử.

## 2. Yêu cầu
- Python >= 3.9 (đã kiểm thử với 3.10).
- GPU CUDA được khuyến nghị, nhưng có thể chạy CPU (chậm hơn).
- Thư viện liệt kê trong `requirements.txt` (PyTorch, torchvision, NLTK, pandas, matplotlib, v.v.).

## 3. Chuẩn bị môi trường
```powershell
# 1) Tạo và kích hoạt virtual env (tùy chọn nhưng nên dùng)
python -m venv .venv
.\.venv\Scripts\activate

# 2) Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý PowerShell:** nếu muốn xuống dòng trong lệnh dài, hãy dùng ký tự `` ` `` (backtick) ở cuối dòng, không dùng `^`.

## 4. Chuẩn bị dữ liệu
Dự án mong đợi cấu trúc thư mục sau (ví dụ):
```
data/
  train/
    images/              # Ảnh JPEG
    train_data.json      # JSON COCO-like chứa trường images/annotations
    processed/           # Sẽ được tạo tự động
  valid/
    images/
    val_data.json
    processed/
  test/
    images/
    test_data.json
    processed/
```
Các file JSON cần chứa trường `images` (với `id`, `filename`) và `annotations` (với `image_id`, `caption`). Sau khi đã đặt dữ liệu đúng vị trí, chạy:
```powershell
python prepare_data.py --force
```
Tùy chọn quan trọng:
- `--max-len`, `--captions-per-image`, `--min-word-freq`: khớp với tham số huấn luyện mong muốn.
- Bỏ `--force` nếu chỉ muốn tái tạo test set khi train/val đã tồn tại.

## 5. Huấn luyện
```powershell
python train_and_evaluate.py `
    --patience 20 `
    --lr-patience 8 `
    --max-len 50 `
    --captions-per-image 5 `
    --min-word-freq 5
```
Thông tin chính:
- Checkpoint tốt nhất được lưu trong `result/pretrained_parameters/` (theo BLEU-4 val).
- Lịch sử huấn luyện ghi vào `result/log_history/training_history.csv` mỗi epoch.
- Thay đổi siêu tham số chung (batch size, embedding dim, lr, số epoch mặc định, …​) trong `config.py`.

## 6. Đánh giá mô hình
Sử dụng `eval.py` để chấm BLEU/METEOR/ROUGE/CIDEr:
```powershell
python eval.py `
    --split TEST `
    --beam-size 5 `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --log-csv result\log_history\eval_history_test.csv
```
Ghi chú:
- `--split` hỗ trợ `TRAIN`, `VAL`, `TEST` và các alias `TEST_V2 … TEST_V6` (nếu bạn tái sử dụng thư mục test hiện tại cho các biến thể đó). Để dùng alias, hãy chắc chắn `utils/datasets.py` ánh xạ chúng tới `PROCESSED_TEST_DIR` hoặc cập nhật `config.TEST_DIR` tương ứng.
- Thêm `--no-log` nếu chỉ muốn in kết quả, không ghi CSV.
- `beam-size = 5` thường cho chất lượng tốt mà thời gian hợp lý; bạn có thể kiểm chứng bằng cách thay đổi và so sánh log.

## 7. Suy luận & trực quan hóa
### 7.1 Caption một ảnh đơn
```powershell
python run_single_inference.py `
    --img data/test/images/000123.jpg `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --beam-size 5
```
Tùy chọn `--word-map` nếu cần chỉ định file `WORDMAP_*.json` khác. Kết quả in ra console.

### 7.2 Sinh PDF GT vs. Pred cho 3 ảnh
```powershell
python scripts/visualize_predictions.py `
    --images data/test/images/000098.jpg data/test/images/000445.jpg data/test/images/000500.jpg `
    --gt-json data/test/test_data.json `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --output result\visualizations\demo.pdf
```
- Script sẽ lấy tối đa 3 ảnh đầu tiên trong danh sách, chạy beam search và lưu **mỗi ảnh** thành một file PDF riêng (ví dụ `demo_000101.pdf`).
- Có thể chỉnh mặc định trong biến `DEFAULT_IMAGE_PATHS` nếu muốn không truyền tham số.

## 8. Vẽ biểu đồ lịch sử huấn luyện
Sau khi có `training_history.csv`, chạy:
```powershell
python plot_training_history.py `
    --history result/log_history/training_history.csv `
    --output charts/training_history.png
```
Không truyền `--output` nếu bạn muốn hiển thị cửa sổ matplotlib.

## 9. Mẹo & khắc phục sự cố
- **ModuleNotFoundError khi chạy script trong `scripts/`**: đảm bảo chạy từ thư mục gốc dự án hoặc đã thêm `PROJECT_ROOT` vào `PYTHONPATH`. `visualize_predictions.py` tự xử lý việc này.
- **Sai đường dẫn dữ liệu**: kiểm tra `config.py` để chắc chắn các biến `*_DIR` và `RAW_*` trỏ đúng thư mục bạn đang sử dụng (ví dụ khi tạo thêm test_vx, cập nhật `TEST_DIR`).
- **Muốn thử beam size khác**: dùng đối số `--beam-size` cho cả `eval.py`, `run_single_inference.py` và `scripts/visualize_predictions.py`.
- **Dọn lại processed data**: xoá thủ công thư mục `data/*/processed` hoặc dùng `python prepare_data.py --force`.
- **Thiếu ground truth khi visualize**: script sẽ hiển thị thông báo `(Không tìm thấy ground truth)` nếu không map được `filename` trong JSON; kiểm tra lại `image_path.name` có xuất hiện trong `images` của JSON hay không.

Nếu thêm tính năng mới (ví dụ test split khác), chỉ cần cập nhật `config.py` và `utils/datasets.py`, rồi lặp lại các bước trên.
