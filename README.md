# Image Captioning Project

## Giới Thiệu Chung

Dự án này tập trung nghiên cứu và triển khai các mô hình \textbf{Image Captioning} (Mô tả Hình ảnh) – nhiệm vụ kết hợp giữa Computer Vision và Xử lý Ngôn ngữ Tự nhiên (NLP). Mục tiêu là xây dựng hệ thống có khả năng tự động sinh ra một câu mô tả tự nhiên và chính xác cho bất kỳ hình ảnh đầu vào nào.

Chúng tôi đã triển khai và so sánh hai kiến trúc tiên tiến để giải quyết bài toán này:

1.  **Encoder-Decoder (CNN-LSTM):** Kiến trúc nền tảng sử dụng CNN để trích xuất đặc trưng ảnh và LSTM để giải mã thành chuỗi từ.
2.  **Detector-Decoder:** Kiến trúc nâng cao tập trung vào việc phát hiện đối tượng trong ảnh trước (Detector) để cung cấp thông tin ngữ cảnh cục bộ chi tiết hơn cho bộ giải mã (Decoder).

---

## Thành Viên Nhóm & Phân Công Nhiệm Vụ (Giữa Kỳ)

| Họ và Tên | Mã sinh viên | Vai trò | Nhiệm vụ chính đến giữa kỳ |
| :--- | :--- | :--- | :--- |
| Trịnh Hải Đăng | 220015xx | Nhóm trưởng | Xây dựng Dataset & Kiểm thử tổng thể |
| Thái Bảo | 220015yy | Thành viên | Xây dựng Dataset & Kiểm thử tổng thể |
| Nguyễn Đình Duy | **22001554** | Thành viên | \textbf{Phát triển Mô hình Encoder-Decoder} |
| Đặng Tùng Anh | **22001537** | Thành viên | \textbf{Phát triển Mô hình Detector-Decoder} |
| Trần Ngọc Nam Hải | 220015zz | Thành viên | Phát triển Mô hình Detector-Decoder |

---

## Cấu Trúc Dự Án & Hướng Dẫn Chạy Code

Toàn bộ code và hướng dẫn chi tiết cho từng kiến trúc mô hình được đặt trong các nhánh riêng biệt để tiện cho việc phát triển và kiểm tra:

* **Kiến trúc Encoder-Decoder (CNN-LSTM):**
    * **Nhánh:** `encoder_decoder`
    * **Nội dung:** Code nguồn, file cấu hình, và hướng dẫn chạy (`README`) cho mô hình CNN-LSTM cơ bản.

* **Kiến trúc Detector-Decoder:**
    * **Nhánh:** `detector_decoder`
    * **Nội dung:** Code nguồn, file cấu hình, và hướng dẫn chạy (`README`) cho mô hình dựa trên phát hiện đối tượng.

**Vui lòng chuyển sang nhánh tương ứng để tìm hướng dẫn cài đặt và thực thi code.**
