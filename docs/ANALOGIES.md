# 🎓 Analogies - Giải Thích Bằng Ví Dụ Đời Sống

Đọc các khái niệm ML/DL bằng ví dụ đời sống, dễ hiểu hơn bao giờ hết!

---

## 📋 Chỉ Mục

- [BiLSTM](#bilstm-bidirectional-lstm)
- [Sliding Window](#sliding-window)
- [Scaling](#scaling-chuẩn-hóa-dữ-liệu)
- [Train/Validation/Test Split](#trainvalidationtest-split)
- [Epochs](#epochs)
- [Batch Size](#batch-size)
- [Learning Rate](#learning-rate)
- [Dropout](#dropout)
- [Overfitting vs Underfitting](#overfitting-vs-underfitting)
- [Loss Function](#loss-function)
- [Callbacks](#callbacks)

---

## BiLSTM (Bidirectional LSTM)

### Giải thích ML
- LSTM: Neural network có khả năng ghi nhớ thông tin dài hạn
- BiLSTM: LSTM nhìn cả quá khứ VÀ tương lai

### Ví dụ đời sống: Đọc một câu văn

**LSTM thường (uni-directional):**
- Bạn đọc câu từ trái → phải
- Khi đọc từ cuối, bạn đã quên từ đầu
- Ví dụ: "Hôm nay trời rất đẹp, tôi thích đi..."
- Khi đọc đến "đi", bạn nhớ "thích" nhưng đã quên "Hôm nay"

**BiLSTM (bi-directional):**
- Bạn đọc câu 2 lần: trái → phải VÀ phải → trái
- Khi đọc bất kỳ từ nào, bạn đều biết cả phần trước và sau
- Ví dụ: "Hôm nay trời rất đẹp, tôi thích đi dạo."
- Khi đọc "thích", bạn biết: trước là "trời rất đẹp", sau là "đi dạo"

### Tại sao BiLSTM tốt hơn cho dự đoán giá?
- Giá Bitcoin bị ảnh hưởng bởi cả quá khứ (60 ngày trước) VÀ tương lai (60 ngày sau)
- Khi train, ta có toàn bộ data → BiLSTM tận dụng được thông tin này

---

## Sliding Window

### Giải thích ML
- Chia dữ liệu thành các sequences (chuỗi) có độ dài cố định
- Mỗi sequence dùng để dự đoán giá trị tiếp theo

### Ví dụ đời sống: Dự đoán thời tiết

Bạn muốn dự đoán thời tiết ngày mai.

**Cách 1: Chỉ nhìn ngày hôm nay**
- Hôm nay: 25°C, nắng
- Dự đoán: Ngày mai 26°C, nắng
- → Không chính xác, chỉ nhìn 1 ngày

**Cách 2: Sliding Window - nhìn 7 ngày trước**
- Thứ 2: 22°C, mưa
- Thứ 3: 23°C, mưa
- ...
- Chủ nhật: 26°C, nắng
- Window size = 7: Bạn nhìn 7 ngày trước → Dự đoán ngày mai
- → Chính xác hơn nhiều!

### Trong dự đoán giá Bitcoin
- Window size = 60: Model nhìn giá 60 ngày trước để dự đoán ngày 61
- Window size = 30: Model nhìn giá 30 ngày trước để dự đoán ngày 31

---

## Scaling (Chuẩn Hóa Dữ Liệu)

### Giải thích ML
- Đưa dữ liệu về khoảng [0, 1] hoặc [-1, 1]
- Giúp model học nhanh và ổn định hơn

### Ví dụ đời sống: Đổi tiền sang tỷ lệ

**Không scaling:**
- Giá nhà: $1,000,000
- Lương: $5,000
- Chi phí ăn uống: $500
- → Số quá chênh lệch, khó so sánh

**Scaling (min-max):**
- Giá nhà: 1.0 (cao nhất)
- Lương: 0.5
- Chi phí ăn uống: 0.0 (thấp nhất)
- → Dễ so sánh, dễ hiểu

### Tại sao phải scale?
- Nếu không scale: $50,000 và $51,000 gần giống nhau, nhưng model khó phân biệt
- Nếu scale: 0.5 và 0.51 → model dễ thấy sự khác biệt

---

## Train/Validation/Test Split

### Giải thích ML
- Chia dữ liệu thành 3 phần để train, validate, và đánh giá model

### Ví dụ đời sống: Học thi đại học

**Train (80%):**
- Bạn học bài ở nhà
- Làm bài tập, đọc sách
- Làm xong đáp án ngay → biết mình đúng/sai

**Validation (10%):**
- Làm đề thử tại trường
- Không có đáp án ngay, chờ thầy chấm
- Điều chỉnh cách học dựa trên kết quả

**Test (10%):**
- Thi thật (đại học, Học viện, v.v.)
- Chưa từng thấy đề này trước
- Kết quả thi: Chỉ có 1 lần!

### Tại sao cần 3 phần?
- Train: Để học pattern
- Validation: Để điều chỉnh cách học (tăng epochs, giảm learning rate, v.v.)
- Test: Để biết model thực chiến được không (chỉ dùng 1 lần!)

---

## Epochs

### Giải thích ML
- Số lần model học qua toàn bộ dữ liệu

### Ví dụ đời sống: Đọc một cuốn sách

**Epoch 1:**
- Đọc cuốn sách lần đầu tiên
- Hiểu sơ sài, nhớ vài điểm chính

**Epoch 2:**
- Đọc lại lần 2
- Hiểu rõ hơn, nhớ nhiều chi tiết hơn

**Epoch 20:**
- Đọc lần thứ 20
- Hiểu rất sâu, nhớ từng chi tiết

### Tại sao cần nhiều epochs?
- Model học lần đầu → chưa hiểu pattern của data
- Model học lại → hiểu rõ hơn
- Model học nhiều lần → hiểu rất sâu

### Nhưng bao nhiêu là đủ?
- Quá ít epochs → underfitting (không hiểu hết)
- Quá nhiều epochs → overfitting (học vẹt)

---

## Batch Size

### Giải thích ML
- Số samples mỗi lần tính gradient (cập nhật weights)

### Ví dụ đời sống: Học từ vựng tiếng Anh

**Batch size = 1 (Online learning):**
- Học 1 từ → kiểm tra → điều chỉnh cách học
- Học tiếp từ tiếp theo → kiểm tra → điều chỉnh
- → Học rất chậm nhưng cập nhật liên tục

**Batch size = 32 (Mini-batch learning):**
- Học 32 từ → kiểm tra 32 từ → điều chỉnh
- Học tiếp 32 từ → kiểm tra → điều chỉnh
- → Học vừa phải, cân bằng

**Batch size = 1000 (Batch learning):**
- Học 1000 từ → kiểm tra 1000 từ → điều chỉnh
- → Học nhanh nhưng có thể bỏ qua chi tiết nhỏ

### Trade-off:
- Batch size nhỏ: Chậm nhưng chính xác
- Batch size lớn: Nhanh nhưng có thể kém chính xác

---

## Learning Rate

### Giải thích ML
- Bước nhảy khi cập nhật weights (điều chỉnh tham số model)

### Ví dụ đời sống: Tìm đường lên đỉnh núi trong sương mù

**Learning rate lớn (0.1):**
- Bước nhảy lớn
- Có thể đích đến nhanh
- Nhưng có thể nhảy quá đích, nhảy xuống vách đá!

**Learning rate nhỏ (0.0001):**
- Bước nhảy nhỏ, cẩn thận
- Chắc chắn đến đích
- Nhưng rất lâu, mệt mỏi

**Learning rate vừa phải (0.001):**
- Bước nhảy vừa phải
- Đến đích nhanh mà an toàn

---

## Dropout

### Giải thích ML
- Bỏ ngẫu nhiên một số neurons trong quá trình training
- Giúp tránh overfitting (học vẹt)

### Ví dụ đời sống: Học trong nhóm

**Không dropout (overfitting):**
- Cùng 1 nhóm học mọi lúc
- Nhớ chính xác ai trả lời câu gì
- Đến thi có nhóm đó → được 10 điểm
- Đến thi không có nhóm đó → rớt!

**Có dropout:**
- Thay đổi thành viên nhóm liên tục
- Học cách học, không chỉ nhớ đáp án
- Đến thi bất kỳ ai → đều làm tốt

### Tại sao Dropout giúp tránh overfitting?
- Model không thể dựa vào một vài neurons cụ thể
- Buộc model học pattern chung, không học vẹt

---

## Overfitting vs Underfitting

### Ví dụ đời sống: Học thi toán

**Underfitting (Học quá ít):**
- Chỉ học sơ lược công thức
- Thi → không làm được bài khó
- Biểu đồ: Train loss cao, Val loss cao

**Tốt (Fit vừa phải):**
- Học vừa phải, hiểu công thức + cách áp dụng
- Thi → làm được cả bài dễ và bài khó
- Biểu đồ: Train loss thấp, Val loss thấp

**Overfitting (Học vẹt):**
- Học vẹt mọi đề thi cũ
- Thi có đề giống đề cũ → được 10 điểm
- Thi có đề mới → rớt
- Biểu đồ: Train loss thấp, Val loss cao

---

## Loss Function

### Giải thích ML
- Hàm đo lường độ sai lệch giữa dự đoán và thực tế
- Model cố gắng giảm loss càng nhỏ càng tốt

### Ví dụ đời sống: Bắn cung

**MSE (Mean Squared Error):**
- Điểm: (khoảng cách²) → Mất điểm nặng hơn nếu bắn lệch nhiều
- Bắn lệch 1cm → mất 1 điểm
- Bắn lệch 10cm → mất 100 điểm
- → Nhấn mạnh vào các lỗi lớn

**MAE (Mean Absolute Error):**
- Điểm: khoảng cách → Mất điểm đều
- Bắn lệch 1cm → mất 1 điểm
- Bắn lệch 10cm → mất 10 điểm
- → Dễ hiểu hơn

---

## Callbacks

### Giải thích ML
- Các functions được gọi trong quá trình training
- Giúp điều chỉnh và kiểm soát training

### Ví dụ đời sống: Huấn luyện viên theo dõi vận động viên

**ModelCheckpoint:**
- Lưu lại kỷ lục tốt nhất
- Về sau có thể load lại kỷ lục này

**EarlyStopping:**
- Nếu vận động viên không cải thiện sau N lần tập → dừng
- Tiết kiệm thời gian, tránh overtraining

**ReduceLROnPlateau:**
- Nếu không cải thiện → giảm cường độ tập
- Giúp "fine-tune" tốt hơn

---

## Metrics (MAE, RMSE, MAPE)

### Ví dụ đời sống: Đánh giá dự báo thời tiết

**MAE (Mean Absolute Error):**
- Sai số trung bình tuyệt đối
- Dự báo sai trung bình 2°C
- Dễ hiểu: "Bình thường mình sai 2 độ thôi"

**RMSE (Root Mean Squared Error):**
- Căn bậc 2 của sai số bình phương
- Dự báo sai 2°C nhưng có vài ngày sai 10°C → RMSE sẽ cao hơn
- Nhấn mạnh vào các lỗi lớn (outliers)

**MAPE (Mean Absolute Percentage Error):**
- Sai số phần trăm trung bình
- Dự báo sai trung bình 5%
- Độc lập với scale: 5°C sai khi 20°C (25%) vs 5°C sai khi 100°C (5%)

---

## 🎯 Tóm Tắt

| Khái niệm | Ví dụ đời sống |
|-----------|---------------|
| BiLSTM | Đọc câu 2 chiều |
| Sliding Window | Dự đoán thời tiết dựa trên 7 ngày trước |
| Scaling | Đổi tiền sang tỷ lệ |
| Train/Val/Test | Học ở nhà, đề thử, thi thật |
| Epochs | Đọc sách nhiều lần |
| Batch Size | Học từ vựng theo nhóm |
| Learning Rate | Bước nhảy tìm đường lên đỉnh núi |
| Dropout | Thay đổi nhóm học |
| Overfitting | Học vẹt đề thi cũ |
| Loss | Điểm bắn cung |
| Callbacks | Huấn luyện viên theo dõi |


