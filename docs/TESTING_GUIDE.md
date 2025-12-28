# 📚 Hướng Dẫn Test Mô Hình - Trả Lời Các Thắc Mắc

**Ngày tạo:** 2025-12-28

---

## ❓ Câu Hỏi 1: Cố Định Dữ Liệu Hay Thay Đổi Số Lượng?

### 🎯 Câu Trả Lời Ngắn Gọn:
**CẢ HAI CÁCH ĐỀU CÓ GIÁ TRỊ, nhưng tùy vào mục đích:**

### 📊 Phân Tích Chi Tiết:

#### **Cách 1: Cố Định 1 Lượng Dữ Liệu (Ví dụ: 30k), Chỉ Thay Đổi Tham Số**
✅ **Ưu điểm:**
- **So sánh công bằng**: Tất cả mô hình dùng cùng dataset → chỉ khác nhau về hyperparameters
- **Kiểm soát biến số**: Dễ xác định yếu tố nào ảnh hưởng đến kết quả (window_size, LSTM units, epochs...)
- **Phù hợp cho nghiên cứu**: Khi muốn tìm hiểu tác động của từng tham số riêng lẻ

❌ **Nhược điểm:**
- Không biết được liệu nhiều dữ liệu hơn có cải thiện kết quả không
- Có thể bỏ lỡ cơ hội cải thiện bằng cách tăng dữ liệu

**Khi nào dùng:**
- Khi muốn **tối ưu hyperparameters** (window_size, LSTM units, dropout, learning rate...)
- Khi muốn **so sánh công bằng** giữa các kiến trúc mô hình khác nhau
- Khi có **thời gian hạn chế** và muốn tập trung vào tuning tham số

---

#### **Cách 2: Cùng Dataset, Trích Nhiều Lần Với Số Lượng Khác Nhau (10k, 30k, 50k...)**
✅ **Ưu điểm:**
- **Hiểu được tác động của dữ liệu**: Biết được liệu nhiều dữ liệu có cải thiện kết quả không
- **Tìm điểm tối ưu**: Xác định được lượng dữ liệu "đủ" cho bài toán của bạn
- **Phù hợp cho production**: Biết được cần bao nhiêu dữ liệu để đạt kết quả tốt nhất

❌ **Nhược điểm:**
- Khó so sánh công bằng vì mỗi test dùng dataset khác nhau
- Tốn thời gian hơn (phải train nhiều lần với dữ liệu khác nhau)

**Khi nào dùng:**
- Khi muốn **tìm lượng dữ liệu tối ưu** cho bài toán
- Khi muốn **chứng minh tác động của dữ liệu** đến độ chính xác
- Khi chuẩn bị **production** và cần biết cần bao nhiêu dữ liệu

---

### 💡 **KHUYẾN NGHỊ CHO BẠN:**

Dựa vào kết quả test hiện tại của bạn (`reports/notebook/EVALUATION.md`):

1. **Giai đoạn hiện tại (đã làm):** ✅ Đúng rồi!
   - Bạn đã test với nhiều lượng dữ liệu khác nhau (10k, 20k, 30k, 50k)
   - Điều này giúp bạn hiểu được tác động của dữ liệu

2. **Bước tiếp theo (khuyến nghị):**
   - **Cố định 30k dữ liệu** (vì w96 với 30k cho kết quả tốt nhất)
   - **Thay đổi các tham số khác:**
     - Window size: 72, 96, 120, 144
     - LSTM units: [32,16], [64,32], [128,64,32]
     - Dropout: 0.1, 0.2, 0.3
     - Learning rate: 0.0001, 0.001, 0.01
   - Điều này giúp bạn **tối ưu hyperparameters** trên dataset đã chọn

3. **Quy trình đề xuất:**
   ```
   Bước 1: Tìm lượng dữ liệu tối ưu (đã làm ✅)
   Bước 2: Cố định lượng dữ liệu đó, tối ưu hyperparameters (nên làm tiếp)
   Bước 3: Test với dataset mới (out-of-sample) để validate
   ```

---

## ❓ Câu Hỏi 2: Càng Nhiều Dữ Liệu Càng Chính Xác?

### 🎯 Câu Trả Lời Ngắn Gọn:
**KHÔNG PHẢI LUÔN LUÔN!** Nhiều dữ liệu chỉ giúp khi:
- Dữ liệu có chất lượng tốt
- Mô hình đủ phức tạp để học được patterns
- Không bị overfitting

### 📊 Phân Tích Từ Kết Quả Của Bạn:

Từ `EVALUATION.md`, ta thấy:

| Model | Data Rows | MAE ($) | RMSE ($) | MAPE (%) | Direction Accuracy (%) |
|-------|-----------|---------|----------|----------|------------------------|
| w24   | 10,000    | 427.97  | 627.18   | 0.49%    | **53.90%** ⭐          |
| w48   | 20,000    | 2,929.36| 3,094.21 | 3.30%    | 51.10%                 |
| w96   | 30,000    | **424.71** ⭐ | **601.66** ⭐ | **0.47%** ⭐ | 52.78%          |
| w144  | 50,000    | 870.68  | 1,006.23 | 0.91%    | 50.46%                 |

**Nhận xét:**
- ✅ **30k dữ liệu (w96) cho kết quả TỐT NHẤT** - không phải 50k!
- ❌ **20k dữ liệu (w48) cho kết quả TỆ NHẤT** - có thể do cấu hình không phù hợp
- ⚠️ **50k dữ liệu (w144) không tốt hơn 30k** - có thể do:
  - Overfitting
  - Window size quá lớn (144) không phù hợp với dữ liệu
  - Cần nhiều epochs hơn để học được patterns

---

### 🔍 **CÁC YẾU TỐ QUYẾT ĐỊNH ĐỘ CHÍNH XÁC:**

#### 1. **Chất Lượng Dữ Liệu** (Quan trọng nhất!)
- ✅ Dữ liệu phải **đại diện** cho patterns bạn muốn học
- ✅ Dữ liệu phải **nhất quán** (không có outliers quá nhiều)
- ✅ Dữ liệu phải **đủ đa dạng** (bao gồm nhiều điều kiện thị trường khác nhau)

#### 2. **Kiến Trúc Mô Hình**
- ✅ **Window size phù hợp**: 
  - Quá nhỏ (24) → không đủ context
  - Quá lớn (144) → học được noise thay vì signal
  - **Tối ưu (96)** → đủ để học patterns nhưng không quá nhiều noise
- ✅ **LSTM units phù hợp**:
  - Quá ít → không đủ capacity để học
  - Quá nhiều → overfitting
- ✅ **Dropout**: Giúp tránh overfitting

#### 3. **Hyperparameters Training**
- ✅ **Epochs**: Đủ để học nhưng không quá nhiều (tránh overfitting)
- ✅ **Learning rate**: Quá cao → không hội tụ, quá thấp → học chậm
- ✅ **Batch size**: Ảnh hưởng đến stability của training

#### 4. **Lượng Dữ Liệu**
- ✅ **Quá ít** (< 10k): Không đủ để học patterns phức tạp
- ✅ **Vừa đủ** (10k-50k): Thường cho kết quả tốt nhất
- ⚠️ **Quá nhiều** (> 100k): 
  - Có thể tốt hơn NHƯNG cần:
    - Mô hình phức tạp hơn
    - Nhiều epochs hơn
    - Regularization tốt hơn
  - Nếu không đáp ứng điều kiện trên → **overfitting** hoặc **underfitting**

#### 5. **Tính Chất Bài Toán**
- ✅ **Time series forecasting**: Cần dữ liệu theo thời gian, không phải random
- ✅ **Financial data**: Rất noisy, cần nhiều dữ liệu nhưng phải xử lý cẩn thận

---

### 💡 **KẾT LUẬN CHO CÂU HỎI 2:**

1. **Nhiều dữ liệu KHÔNG tự động = tốt hơn**
   - Phải đi kèm với mô hình và hyperparameters phù hợp
   - Phải có chất lượng tốt

2. **Từ kết quả của bạn:**
   - **30k dữ liệu + window=96** là combo tốt nhất
   - **50k dữ liệu + window=144** không tốt hơn → có thể do:
     - Window size quá lớn
     - Cần điều chỉnh hyperparameters khác

3. **Yếu tố quan trọng nhất:**
   - **Window size** (ảnh hưởng lớn nhất đến kết quả)
   - **Kiến trúc mô hình** (LSTM units, dropout)
   - **Hyperparameters training** (epochs, learning rate)
   - **Lượng dữ liệu** (quan trọng nhưng không phải yếu tố quyết định duy nhất)

---

## ❓ Câu Hỏi 3: Quy Trình Test Mô Hình Để Nộp Thầy Giáo

### 🎯 **QUY TRÌNH CHUẨN CHO BÀI TẬP:**

#### **BƯỚC 1: Chuẩn Bị Dữ Liệu** ✅
```
1. Chọn dataset: btc_15m_data_2018_to_2025.csv (280k dòng)
2. Quyết định lượng dữ liệu test:
   - Khuyến nghị: 30k dòng (dựa trên kết quả tốt nhất của bạn)
   - Hoặc: Test với nhiều lượng khác nhau để so sánh
3. Chia train/val/test: 70%/15%/15% (đã có sẵn trong code)
```

#### **BƯỚC 2: Thiết Kế Thí Nghiệm** ✅
```
1. Xác định các biến số cần test:
   - Window size: 24, 48, 96, 144 (hoặc các giá trị khác)
   - LSTM units: [16], [32,16], [64,32], [128,64,32]
   - Dropout: 0.1, 0.2, 0.3
   - Epochs: 10, 15, 25, 50
   - Learning rate: 0.0001, 0.001, 0.01

2. Tạo bảng thí nghiệm (experiment table):
   | Exp | Data | Window | LSTM | Dropout | Epochs | LR | Kết quả |
   |-----|------|--------|------|---------|--------|----|---------|
   | 1   | 30k  | 24     | [32,16] | 0.2   | 15    | 0.001 | ... |
   | 2   | 30k  | 48     | [32,16] | 0.2   | 15    | 0.001 | ... |
   | ... | ...  | ...    | ...  | ...     | ...   | ... | ... |
```

#### **BƯỚC 3: Chạy Thí Nghiệm** ✅
```
1. Sử dụng notebook: notebooks/run_complete.ipynb
2. Hoặc CLI: python -m cli.main (nếu có)
3. Với mỗi cấu hình:
   - Set PRESET_NAME hoặc cấu hình thủ công
   - Chạy từ đầu đến cuối
   - Lưu kết quả vào reports/notebook/
```

#### **BƯỚC 4: Thu Thập Kết Quả** ✅
```
Mỗi thí nghiệm sẽ tạo folder trong reports/notebook/:
- config.json: Tất cả tham số đã dùng
- metrics.json: MAE, RMSE, MAPE, Direction Accuracy
- results_*.md: Báo cáo chi tiết
- *.png: Biểu đồ training history, predictions
```

#### **BƯỚC 5: Phân Tích Kết Quả** ✅
```
1. So sánh các metrics:
   - MAE (Mean Absolute Error): Càng thấp càng tốt
   - RMSE (Root Mean Squared Error): Càng thấp càng tốt
   - MAPE (Mean Absolute Percentage Error): Càng thấp càng tốt
   - Direction Accuracy: Càng cao càng tốt (nhưng > 50% là tốt)

2. Xác định mô hình tốt nhất:
   - Cân bằng giữa các metrics
   - Không chỉ nhìn vào 1 metric

3. Phân tích tại sao mô hình này tốt:
   - Window size phù hợp?
   - LSTM units đủ?
   - Dropout hiệu quả?
   - Epochs đủ?
```

#### **BƯỚC 6: Viết Báo Cáo** ✅
```
1. Tạo file EVALUATION.md (đã có sẵn trong reports/notebook/)
2. Bao gồm:
   - Tổng quan các thí nghiệm
   - Bảng so sánh kết quả
   - Phân tích chi tiết từng mô hình
   - Kết luận và khuyến nghị
   - Giải thích tại sao mô hình tốt nhất được chọn
```

#### **BƯỚC 7: Validation (Quan Trọng!)** ✅
```
1. Test mô hình tốt nhất trên dữ liệu mới:
   - Dùng phần dữ liệu chưa từng thấy (out-of-sample)
   - Hoặc dùng timeframe khác (1h, 4h, 1d)

2. Kiểm tra tính tổng quát:
   - Mô hình có hoạt động tốt trên dữ liệu mới không?
   - Có bị overfitting không?
```

---

### 📋 **CHECKLIST ĐỂ NỘP THẦY GIÁO:**

#### **Phần 1: Code & Cấu Trúc Project** ✅
- [x] Code có tổ chức tốt (src/, notebooks/, reports/)
- [x] Có config tập trung (src/config.py)
- [x] Có documentation (docs/)
- [x] Có README.md giải thích project

#### **Phần 2: Thí Nghiệm** ✅
- [x] Đã test với nhiều cấu hình khác nhau
- [x] Có bảng so sánh kết quả (EVALUATION.md)
- [x] Có giải thích tại sao chọn mô hình này
- [x] Có biểu đồ minh họa (training history, predictions)

#### **Phần 3: Kết Quả** ✅
- [x] Metrics rõ ràng (MAE, RMSE, MAPE, Direction Accuracy)
- [x] So sánh công bằng giữa các mô hình
- [x] Phân tích chi tiết từng mô hình
- [x] Kết luận và khuyến nghị

#### **Phần 4: Báo Cáo** ✅
- [x] Có file EVALUATION.md tổng hợp
- [x] Có giải thích phương pháp
- [x] Có phân tích kết quả
- [x] Có kết luận và hướng phát triển

---

### 🎓 **ĐIỂM QUAN TRỌNG ĐỂ THẦY GIÁO ĐÁNH GIÁ CAO:**

1. **Phương Pháp Khoa Học:**
   - ✅ Test có hệ thống (nhiều cấu hình)
   - ✅ So sánh công bằng
   - ✅ Giải thích rõ ràng

2. **Phân Tích Sâu:**
   - ✅ Không chỉ báo số liệu, mà còn giải thích TẠI SAO
   - ✅ Phân tích điểm mạnh/yếu của từng mô hình
   - ✅ Đề xuất cải thiện

3. **Trình Bày Rõ Ràng:**
   - ✅ Bảng biểu dễ đọc
   - ✅ Biểu đồ minh họa tốt
   - ✅ Code có comment giải thích

4. **Tính Thực Tế:**
   - ✅ Kết quả có ý nghĩa (Direction Accuracy > 50%)
   - ✅ Metrics phù hợp với bài toán
   - ✅ Có validation trên dữ liệu mới

---

### 📝 **MẪU BÁO CÁO ĐỀ XUẤT:**

```markdown
# Báo Cáo Kết Quả Mô Hình Dự Đoán Giá Bitcoin

## 1. Mục Tiêu
- Xây dựng mô hình BiLSTM để dự đoán giá Bitcoin
- So sánh các cấu hình khác nhau
- Tìm mô hình tối ưu

## 2. Phương Pháp
- Dataset: BTC/USDT 15m (280k dòng)
- Test với: 10k, 20k, 30k, 50k dòng
- Window sizes: 24, 48, 96, 144
- Metrics: MAE, RMSE, MAPE, Direction Accuracy

## 3. Kết Quả
[Bảng so sánh các mô hình]

## 4. Phân Tích
[Mô hình tốt nhất: w96 với 30k dữ liệu]
[Lý do: MAE thấp nhất, RMSE thấp nhất, MAPE thấp nhất]

## 5. Kết Luận
[Mô hình w96 là tốt nhất]
[Hướng phát triển: thêm features, thử ensemble methods]
```

---

## 🎯 **TÓM TẮT:**

1. **Câu hỏi 1:** Cả hai cách đều có giá trị. Khuyến nghị: cố định 30k dữ liệu, sau đó tối ưu hyperparameters.

2. **Câu hỏi 2:** Nhiều dữ liệu không tự động = tốt hơn. Yếu tố quan trọng nhất là window size và kiến trúc mô hình.

3. **Câu hỏi 3:** Quy trình 7 bước đã nêu ở trên. Quan trọng nhất là có phương pháp khoa học và phân tích sâu.

---

**Chúc bạn thành công với bài nộp! 🚀**
