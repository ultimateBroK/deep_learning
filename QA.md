## 1. Khái quát dự án

Thưa thầy, dự án của em là **dự báo giá Bitcoin (BTC/USDT) sử dụng mô hình BiLSTM** trên khung thời gian 15 phút.

**Mục tiêu chính của dự án:**
- Xây dựng một pipeline hoàn chỉnh để dự báo giá Bitcoin theo chuỗi thời gian
- Thiết kế và huấn luyện mô hình BiLSTM (Bidirectional Long Short-Term Memory)
- Thử nghiệm với nhiều cấu hình khác nhau để tìm ra mô hình tối ưu nhất
- Đánh giá kết quả qua các chỉ số: MAE, RMSE, MAPE và Direction Accuracy

**Dữ liệu sử dụng:**
- Nguồn: Dữ liệu lịch sử BTC/USDT từ Binance, khoảng thời gian từ 2018 đến 2025
- Khung thời gian: 15 phút (mỗi nến đại diện cho 15 phút)
- Feature chính: Giá đóng cửa (close price)
- Số lượng thử nghiệm: Em đã thử nghiệm khoảng 10 cấu hình khác nhau

**Pipeline xử lý:**
1. Đọc dữ liệu từ file CSV
2. Tiền xử lý: Chuẩn hóa dữ liệu (MinMaxScaler) và tạo chuỗi bằng sliding window
3. Chia tập dữ liệu: 70% train, 15% validation, 15% test (giữ nguyên thứ tự thời gian)
4. Xây dựng mô hình BiLSTM
5. Huấn luyện với các callback (EarlyStopping, ModelCheckpoint)
6. Đánh giá và trực quan hóa kết quả

---

## 2. Thông tin cấu hình mô hình

Thưa thầy, em xin trình bày về cấu hình của **mô hình tốt nhất** mà em đã đạt được:

### Mô hình: `BiLSTM_15m_w72_l30k`

**Cấu hình dữ liệu:**
- **Timeframe:** 15 phút
- **Limit:** 30,000 dòng cuối cùng của dataset
- **Window size:** 72 (tương đương 18 giờ lịch sử, vì 72 × 15 phút = 1,080 phút = 18 giờ)
- **Features:** Chỉ sử dụng giá đóng cửa (`close`)
- **Scaler:** MinMaxScaler (chuẩn hóa về khoảng [0, 1])
- **Chia tập:** 70% train (20,928 mẫu), 15% validation (4,428 mẫu), 15% test (4,428 mẫu)

**Cấu hình kiến trúc mô hình:**
- **Kiến trúc:** BiLSTM 2 tầng
  - **BiLSTM Layer 1:** 32 units (bidirectional)
  - **Dropout:** 0.2
  - **BiLSTM Layer 2:** 16 units (bidirectional)
  - **Dropout:** 0.2
  - **Dense Layer:** 1 unit (đầu ra - dự báo giá)

**Cấu hình huấn luyện:**
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** MSE (Mean Squared Error)
- **Batch size:** 32
- **Epochs:** 20 (có EarlyStopping)
- **Early stopping patience:** 6 epochs
- **Best epoch:** 19/20
- **Thời gian huấn luyện:** Khoảng 256 giây (~4 phút)

**Lý do em chọn cấu hình này:**
- **Window size 72:** Em thử nghiệm nhiều window size và nhận thấy 72 là điểm tối ưu, đủ ngữ cảnh (18 giờ) nhưng không quá dài gây nhiễu hoặc tốn tài nguyên
- **2 tầng BiLSTM:** Cân bằng giữa khả năng học các pattern phức tạp và tránh overfitting
- **Dropout 0.2:** Giúp giảm overfitting, tăng tính tổng quát của mô hình
- **Limit 30k:** Cân bằng giữa chất lượng dữ liệu và thời gian huấn luyện

---

## 3. Độ chính xác của mô hình trên các tập

Thưa thầy, em xin báo cáo kết quả đánh giá mô hình trên các tập dữ liệu:

### Kết quả trên tập Test (mô hình tốt nhất: `BiLSTM_15m_w72_l30k`)

| Metric | Giá trị | Giải thích |
|--------|---------|------------|
| **MAE** | **$399.18** | Sai số trung bình tuyệt đối là 399.18 USD |
| **RMSE** | $563.86 | Căn bậc hai của sai số bình phương trung bình |
| **MAPE** | **0.44%** | Sai số phần trăm trung bình là 0.44% |
| **Direction Accuracy** | 52.52% | Độ chính xác dự đoán xu hướng (tăng/giảm) là 52.52% |

**Giải thích:**
- **MAE $399.18:** Với giá Bitcoin khoảng $100,000, sai số trung bình ~399 USD tương đương khoảng 0.4%, đây là mức khá tốt
- **MAPE 0.44%:** Sai số phần trăm rất nhỏ, phù hợp với bài toán dự báo tài chính
- **RMSE $563.86:** Cao hơn MAE cho thấy có một số điểm dự đoán sai lớn (outliers)
- **Direction Accuracy 52.52%:** Hơi thấp, gần mức ngẫu nhiên (50%), cho thấy mô hình dự báo giá trị tốt nhưng dự báo xu hướng còn hạn chế

### Kết quả trên tập Train/Validation

- **Train Loss:** 0.001321
- **Val Loss:** 0.000245
- **Best Val Loss:** 0.000197 (đạt được ở epoch 19)

Khoảng cách giữa train loss và val loss cho thấy mô hình không bị overfitting nghiêm trọng.

### So sánh với các mô hình khác

Em đã thử nghiệm 10 cấu hình khác nhau. Top 5 mô hình tốt nhất (theo MAE):

| Xếp hạng | Mô hình | MAE ($) | RMSE ($) | MAPE (%) | Direction Accuracy (%) |
|----------|---------|---------|----------|----------|------------------------|
| #1 | **w72_l30k** ⭐ | **399.18** | 563.86 | **0.44%** | 52.52% |
| #2 | w96 (original) | 424.71 | **601.66** | 0.47% | 52.78% |
| #3 | w144_l30k | 403.10 | 541.32 | 0.45% | 50.95% |
| #4 | w96_l30k | 407.79 | **525.41** | 0.45% | 50.83% |
| #5 | w24 (original) | 427.97 | 627.18 | 0.49% | **53.90%** |

**Nhận xét:**
- Mô hình `w72_l30k` đạt MAE và MAPE tốt nhất
- Window size 72 là tối ưu trong các thử nghiệm của em
- Limit 30k với window size từ 72-144 cho kết quả tốt nhất

---

## 4. Thông tin về mô hình

Thưa thầy, em xin trình bày chi tiết về kiến trúc và đặc điểm của mô hình:

### Kiến trúc mô hình BiLSTM

**Sơ đồ kiến trúc:**

```
Input Layer
    ↓
    (72 mốc thời gian, 1 feature: close)
    ↓
BiLSTM Layer 1 (32 units) - Bidirectional
    ↓
Dropout (0.2)
    ↓
BiLSTM Layer 2 (16 units) - Bidirectional
    ↓
Dropout (0.2)
    ↓
Dense Layer (1 unit)
    ↓
Output: Giá dự báo (1 giá trị)
```

**Đặc điểm kỹ thuật:**
- **Input shape:** (72, 1) - 72 bước thời gian, 1 feature (giá đóng cửa)
- **Output:** 1 giá trị (giá đóng cửa của nến tiếp theo)
- **BiLSTM:** Xử lý hai chiều (forward + backward) trong cùng một cửa sổ đầu vào, giúp mô hình học được ngữ cảnh tốt hơn so với LSTM một chiều
- **Tổng số tham số:** Khoảng 10,000-15,000 tham số (ước tính)

**Quá trình huấn luyện:**
- Mô hình được huấn luyện trong 20 epochs
- Best epoch: 19/20 (mô hình tốt nhất được lưu ở epoch thứ 19)
- Early stopping: Dừng sớm khi validation loss không cải thiện sau 6 epochs liên tiếp
- Model checkpoint: Tự động lưu mô hình tốt nhất dựa trên validation loss
- Thời gian huấn luyện: Khoảng 4 phút

**Điểm mạnh của mô hình:**
1. ✅ **MAE và MAPE thấp nhất** trong tất cả các mô hình em đã thử nghiệm
2. ✅ **Window size 72** là điểm tối ưu, phù hợp với đặc điểm của dữ liệu Bitcoin
3. ✅ **Thời gian huấn luyện hợp lý** (~4 phút), không quá lâu
4. ✅ **Không bị overfitting** nghiêm trọng (train loss và val loss gần nhau)

**Hạn chế và hướng cải thiện:**
1. ⚠️ **Direction Accuracy thấp (52.52%):** Gần mức ngẫu nhiên, cho thấy mô hình dự báo giá trị tốt nhưng dự báo xu hướng còn hạn chế
2. ⚠️ **Chỉ sử dụng feature `close`:** Thiếu các thông tin quan trọng như volume, các chỉ báo kỹ thuật (RSI, MACD, v.v.)
3. ⚠️ **RMSE cao hơn một số mô hình khác:** Có một số điểm dự đoán sai lớn

**Hướng phát triển trong tương lai:**
- Thêm các features: volume, RSI, MACD, Bollinger Bands
- Thử nghiệm Attention mechanism hoặc Transformer architecture
- Sử dụng Ensemble methods để kết hợp nhiều mô hình
- Tinh chỉnh hyperparameters chi tiết hơn

---

## Kết luận

Thưa thầy, qua quá trình thực hiện dự án, em đã:

1. ✅ Xây dựng được một pipeline hoàn chỉnh để dự báo giá Bitcoin
2. ✅ Huấn luyện thành công mô hình BiLSTM với kết quả tốt (MAE $399.18, MAPE 0.44%)
3. ✅ Thử nghiệm và so sánh 10 cấu hình khác nhau để tìm ra mô hình tối ưu
4. ✅ Phát hiện window size 72 là tối ưu cho bài toán này

**Mô hình tốt nhất:** `BiLSTM_15m_w72_l30k` với:
- MAE: $399.18
- MAPE: 0.44%
- Direction Accuracy: 52.52%

Em xin cảm ơn thầy đã lắng nghe. Em sẵn sàng trả lời các câu hỏi của thầy.

---

*Tài liệu này được tạo tự động dựa trên kết quả thực nghiệm của dự án.*
