# 🧠 Concepts (giải thích ngắn + đúng ngữ cảnh time-series)

## LSTM vs BiLSTM (điều quan trọng nhất)

- **LSTM**: đọc chuỗi theo 1 chiều (từ quá khứ → gần hiện tại).
- **BiLSTM**: chạy **2 LSTM** trên *cùng một input window*:
  - forward: trái → phải
  - backward: phải → trái

> **Lưu ý để tránh hiểu sai:** BiLSTM *không “nhìn tương lai” ngoài điểm dự đoán*. Nó chỉ tận dụng ngữ cảnh “hai phía” **bên trong window đầu vào**. Nếu bạn làm streaming real-time từng tick, BiLSTM thường không phù hợp trừ khi bạn chấp nhận buffer đủ window rồi mới dự đoán.

## Sliding window (window size)

Bạn biến time-series thành nhiều mẫu dạng:

- **Input**: \(x_{t-window+1}, ..., x_t\)
- **Target**: \(x_{t+1}\)

Với timeframe 15m:

- window=96 ≈ 1 ngày (96 * 15 phút)
- window=144 ≈ 1.5 ngày
- window=240 ≈ 2.5 ngày

## Scaling (chuẩn hoá)

Giá BTC rất lớn (10k–100k+). Scaling giúp model học ổn định hơn.

- Thường dùng: **MinMaxScaler** đưa về \([0, 1]\)

## Split train/val/test (time-series)

- **Không shuffle** khi split (phải giữ thứ tự thời gian).
- Thường: 70% train / 15% val / 15% test.

## Loss vs Metrics (training vs đánh giá)

- **Loss (MSE)**: dùng để tối ưu trong training (phạt mạnh lỗi lớn).
- **MAE (USD)**: “trung bình sai bao nhiêu USD” (dễ hiểu).
- **RMSE (USD)**: nhấn mạnh outliers (sai lớn bị phạt nặng).
- **MAPE (%)**: sai số theo % (cẩn thận khi \(y\) gần 0).
- **Direction Accuracy**: đúng hướng tăng/giảm (trading hay nhìn chỉ số này).

> Trong log training, MAE/Loss thường ở **thang scaled**. Khi evaluate, code sẽ inverse-transform để ra **USD**.

## Overfitting / Underfitting (đọc train vs val)

- **Overfitting**: train loss ↓ nhưng val loss ↑ → model “học vẹt”.
- **Underfitting**: cả train & val đều cao → model chưa học được pattern.

## Callbacks hay gặp

- **ModelCheckpoint**: lưu model tốt nhất theo val_loss.
- **EarlyStopping**: dừng khi val_loss không cải thiện.
- **ReduceLROnPlateau**: giảm learning rate khi bị “kẹt”.

