# Sơ đồ luồng xử lý (flow)

Mô tả luồng xử lý chính của project.

---

## Mục lục

- [Luồng tổng quan](#flow-tổng-quan)
- [Luồng chi tiết từng bước](#flow-chi-tiết-từng-bước)
- [Luồng dữ liệu](#data-flow)
- [Luồng training](#training-flow)
- [Luồng đánh giá](#evaluation-flow)

---

## Flow Tổng Quan

```
┌─────────────────────────────────────────────────────────────────┐
│                        START                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BƯỚC 1: ĐỌC DỮ LIỆU CSV (LOCAL)                                 │
│  • Đọc data từ file CSV trong thư mục data/                      │
│  • Cache (CSV đã chuẩn hoá) vào data/cache/ (optional)           │
│  • Trả về DataFrame với: datetime, open, high, low, close, vol  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BƯỚC 2: XỬ LÝ DỮ LIỆU                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2.1: Scaling (MinMaxScaler)                             │    │
│  │     • Đưa data về [0, 1]                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2.2: Sliding Window                                     │    │
│  │     • Tạo sequences (window_size=240 default 15m)       │    │
│  │     • Input: 60 ngày → Output: ngày 61                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2.3: Split Data                                         │    │
│  │     • Train (70%), Val (15%), Test (15%)                │    │
│  │     • KHÔNG shuffle (time series!)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BƯỚC 3: XÂY DỰNG MODEL BiLSTM                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input Layer                                             │    │
│  │    shape=(60, 1) = (window_size, n_features)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Bidirectional LSTM Layer 1                              │    │
│  │    • 64 units, return_sequences=True                    │    │
│  │    • Dropout (0.2)                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Bidirectional LSTM Layer 2                              │    │
│  │    • 32 units, return_sequences=False                   │    │
│  │    • Dropout (0.2)                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Dense Layer                                             │    │
│  │    • 16 units, activation='relu'                        │    │
│  │    • Dropout (0.1)                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Output Layer                                            │    │
│  │    • 1 unit (dự đoán giá)                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│  • Compile: optimizer=Adam, loss=MSE, metrics=[MAE]             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BƯỚC 4: TRAINING MODEL                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Callbacks:                                              │    │
│  │    • ModelCheckpoint: Lưu model tốt nhất                │    │
│  │    • EarlyStopping: Dừng nếu val_loss không giảm        │    │
│  │    • ReduceLROnPlateau: Giảm LR nếu không cải thiện     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  • Epochs: 30 (hoặc EarlyStopping)                              │
│  • Batch size: 32                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BƯỚC 5: ĐÁNH GIÁ & VẼ BIỂU ĐỒ                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Evaluate Model                                          │    │
│  │    • Dự đoán trên Test set                              │    │
│  │    • Tính metrics: MAE, RMSE, MAPE                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Visualize                                               │    │
│  │    • Training History (loss, val_loss, mae, val_mae)    │    │
│  │    • Predictions vs Actual                              │    │
│  │    • Residuals                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LƯU KẾT QUẢ                                                    │
│  • Báo cáo Markdown (metrics, config, training history)         │
│  • Biểu đồ PNG (training history, predictions, residuals)       │
│  • Config JSON                                                  │
│  • Metrics JSON                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        END                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flow Chi Tiết Từng Bước

### BƯỚC 1: ĐỌC DỮ LIỆU CSV (LOCAL)

```
fetch_binance_data()
        │
        ▼
┌──────────────────┐
│ Kiểm tra cache   │
│   (CSV tồn tại?) │
└──────┬───────────┘
       │
   ┌───┴─────────────┐
   │ Có              │ Không
   ▼                 ▼       
┌──────────────┐   ┌──────────────────────┐
│ Đọc cache    │   │ Đọc CSV gốc (data/)  │
│ normalized   │   └──────────┬───────────┘
└──────┬───────┘              ▼
       │              ┌──────────────────────┐
       │              │ Chuẩn hoá cột        │
       │              │ (Open time→datetime, │
       │              │  Open/High/... )     │
       │              └──────────┬───────────┘
       │                         ▼
       │              ┌──────────────────────┐
       │              │ Lấy N dòng cuối      │
       │              │ (limit, optional)    │
       │              └──────────┬───────────┘
       │                         ▼
       │              ┌──────────────────────┐
       │              │ Lưu cache normalized │
       │              └──────────┬───────────┘
       │                         │
       ▼                         ▼
              ┌───────────┐
              │ Trả về df │
              └───────────┘
```

### BƯỚC 2: XỬ LÝ DỮ LIỆU

```
prepare_data_for_lstm()
        │
        ▼
┌──────────────────┐
│ Chọn features    │
│ (close, open...) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Scaling          │
│ (MinMaxScaler)   │
│ [0, 1]           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Sliding Window   │
│ (window_size=240) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Split Data       │
│ 70% train        │
│ 15% val          │
│ 15% test         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Trả về dict:     │
│ X_train, y_train │
│ X_val, y_val     │
│ X_test, y_test   │
│ scaler, data...  │
└──────────────────┘
```

### BƯỚC 3: XÂY DỰNG MODEL

```
build_bilstm_model()
        │
        ▼
┌──────────────────┐
│ Tạo Sequential   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Input Layer      │
│ (window_size, n_features) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Bidirectional    │
│ LSTM (64)        │
│ + Dropout (0.2)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Bidirectional    │
│ LSTM (32)        │
│ + Dropout (0.2)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Dense (16)       │
│ + Dropout (0.1)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output (1)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Compile:         │
│ Adam, MSE, MAE   │
└────────┬─────────┘
         │
         ▼
  ┌──────────┐
  │ Return   │
  │ model    │
  └──────────┘
```

### BƯỚC 4: TRAINING

```
train_model()
        │
        ▼
┌──────────────────┐
│ Setup Callbacks  │
│ • Checkpoint     │
│ • EarlyStop      │
│ • ReduceLR       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Loop Epochs      │
│   for epoch 1..N │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Loop Batches     │
│   for batch 1..M │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Forward Pass     │
│ (Dự đoán)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tính Loss        │
│ (MSE, MAE)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Backward Pass    │
│ (Cập nhật)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Validate         │
│ (trên val set)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Callbacks check  │
│ • Lưu model tốt  │
│ • Early stop?    │
│ • Giảm LR?       │
└────────┬─────────┘
         │
         ▼
    ┌─────────┐
    │ Return  │
    │ history │
    └─────────┘
```

### BƯỚC 5: ĐÁNH GIÁ & VẼ BIỂU ĐỒ

```
evaluate_model() + plot_all()
         │
         ▼
┌──────────────────┐
│ model.predict()  │
│ (trên test set)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Inverse Transform│
│ (giá trị gốc)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tính Metrics     │
│ • MAE            │
│ • RMSE           │
│ • MAPE           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Vẽ Biểu Đồ       │
│ • Training Hist  │
│ • Predictions    │
│ • Residuals      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Lưu Kết Quả      │
│ • Markdown       │
│ • PNG            │
│ • JSON           │
└────────┬─────────┘
         │
         ▼
     ┌──────┐
     │ DONE │
     └──────┘
```

---

## Data Flow

```
Raw Data (CSV local)
        │
        ▼
   DataFrame
   [50000 rows, 6 cols] (default: 15m)
        │
        ▼
   Selected Features
   [50000 rows, 1 col] (close)
        │
        ▼
   Scaled Data
   [50000, 1] (0-1)
        │
        ▼
   Sliding Window
   window_size=240 (default: 15m)
        │
        ▼
   X: [49760, 240, 1]   y: [49760, 1]
        │
        ▼
   ┌──────┴──────┐
   │             │
   ▼             ▼
Train        Split
   │     80/10/10
   │             │
   ▼             ▼
X_train      X_test
[1152,60,1] [144,60,1]
   │
   ▼
Model Forward
   │
   ▼
y_pred_scaled
[144, 1]
   │
   ▼
Inverse Transform
   │
   ▼
y_pred (USD)
[144, 1]
```

---

## Training Flow

```
Epoch 1
│
├─ Train Loss: 0.00500
├─ Val Loss:   0.00520
│
Epoch 2
│
├─ Train Loss: 0.00300  ↓
├─ Val Loss:   0.00350  ↓
│
...
│
Epoch 10
│
├─ Train Loss: 0.00100  ↓
├─ Val Loss:   0.00200  ↑ (start overfitting?)
│
EarlyStopping detected!
Best epoch: 5
Best val_loss: 0.00150
```

---

## Tóm tắt

Quy trình tổng thể:

```
Fetch Data → Preprocess → Build Model → Train → Evaluate → Save
```

Quan trọng:
1. Data chia theo thời gian (KHÔNG shuffle!)
2. Train để học, Val để điều chỉnh, Test để đánh giá
3. EarlyStopping giúp tránh overfitting
4. Metrics đánh giá chất lượng model


