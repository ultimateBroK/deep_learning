# Hướng dẫn xử lý sự cố

Tài liệu này mô tả các bước chạy chính và cách xử lý các lỗi thường gặp.

---

## Mục lục

- [Bước 1: Đọc dữ liệu từ CSV](#bước-1-đọc-dữ-liệu-từ-csv-local)
- [Bước 2: Xử lý dữ liệu](#bước-2-xử-lý-dữ-liệu)
- [Bước 3: Xây dựng model BiLSTM](#bước-3-xây-dựng-model-bilstm)
- [Bước 4: Training model](#bước-4-training-model)
- [Bước 5: Đánh giá & Vẽ biểu đồ](#bước-5-đánh-giá--vẽ-biểu-đồ)
- [Xử lý sự cố](#xử-lý-sự-cố)

---

## Bước 1: Đọc dữ liệu từ CSV

### Giải thích
- **fetch_binance_data()**: giữ tên cũ để tương thích; thực tế là đọc file CSV
- **Dữ liệu mặc định**: `data/btc_15m_data_2018_to_2025.csv` (tập trung 15m)
- **Cache**: Lưu CSV đã chuẩn hoá (datetime/open/high/low/close/volume) vào `data/cache/` để lần sau đọc nhanh hơn
- **Timeframe**: Chỉ dùng để chọn file mặc định nếu không set `data_path` (15m/1h/4h/1d)

### Các tham số
| Tham số | Giải thích | Mặc định |
|---------|------------|----------|
| `data_path` | Đường dẫn CSV | data/btc_15m_data_2018_to_2025.csv |
| `timeframe` | Dùng để chọn file mặc định | 15m |
| `limit` | Lấy N dòng cuối của CSV | 50000 (cho 15m) |
| `save_cache` | Có lưu cache không | True |

### Dữ liệu trả về
DataFrame với các cột:
- `datetime`: Thời gian
- `open`: Giá mở nến
- `high`: Giá cao nhất
- `low`: Giá thấp nhất
- `close`: Giá đóng nến
- `volume`: Khối lượng giao dịch

### Ví dụ code
```python
from src.core import fetch_binance_data

df = fetch_binance_data(
    data_path="data/btc_15m_data_2018_to_2025.csv",
    timeframe="15m",
    limit=50000
)

print(df.head())
```

---

## Bước 2: Xử Lý Dữ Liệu

### 2.1: Scaling (Chuẩn hóa dữ liệu)

#### Giải thích
- **Tại sao cần scaling?** 
  - Giá Bitcoin dao động từ $10,000 đến $100,000
  - Số quá lớn khiến model khó học
  - Scaling đưa tất cả về khoảng [0, 1] hoặc [-1, 1]

- **MinMaxScaler**: 
  - Đưa dữ liệu về khoảng [0, 1]
  - Công thức: (x - min) / (max - min)
  - Ví dụ: $50,000 (trong range $10K-$100K) → (50K-10K)/(100K-10K) = 0.44

- **StandardScaler**:
  - Đưa dữ liệu về mean=0, std=1
  - Công thức: (x - mean) / std
  - Tốt khi data có phân phối Gaussian

#### Dùng scaler nào?
- Với giá crypto: Dùng **MinMaxScaler** (giá luôn > 0, ta biết min/max)
- Với data có outliers nhiều: Dùng **StandardScaler**

### 2.2: Sliding Window (Tạo sequences)

#### Giải thích
- **Tại sao cần sliding window?**
  - LSTM học từ sequences (chuỗi) chứ không phải từng điểm đơn lẻ
  - Để dự đoán giá ngày mai, cần nhìn giá của 60 ngày trước đó

- **Window size**: Số bước nhìn lại (past days)
  - Window size = 60: Model nhìn 60 ngày trước để dự đoán ngày tiếp theo
  - Window size = 30: Model nhìn 30 ngày trước

#### Ví dụ
```
Dữ liệu: [10000, 15000, 20000, 25000, 30000, 35000, 40000]
Window size = 3

Sample 1: Input: [10000, 15000, 20000] → Output: 25000
Sample 2: Input: [15000, 20000, 25000] → Output: 30000
Sample 3: Input: [20000, 25000, 30000] → Output: 35000
...
```

### 2.3: Split Data (Chia train/val/test)

#### Giải thích
- **Train (70%)**: Dùng để huấn luyện model
- **Validation (15%)**: Dùng để điều chỉnh hyperparameters, dừng training sớm
- **Test (15%)**: Dùng để đánh giá cuối cùng (chỉ dùng 1 lần!)

#### Tại sao phải chia?
- Nếu train và test trên cùng dữ liệu → model "vẹt", không thực chiến được
- Test set như "đề thi cuối kỳ" - model chưa bao giờ thấy

#### Lưu ý quan trọng
- KHÔNG shuffle data khi split (vì là time series!)
- Phải giữ nguyên thứ tự thời gian

### Ví dụ code
```python
from src.core import prepare_data_for_lstm

data_dict = prepare_data_for_lstm(
    df=df,
    features=["close"],
    window_size=60,
    scaler_type="minmax"
)

X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_val = data_dict['X_val']
y_val = data_dict['y_val']
X_test = data_dict['X_test']
y_test = data_dict['y_test']
```

---

## Bước 3: Xây Dựng Model BiLSTM

### Giải thích
- **LSTM (Long Short-Term Memory)**:
  - RNN cải tiến có khả năng ghi nhớ thông tin dài hạn
  - Giải quyết vanishing gradient problem của RNN thường

- **BiLSTM (Bidirectional LSTM)**:
  - Chạy 2 chiều trên *cùng một input window* (forward + backward) để tận dụng ngữ cảnh tốt hơn
  - 2 LSTM: 1 đọc từ trái → phải, 1 đọc từ phải → trái
  - Lưu ý: Không “nhìn tương lai” ngoài điểm dự đoán; chỉ là xử lý hai chiều bên trong window đầu vào

- **Dropout**:
  - Bỏ ngẫu nhiên một số neurons trong quá trình training
  - Giúp tránh overfitting (model học vẹt)

- **Dense layers**:
  - Layers kết nối đầy đủ
  - Kết hợp các features đã học được từ LSTM để đưa ra dự đoán

### Các tham số quan trọng
| Tham số | Giải thích | Mặc định | Khuyến nghị |
|---------|------------|----------|-------------|
| `lstm_units` | Số neurons trong mỗi LSTM layer | [64, 32] | 32-128 |
| `dropout_rate` | Tỷ lệ bỏ neurons | 0.2 | 0.1-0.5 |
| `dense_units` | Số neurons trong Dense layers | [16] | 8-64 |

### Cách chọn số layers & units?
- **Data ít (< 1000 samples)**: 1-2 LSTM layers, 16-32 units
- **Data vừa (1000-10000 samples)**: 2-3 LSTM layers, 32-64 units
- **Data nhiều (> 10000 samples)**: 3+ LSTM layers, 64-128 units

### Ví dụ code
```python
from src.core import build_bilstm_model

input_shape = (60, 1)  # (window_size, n_features)
model = build_bilstm_model(
    input_shape=input_shape,
    lstm_units=[64, 32],
    dropout_rate=0.2,
    dense_units=[16],
    output_units=1
)
```

---

## Bước 4: Training Model

### Giải thích
- **Epochs**: Số lần model học qua toàn bộ dữ liệu
  - Epoch 1: Model học lần đầu, chưa hiểu nhiều
  - Epoch 2: Model học lại, hiểu rõ hơn
  - Epoch 20: Model đã hiểu tốt pattern của dữ liệu

- **Batch size**: Số samples mỗi lần tính gradient
  - Batch size nhỏ → train chậm nhưng chính xác hơn
  - Batch size lớn → train nhanh nhưng có thể kém chính xác hơn

- **Learning rate**: Bước nhảy khi cập nhật weights
  - LR lớn → học nhanh nhưng có thể "nhảy qua" optimum
  - LR nhỏ → học chậm nhưng chính xác hơn

### Callbacks là gì?

#### 1. ModelCheckpoint
- Lưu lại model tốt nhất (có val_loss thấp nhất)
- Về sau có thể load lại mà không cần train lại

#### 2. EarlyStopping
- Dừng training khi val_loss không giảm sau N epochs
- Tiết kiệm thời gian, tránh overfitting

#### 3. ReduceLROnPlateau
- Giảm learning rate khi val_loss không giảm
- Giúp model "fine-tune" tốt hơn

### Ví dụ code
```python
from src.training import train_model
from src import Config

# Tạo Config object
config = Config()
config.training.epochs = 20
config.training.batch_size = 32
config.training.early_stopping_patience = 5

train_result = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    config=config
)

history = train_result['history']
```

---

## Bước 5: Đánh Giá & Vẽ Biểu Đồ

### Metrics là gì?

#### MAE (Mean Absolute Error)
- **Công thức**: mean(|y_true - y_pred|)
- **Giải thích**: Sai số trung bình tuyệt đối
- **Ví dụ**: MAE = $500 → Model dự đoán sai trung bình $500
- **Ưu điểm**: Dễ hiểu

#### RMSE (Root Mean Squared Error)
- **Công thức**: sqrt(mean((y_true - y_pred)²))
- **Giải thích**: Căn bậc 2 của sai số bình phương trung bình
- **Ưu điểm**: Nhấn mạnh vào các lỗi lớn (outliers)

#### MAPE (Mean Absolute Percentage Error)
- **Công thức**: mean(|y_true - y_pred| / y_true) * 100
- **Giải thích**: Sai số phần trăm trung bình
- **Ví dụ**: MAPE = 2% → Model sai trung bình 2%
- **Ưu điểm**: Độc lập với scale của giá

### Đánh giá: Kết quả tốt hay xấu?

| MAE | RMSE | MAPE | Đánh giá |
|-----|------|------|----------|
| < $200 | < $300 | < 1% | Tuyệt vời |
| $200-$500 | $300-$800 | 1-2% | Tốt |
| $500-$1000 | $800-$1500 | 2-5% | Trung bình |
| > $1000 | > $1500 | > 5% | Kém |

### Ví dụ code
```python
from src.core import evaluate_model, print_sample_predictions

eval_result = evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    scaler=scaler,
    return_predictions=True
)

y_true = eval_result['y_true']
y_pred = eval_result['predictions']

print_sample_predictions(y_true, y_pred, n_samples=10)
```

---

## Xử lý sự cố

### Lỗi 1: "No module named 'tensorflow'" (hoặc polars/numpy)

**Nguyên nhân**: Chưa cài dependencies

**Giải pháp**:
```bash
uv sync
```

---

### Lỗi 2: "FileNotFoundError: Không tìm thấy file data"

**Nguyên nhân**: `--data-path` trỏ sai, hoặc bạn chưa có file CSV trong `data/`

**Giải pháp**:
- Kiểm tra file mặc định: `data/btc_1d_data_2018_to_2025.csv`
- Hoặc chỉ định rõ:

```bash
uv run python -m cli.main --data-path data/btc_1d_data_2018_to_2025.csv
```

---

### Lỗi 3: "CSV thiếu cột bắt buộc"

**Nguyên nhân**: File CSV không đúng format (cần có các cột kiểu Binance export: `Open time`, `Open`, `High`, `Low`, `Close`, `Volume`)

**Giải pháp**:
- Dùng đúng file mặc định trong `data/`
- Hoặc sửa header CSV cho khớp các cột trên

---

### Lỗi 4: Overfitting (Train loss thấp, Val loss cao)

**Nguyên nhân**: Model học vẹt data training

**Giải pháp**:
1. Tăng dropout rate (0.2 → 0.3, 0.4)
2. Giảm số units trong LSTM
3. Giảm số epochs
4. Tăng data training

---

### Lỗi 5: Underfitting (Cả train và val loss đều cao)

**Nguyên nhân**: Model quá đơn giản, không học được pattern

**Giải pháp**:
1. Tăng số LSTM layers
2. Tăng số units
3. Tăng số epochs
4. Thêm các features khác (volume, open, high, low)

---

### Lỗi 6: Model không converge (loss dao động)

**Nguyên nhân**: Learning rate quá lớn

**Giải pháp**:
1. Giảm learning rate (mặc định 0.001 → 0.0005, 0.0001)
2. Sử dụng ReduceLROnPlateau callback (đã bật mặc định)

---

### Lỗi 7: Out of Memory (OOM)

**Nguyên nhân**: Batch size quá lớn hoặc data quá nhiều

**Giải pháp**:
1. Giảm batch size (32 → 16, 8)
2. Giảm window size
3. Giảm số units trong LSTM

---

### Lỗi 8: Kết quả dự đoán rất kém

**Nguyên nhân có thể**:
1. Data quá ít (< 10000 samples cho 15m)
2. Window size không phù hợp
3. Model quá phức tạp so với data
4. Market đang volatile (dự đoán giá crypto rất khó!)

**Giải pháp**:
1. Tăng limit (10000 → 50000 cho 15m)
2. Thử preset khác (scalping-ultra-fast → intraday-balanced)
3. Thử window size khác (24 → 96 → 240)
4. Nhận rằng dự đoán giá crypto là vấn đề rất khó!

---

### Lỗi 9: GPU không được sử dụng

**Nguyên nhân**: TensorFlow không tìm thấy GPU

**Giải pháp**:
- Với CPU AMD: Đây là bình thường, project đã tối ưu cho CPU
- Nếu có NVIDIA GPU: Cài CUDA, cuDNN

---

## Gợi ý

1. **Luôn dùng cache**: Đừng tải lại data mỗi lần chạy
2. **Bắt đầu với config đơn giản**: 1-2 LSTM layers, 32-64 units
3. **Theo dõi val_loss**: Không chỉ train loss!
4. **Dùng EarlyStopping**: Tiết kiệm thời gian
5. **Test trên data thật**: Không chỉ nhìn train/val metrics

---

## Kết luận

Nếu gặp vấn đề:
1. Đọc lại phần liên quan trong file này
2. Tham khảo `docs/CONCEPTS.md` nếu cần giải thích thuật ngữ
3. Tìm lỗi trong mục “Xử lý sự cố”

