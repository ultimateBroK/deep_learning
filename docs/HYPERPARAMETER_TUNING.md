# 🎯 Hyperparameter Tuning Guide - Tìm Kết Quả Tốt Nhất

Tài liệu này cung cấp danh sách các lệnh với các tham số khác nhau để tìm ra cấu hình tốt nhất cho model BiLSTM dự đoán giá Bitcoin.

> **[!IMPORTANT]**
> Project tập trung vào **15m timeframe** với data khổng lồ (~280K dòng).

---

## 📋 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Danh Sách Presets (Khuyến Nghị)](#danh-sách-presets-khuyến-nghị)
3. [Các Tham Số Quan Trọng](#các-tham-số-quan-trọng)
4. [Chiến Lược Tuning](#chiến-lược-tuning)
5. [Danh Sách Lệnh (Manual Tuning)](#danh-sách-lệnh-manual-tuning)
6. [So Sánh Kết Quả](#so-sánh-kết-quả)

---

## 🎯 Tổng Quan

### Mục Tiêu
Tìm ra cấu hình tốt nhất bằng cách thử nghiệm các tổ hợp tham số khác nhau và so sánh kết quả (MAE, RMSE, MAPE, Direction Accuracy).

### Quy Trình
1. **Chọn preset phù hợp** (xem bảng bên dưới)
2. **Chạy experiment** với preset hoặc tham số tùy chỉnh
3. **Lưu kết quả** vào `reports/cli/` (tên folder tự động chứa timeframe và window_size)
4. **So sánh metrics** từ các file `results_*.md`
5. **Chọn cấu hình tốt nhất** dựa trên mục tiêu (MAE thấp nhất, Direction Accuracy cao nhất, v.v.)

---

## 📦 Danh Sách Presets (Khuyến Nghị)

> **Tip:** Sử dụng preset có sẵn thay vì cấu hình thủ công - được tối ưu hóa cho từng mục đích!

| Preset | Limit | Window | Epochs | Mục đích | Khuyến nghị |
|--------|-------|--------|--------|----------|-------------|
| **Scalping** (Siêu ngắn hạn) |
| `scalping-ultra-fast` | 10K | 24 (6h) | 5 | Scalping cực nhanh | Test nhanh |
| `scalping-fast` | 20K | 48 (12h) | 10 | Scalping nhanh | Scalping 15m |
| **Intraday** (Ngắn hạn) |
| `intraday-light` | 30K | 96 (1 ngày) | 15 | Intraday nhẹ | Intraday 15m |
| `intraday-balanced` | 50K | 144 (1.5 ngày) | 25 | Intraday cân bằng | **Khuyến nghị** |
| **Swing** (Trung hạn) |
| `swing-fast` | 70K | 240 (2.5 ngày) | 30 | Swing nhanh | Swing 15m |
| `swing-balanced` | 100K | 384 (4 ngày) | 50 | Swing cân bằng | Swing 15m |
| **Long-term** (Dài hạn) |
| `long-term` | 150K | 576 (6 ngày) | 80 | Dự đoán dài hạn | Long-term 15m |
| **Production** (Chất lượng cao) |
| `production` | 200K | 768 (8 ngày) | 100 | Production tốt nhất | Production 15m |
| **Legacy** (Other timeframes) |
| `default` | 50K | 240 (2.5 ngày) | 30 | Default (15m) | Default config |
| `fast` | 20K | 48 (12h) | 10 | Test nhanh (15m) | Test nhanh |
| `1h-light` | 10K | 48 (2 ngày) | 15 | Test (1h) | Test 1h |
| `4h-balanced` | 2K | 24 (4 ngày) | 30 | Test (4h) | Test 4h |
| **30k Dataset** (15m - fixed dataset 30k để so sánh window) |
| `30k-w24` | 30K | 24 (6h) | 15 | Ngắn hạn cực nhanh | So sánh window |
| `30k-w48` | 30K | 48 (12h) | 15 | Ngắn hạn nhanh | So sánh window |
| `30k-w72` | 30K | 72 (18h) | 20 | Ngắn hạn | So sánh window |
| `30k-w96` | 30K | 96 (1 ngày) | 20 | Ngắn hạn cân bằng | So sánh window |
| `30k-w144` | 30K | 144 (1.5 ngày) | 25 | Trung hạn ngắn | So sánh window |
| `30k-w192` | 30K | 192 (2 ngày) | 25 | Trung hạn | So sánh window |
| `30k-w240` | 30K | 240 (2.5 ngày) | 30 | Trung hạn cân bằng | So sánh window |
| `30k-w336` | 30K | 336 (3.5 ngày) | 30 | Trung hạn dài | So sánh window |
| `30k-w480` | 30K | 480 (5 ngày) | 40 | Dài hạn ngắn | So sánh window |
| `30k-w672` | 30K | 672 (7 ngày) | 40 | Dài hạn | So sánh window |

**Cách dùng presets:**
```bash
# Scalping cực nhanh (6h)
python -m cli.main --preset scalping-ultra-fast

# Intraday cân bằng (1.5 ngày) - Khuyến nghị
python -m cli.main --preset intraday-balanced

# Production chất lượng cao (8 ngày)
python -m cli.main --preset production

# Test với 1h timeframe
python -m cli.main --preset 1h-light
```

---

## 📊 Các Tham Số Quan Trọng

### 1. **Timeframe** (`--timeframe`)
- **Ảnh hưởng**: Độ phân giải dữ liệu
- **Giá trị**: `15m`, `1h`, `4h`, `1d` (mặc định: `15m`)
- **Khuyến nghị**:
  - `15m`: Tập trung chính, nhiều dữ liệu (~280K dòng), phù hợp cho scalping/intraday
  - `1h`: Dữ liệu trung bình, phù hợp cho swing trading
  - `4h`: Dữ liệu ít hơn, phù hợp cho swing dài hạn
  - `1d`: Dữ liệu ít nhất, phù hợp cho dự đoán dài hạn, ít noise

### 2. **Window Size** (`--window`)
- **Ảnh hưởng**: Số nến nhìn lại để dự đoán
- **Giá trị**: 24-768 (tùy timeframe)
- **Khuyến nghị**:
  - Nhỏ (24-48): Phản ứng nhanh với thay đổi gần đây (scalping)
  - Trung bình (96-240): Cân bằng giữa ngắn hạn và dài hạn (intraday)
  - Lớn (384-768): Tập trung vào xu hướng dài hạn (swing/long-term)

### 3. **LSTM Units** (`--lstm-units`)
- **Ảnh hưởng**: Độ phức tạp và khả năng học của model
- **Giá trị**: List các số nguyên, ví dụ: `64 32` hoặc `128 64 32`
- **Khuyến nghị**:
  - Nhỏ (`16` hoặc `32 16`): Nhanh, ít overfitting, phù hợp scalping
  - Trung bình (`64 32`): Cân bằng tốt, khuyến nghị cho intraday
  - Lớn (`128 64 32` hoặc `256 128 64 32`): Mạnh hơn nhưng dễ overfitting, cần nhiều dữ liệu (swing/long-term)

### 4. **Dropout Rate** (`--dropout`)
- **Ảnh hưởng**: Giảm overfitting
- **Giá trị**: 0.0 - 0.5 (thường dùng: 0.2)
- **Khuyến nghị**:
  - Thấp (0.1-0.2): Khi model chưa overfit
  - Cao (0.3-0.5): Khi model bị overfit (val_loss >> train_loss)

### 5. **Epochs** (`--epochs`)
- **Ảnh hưởng**: Số lần học qua toàn bộ dữ liệu
- **Giá trị**: 5-100 (thường dùng: 10-50)
- **Khuyến nghị**:
  - Ít (5-15): Nhanh, phù hợp scalping/test
  - Trung bình (25-50): Khuyến nghị cho intraday
  - Nhiều (80-100): Cho kết quả tốt hơn nhưng lâu hơn (swing/long-term)

### 6. **Batch Size** (`--batch-size`)
- **Ảnh hưởng**: Kích thước batch trong training
- **Giá trị**: 16, 32, 64, 128
- **Khuyến nghị**:
  - Nhỏ (16-32): Gradient update thường xuyên hơn, ổn định hơn
  - Lớn (64-128): Nhanh hơn nhưng có thể kém ổn định

### 7. **Limit** (`--limit`)
- **Ảnh hưởng**: Số lượng dữ liệu sử dụng
- **Giá trị**: 10K-200K (mặc định: `50000` cho 15m)
- **Khuyến nghị**:
  - Ít (10K-20K): Nhanh, phù hợp test/scalping
  - Trung bình (50K-70K): Khuyến nghị cho intraday
  - Nhiều (100K-200K): Cho kết quả tốt nhất nhưng lâu hơn (swing/long-term)

---

## 🔬 Chiến Lược Tuning

### Chiến Lược 1: Grid Search (Tìm Toàn Diện)
Thử tất cả tổ hợp tham số trong một phạm vi nhất định.

- **Ưu điểm**: Tìm được cấu hình tốt nhất
- **Nhược điểm**: Tốn thời gian

### Chiến Lược 2: Random Search (Ngẫu Nhiên)
Thử ngẫu nhiên các tổ hợp tham số.

- **Ưu điểm**: Nhanh hơn, có thể tìm được điểm tốt bất ngờ
- **Nhược điểm**: Có thể bỏ sót một số tổ hợp tốt

### Chiến Lược 3: Manual Tuning (Thủ Công)
Dựa vào kinh nghiệm và kết quả trước đó để điều chỉnh từng tham số.

- **Ưu điểm**: Kiểm soát được, hiểu rõ ảnh hưởng của từng tham số
- **Nhược điểm**: Cần kinh nghiệm

---

## 📝 Danh Sách Lệnh

### 🚀 Phase 1: Baseline - Thiết Lập Điểm Chuẩn

Chạy các lệnh này để có baseline để so sánh:

```bash
# Baseline 1: Config mặc định
uv run python -m cli.main

# Baseline 2: Preset fast (nhanh, test)
uv run python -m cli.main --preset fast

# Baseline 3: Preset intraday-balanced (khuyến nghị)
uv run python -m cli.main --preset intraday-balanced
```

---

### 🔍 Phase 2: Tuning Timeframe

So sánh giữa timeframe 1d và 4h (lưu ý: project mặc định/preset default tập trung 15m):

```bash
# Timeframe 1d
uv run python -m cli.main --timeframe 1d --window 60 --epochs 20

# Timeframe 4h (nhiều dữ liệu hơn)
uv run python -m cli.main --timeframe 4h --window 60 --epochs 20
```

---

### 📏 Phase 3: Tuning Window Size

Thử các window size khác nhau (gợi ý window “đúng ngữ cảnh” cho 15m):

```bash
# Window nhỏ - Scalping (15m)
uv run python -m cli.main --window 24 --epochs 10
uv run python -m cli.main --window 48 --epochs 10

# Window trung bình - Intraday (15m)
uv run python -m cli.main --window 96 --epochs 20
uv run python -m cli.main --window 144 --epochs 25

# Window lớn - Swing/Longer (15m)
uv run python -m cli.main --window 240 --epochs 30
uv run python -m cli.main --window 384 --epochs 50
```

---

### 🧠 Phase 4: Tuning Model Architecture (LSTM Units)

Thử các kiến trúc model khác nhau:

```bash
# Model nhỏ - Nhanh, ít overfitting
uv run python -m cli.main --lstm-units 32 16 --epochs 20
uv run python -m cli.main --lstm-units 48 24 --epochs 20

# Model trung bình - Cân bằng (mặc định)
uv run python -m cli.main --lstm-units 64 32 --epochs 20
uv run python -m cli.main --lstm-units 64 32 16 --epochs 20

# Model lớn - Mạnh hơn
uv run python -m cli.main --lstm-units 128 64 --epochs 20
uv run python -m cli.main --lstm-units 128 64 32 --epochs 20
uv run python -m cli.main --lstm-units 256 128 64 --epochs 30
```

---

### 🎚️ Phase 5: Tuning Dropout Rate

Điều chỉnh dropout để giảm overfitting:

```bash
# Dropout thấp - Khi model chưa overfit
uv run python -m cli.main --dropout 0.1 --epochs 20
uv run python -m cli.main --dropout 0.15 --epochs 20

# Dropout trung bình - Mặc định
uv run python -m cli.main --dropout 0.2 --epochs 20

# Dropout cao - Khi model bị overfit
uv run python -m cli.main --dropout 0.3 --epochs 20
uv run python -m cli.main --dropout 0.4 --epochs 20
uv run python -m cli.main --dropout 0.5 --epochs 20
```

---

### ⏱️ Phase 6: Tuning Training Parameters

Điều chỉnh epochs và batch size:

```bash
# Epochs ít - Nhanh
uv run python -m cli.main --epochs 10
uv run python -m cli.main --epochs 15

# Epochs trung bình - Mặc định
uv run python -m cli.main --epochs 20
uv run python -m cli.main --epochs 30

# Epochs nhiều - Chất lượng cao
uv run python -m cli.main --epochs 50
uv run python -m cli.main --epochs 100

# Batch size nhỏ
uv run python -m cli.main --batch-size 16 --epochs 20
uv run python -m cli.main --batch-size 32 --epochs 20

# Batch size lớn
uv run python -m cli.main --batch-size 64 --epochs 20
uv run python -m cli.main --batch-size 128 --epochs 20
```

---

### 📊 Phase 7: Tuning Data Amount

Thử với lượng dữ liệu khác nhau (15m):

```bash
# Ít dữ liệu - Nhanh, test
uv run python -m cli.main --limit 10000 --epochs 10
uv run python -m cli.main --limit 20000 --epochs 10

# Trung bình - Khuyến nghị
uv run python -m cli.main --limit 30000 --epochs 20
uv run python -m cli.main --limit 50000 --epochs 25

# Nhiều dữ liệu - Chất lượng cao (chậm)
uv run python -m cli.main --limit 100000 --epochs 50
uv run python -m cli.main --limit 200000 --epochs 100
```

---

### 🎯 Phase 8: Tổ Hợp Tốt Nhất (Best Combinations)

Dựa trên kết quả từ các phase trước, thử các tổ hợp tốt nhất:

```bash
# Tổ hợp 1: Timeframe 1d, Window lớn, Model lớn
uv run python -m cli.main \
    --timeframe 1d \
    --window 100 \
    --lstm-units 128 64 32 \
    --dropout 0.2 \
    --epochs 50 \
    --limit 3000

# Tổ hợp 2: Timeframe 4h, Window trung bình, Model trung bình
uv run python -m cli.main \
    --timeframe 4h \
    --window 60 \
    --lstm-units 64 32 \
    --dropout 0.2 \
    --epochs 30 \
    --limit 2000

# Tổ hợp 3: Timeframe 1d, Window lớn, Model lớn, Dropout cao
uv run python -m cli.main \
    --timeframe 1d \
    --window 90 \
    --lstm-units 128 64 \
    --dropout 0.3 \
    --epochs 40 \
    --limit 2500

# Tổ hợp 4: Timeframe 4h, Window nhỏ, Model nhỏ (nhanh)
uv run python -m cli.main \
    --timeframe 4h \
    --window 40 \
    --lstm-units 48 24 \
    --dropout 0.2 \
    --epochs 20 \
    --limit 1500
```

---

### 🔄 Phase 9: Advanced Tuning

Các thử nghiệm nâng cao:

```bash
# Thử với nhiều features (nếu có)
uv run python -m cli.main --features close volume --window 60

# Refresh cache để đảm bảo dữ liệu mới nhất
uv run python -m cli.main --refresh-cache --window 60 --epochs 20

# Seed khác nhau để kiểm tra tính ổn định
uv run python -m cli.main --seed 42 --window 60 --epochs 20
uv run python -m cli.main --seed 123 --window 60 --epochs 20
uv run python -m cli.main --seed 999 --window 60 --epochs 20
```

---

## 📈 So Sánh Kết Quả

### Cách So Sánh

1. **Xem danh sách kết quả**:
   ```bash
   ls -lt reports/cli/
   ```

2. **Đọc file markdown** của mỗi experiment:
   ```bash
   cat reports/cli/BiLSTM_1d_w60_20251227_133014/results_BiLSTM_1d_w60_20251227_133014.md
   ```

3. **So sánh các metrics quan trọng**:
   - **MAE** (Mean Absolute Error): Càng thấp càng tốt
   - **RMSE** (Root Mean Squared Error): Càng thấp càng tốt
   - **MAPE** (Mean Absolute Percentage Error): Càng thấp càng tốt
   - **Direction Accuracy**: Càng cao càng tốt (lý tưởng > 55%)

### Script So Sánh (Tùy Chọn)

Bạn có thể tạo script Python để tự động so sánh:

```python
import json
from pathlib import Path

def compare_results(base_dir="reports/cli"):
    results = []
    for folder in Path(base_dir).glob("BiLSTM_*"):
        metrics_file = folder / "metrics.json"
        config_file = folder / "config.json"
        if metrics_file.exists() and config_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            results.append({
                'folder': folder.name,
                'mae': metrics.get('mae', 0),
                'rmse': metrics.get('rmse', 0),
                'mape': metrics.get('mape', 0),
                'direction_accuracy': metrics.get('direction_accuracy', 0),
                'window_size': config.get('window_size'),
                'timeframe': config.get('timeframe'),
                'lstm_units': config.get('lstm_units'),
            })
    
    # Sắp xếp theo MAE (tốt nhất trước)
    results.sort(key=lambda x: x['mae'])
    
    print("\n" + "="*80)
    print("TOP 10 BEST RESULTS (sorted by MAE)")
    print("="*80)
    for i, r in enumerate(results[:10], 1):
        print(f"\n{i}. {r['folder']}")
        print(f"   MAE: ${r['mae']:.2f} | RMSE: ${r['rmse']:.2f} | MAPE: {r['mape']:.2f}%")
        print(f"   Direction Accuracy: {r['direction_accuracy']*100:.2f}%")
        print(f"   Config: {r['timeframe']}, w{r['window_size']}, {r['lstm_units']}")

if __name__ == "__main__":
    compare_results()
```

---

## 💡 Tips & Best Practices

### 1. **Bắt Đầu Từ Baseline**
Luôn chạy baseline trước để có điểm so sánh.

### 2. **Tune Từng Tham Số Một**
Đừng thay đổi tất cả cùng lúc. Tune từng tham số một để hiểu ảnh hưởng của nó.

### 3. **Ghi Chép Kết Quả**
Ghi lại các tham số và kết quả vào file Excel hoặc notebook để theo dõi.

### 4. **Kiểm Tra Overfitting**
So sánh `train_loss` và `val_loss`:
- Nếu `val_loss >> train_loss`: Model bị overfitting → Tăng dropout hoặc giảm model size
- Nếu cả hai đều cao: Model chưa học đủ → Tăng epochs hoặc model size

### 5. **Sử Dụng Early Stopping**
Early stopping tự động dừng khi val_loss không cải thiện, giúp tránh overfitting.

### 6. **Chạy Nhiều Lần Với Seed Khác Nhau**
Để đảm bảo kết quả ổn định, chạy cùng config với seed khác nhau.

### 7. **Ưu Tiên Direction Accuracy**
Đối với trading, Direction Accuracy quan trọng hơn MAE/RMSE vì nó đo khả năng dự đoán đúng hướng giá.

---

## 🎯 Kết Luận

Sau khi chạy các experiments trên:

1. **So sánh kết quả** từ các file `results_*.md`
2. **Chọn cấu hình tốt nhất** dựa trên mục tiêu của bạn:
   - Nếu muốn MAE thấp nhất → Chọn experiment có MAE thấp nhất
   - Nếu muốn Direction Accuracy cao nhất → Chọn experiment có Direction Accuracy cao nhất
   - Nếu muốn cân bằng → Chọn experiment có điểm số tổng hợp tốt nhất

3. **Sử dụng cấu hình tốt nhất** cho production hoặc tiếp tục fine-tune từ đó.

**Chúc bạn tìm được cấu hình tốt nhất! 🚀**
