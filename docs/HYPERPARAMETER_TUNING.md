# 🎯 Hyperparameter Tuning Guide - Tìm Kết Quả Tốt Nhất

Tài liệu này cung cấp danh sách các lệnh với các tham số khác nhau để tìm ra cấu hình tốt nhất cho model BiLSTM dự đoán giá Bitcoin.

---

## 📋 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Các Tham Số Quan Trọng](#các-tham-số-quan-trọng)
3. [Chiến Lược Tuning](#chiến-lược-tuning)
4. [Danh Sách Lệnh](#danh-sách-lệnh)
5. [So Sánh Kết Quả](#so-sánh-kết-quả)

---

## 🎯 Tổng Quan

### Mục Tiêu
Tìm ra cấu hình tốt nhất bằng cách thử nghiệm các tổ hợp tham số khác nhau và so sánh kết quả (MAE, RMSE, MAPE, Direction Accuracy).

### Quy Trình
1. **Chạy nhiều experiments** với các tham số khác nhau
2. **Lưu kết quả** vào `reports/cli/` (tên folder tự động chứa timeframe và window_size)
3. **So sánh metrics** từ các file `results_*.md`
4. **Chọn cấu hình tốt nhất** dựa trên mục tiêu (MAE thấp nhất, Direction Accuracy cao nhất, v.v.)

---

## 📊 Các Tham Số Quan Trọng

### 1. **Timeframe** (`--timeframe`)
- **Ảnh hưởng**: Độ phân giải dữ liệu
- **Giá trị**: `1d` (ngày), `4h` (4 giờ)
- **Khuyến nghị**: 
  - `1d`: Dự đoán dài hạn, ít noise
  - `4h`: Dự đoán ngắn hạn, nhiều dữ liệu hơn

### 2. **Window Size** (`--window`)
- **Ảnh hưởng**: Số nến nhìn lại để dự đoán
- **Giá trị**: 30-120 (thường dùng: 60)
- **Khuyến nghị**:
  - Nhỏ (30-40): Phản ứng nhanh với thay đổi gần đây
  - Trung bình (60-80): Cân bằng giữa ngắn hạn và dài hạn
  - Lớn (90-120): Tập trung vào xu hướng dài hạn

### 3. **LSTM Units** (`--lstm-units`)
- **Ảnh hưởng**: Độ phức tạp và khả năng học của model
- **Giá trị**: List các số nguyên, ví dụ: `64 32` hoặc `128 64 32`
- **Khuyến nghị**:
  - Nhỏ (`32 16`): Nhanh, ít overfitting, phù hợp dữ liệu nhỏ
  - Trung bình (`64 32`): Cân bằng tốt (mặc định)
  - Lớn (`128 64 32`): Mạnh hơn nhưng dễ overfitting, cần nhiều dữ liệu

### 4. **Dropout Rate** (`--dropout`)
- **Ảnh hưởng**: Giảm overfitting
- **Giá trị**: 0.0 - 0.5 (thường dùng: 0.2)
- **Khuyến nghị**:
  - Thấp (0.1-0.2): Khi model chưa overfit
  - Cao (0.3-0.5): Khi model bị overfit (val_loss >> train_loss)

### 5. **Epochs** (`--epochs`)
- **Ảnh hưởng**: Số lần học qua toàn bộ dữ liệu
- **Giá trị**: 10-100 (thường dùng: 20-50)
- **Khuyến nghị**:
  - Ít (10-20): Nhanh, phù hợp khi có early stopping
  - Nhiều (50-100): Cho kết quả tốt hơn nhưng lâu hơn

### 6. **Batch Size** (`--batch-size`)
- **Ảnh hưởng**: Kích thước batch trong training
- **Giá trị**: 16, 32, 64, 128
- **Khuyến nghị**: 
  - Nhỏ (16-32): Gradient update thường xuyên hơn, ổn định hơn
  - Lớn (64-128): Nhanh hơn nhưng có thể kém ổn định

### 7. **Limit** (`--limit`)
- **Ảnh hưởng**: Số lượng dữ liệu sử dụng
- **Giá trị**: 500-5000 (mặc định: 1500)
- **Khuyến nghị**:
  - Ít (500-1000): Nhanh, phù hợp test
  - Trung bình (1500-2000): Cân bằng tốt
  - Nhiều (3000+): Kết quả tốt hơn nhưng chậm hơn

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

# Baseline 3: Preset high-quality (chất lượng cao)
uv run python -m cli.main --preset high-quality
```

---

### 🔍 Phase 2: Tuning Timeframe

So sánh giữa timeframe 1d và 4h:

```bash
# Timeframe 1d (mặc định)
uv run python -m cli.main --timeframe 1d --window 60 --epochs 20

# Timeframe 4h (nhiều dữ liệu hơn)
uv run python -m cli.main --timeframe 4h --window 60 --epochs 20
```

---

### 📏 Phase 3: Tuning Window Size

Thử các window size khác nhau:

```bash
# Window nhỏ - Phản ứng nhanh
uv run python -m cli.main --window 30 --epochs 20
uv run python -m cli.main --window 40 --epochs 20

# Window trung bình - Cân bằng (mặc định)
uv run python -m cli.main --window 60 --epochs 20
uv run python -m cli.main --window 80 --epochs 20

# Window lớn - Xu hướng dài hạn
uv run python -m cli.main --window 90 --epochs 20
uv run python -m cli.main --window 100 --epochs 20
uv run python -m cli.main --window 120 --epochs 20
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

Thử với lượng dữ liệu khác nhau:

```bash
# Ít dữ liệu - Nhanh, test
uv run python -m cli.main --limit 500 --epochs 10
uv run python -m cli.main --limit 1000 --epochs 15

# Trung bình - Mặc định
uv run python -m cli.main --limit 1500 --epochs 20
uv run python -m cli.main --limit 2000 --epochs 20

# Nhiều dữ liệu - Chất lượng cao
uv run python -m cli.main --limit 3000 --epochs 30
uv run python -m cli.main --limit 5000 --epochs 50
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
