# Mô hình dự báo giá Bitcoin với BiLSTM

**Được thiết kế đặc biệt cho người ADHD - cấu trúc rõ ràng, dễ hiểu!**

> [!NOTE]
> Project đã được refactor theo **KISS**, **DRY**, **SoC**. Xem cấu trúc chi tiết bên dưới.
>
> **[!IMPORTANT]**
> Tập trung vào **15m timeframe** với data khổng lồ (~280K dòng).

---

## 📁 Cấu Trúc Project

```
deep_learning/
├── src/                        # ⭐ SOURCE CODE CHÍNH
│   ├── config.py               # ⚙️ Config tập trung (DRY) - Default: 15m, 50K lines
│   ├── pipeline.py             # 🔄 Pipeline chính (SoC)
│   ├── results.py              # 💾 Lưu kết quả
│   ├── training.py             # 🏋️ Training logic
│   ├── core/                   # 🎯 Business logic
│   │   ├── data.py            # 📥 Đọc dữ liệu (hỗ trợ 15m, 1h, 4h, 1d)
│   │   ├── preprocessing.py   # 🔧 Xử lý dữ liệu
│   │   ├── model.py           # 🧠 Xây dựng model
│   │   └── metrics.py         # 📊 Tính metrics
│   ├── runtime/                # ⚡ Runtime config
│   └── visualization/          # 📊 Vẽ biểu đồ
│
├── cli/                        # 🖥️ COMMAND LINE
│   └── main.py                # Entry point (KISS)
│
├── scripts/                    # 🔧 UTILITY SCRIPTS
│   └── clean.py               # Dọn dẹp project
│
├── data/                       # 📂 Dữ liệu
├── reports/                    # 📊 Kết quả
├── docs/                       # 📚 Tài liệu
├── notebooks/                  # 📓 Jupyter notebooks
├── START_HERE.md               # ⭐ ĐỌC ĐÂY TRƯỚC!
└── pyproject.toml
```

**Mỗi module chỉ làm 1 việc duy nhất, rõ ràng (SoC) - Don't Repeat Yourself (DRY) - Keep It Simple (KISS)**

---

## 📚 Tài Liệu Quan Trọng

| Tài liệu | Nội dung | Khi nào đọc? |
|----------|---------|--------------|
| **[START_HERE.md](START_HERE.md)** | Onboarding (từng bước) | **Bước 1** |
| **[docs/README.md](docs/README.md)** | Index docs (đọc gì ở đâu) | Khi bị lạc / muốn overview |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Workflow 1 trang (run → xem kết quả → so sánh) | Khi muốn làm nhanh, ít rối |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Khái niệm (window, scaling, metrics, LSTM/BiLSTM) | Khi cần hiểu “tại sao” |
| [docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md) | Troubleshooting / runbook | Khi gặp lỗi |
| [docs/HYPERPARAMETER_TUNING.md](docs/HYPERPARAMETER_TUNING.md) | Presets & tuning | Khi cần chọn preset/tuning |
| [docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md) | Sơ đồ flow chương trình | Khi muốn hiểu pipeline |

---

## 🚀 Quick Start

### Option 1: Chạy CLI (Nhanh)

```bash
# Cài đặt dependencies
uv sync

# Chạy với config mặc định (15m, 50K lines)
uv run python -m cli.main

# Chạy với preset tùy chỉnh (tập trung 15m)
uv run python -m cli.main --preset scalping-ultra-fast    # Scalping cực nhanh (6h)
uv run python -m cli.main --preset intraday-light          # Intraday nhẹ (1 ngày)
uv run python -m cli.main --preset swing-balanced          # Swing cân bằng (4 ngày)
uv run python -m cli.main --preset production              # Production chất lượng cao (8 ngày)

# Chạy với các timeframe khác
uv run python -m cli.main --timeframe 1h --preset 1h-light
uv run python -m cli.main --timeframe 4h --preset 4h-balanced
uv run python -m cli.main --timeframe 1d --preset default

# Chạy với tham số tùy chỉnh
uv run python -m cli.main --epochs 20 --limit 15000
uv run python -m cli.main --timeframe 15m --window 240
uv run python -m cli.main --data-path data/btc_15m_data_2018_to_2025.csv
```

**Các tham số quan trọng:**
- `--data-path`: Đường dẫn file CSV (nếu không chỉ định → tự chọn theo timeframe)
- `--timeframe`: `15m`, `1h`, `4h`, `1d` (mặc định: `15m`)
- `--limit`: Lấy N dòng cuối (mặc định: `50000` cho 15m)
- `--window`: Số nến nhìn lại (mặc định: `240` cho 15m)
- `--epochs`: Số epochs (mặc định: `30`)
- `--preset`: Preset có sẵn

### Option 2: Chạy Notebook (Khuyến nghị cho người mới)

```bash
uv sync
uv run jupyter notebook
```

Mở file `notebooks/run_complete.ipynb` và chạy từng cell theo thứ tự.

---

## 🧭 Workflow “không rối não”

Xem hướng dẫn 1 trang: `docs/WORKFLOW.md`

## 📦 Presets / tuning

Danh sách presets và cách tuning được gom về 1 chỗ (tránh lặp): `docs/HYPERPARAMETER_TUNING.md`

---

## 🧹 Dọn Dẹp Project

```bash
# Xem trước (dry-run)
uv run python -m scripts.clean

# Thực sự xóa
uv run python -m scripts.clean --execute

# Chỉ xóa cache cũ (> 7 ngày)
uv run python -m scripts.clean --cache --days 7

# Chỉ xóa reports cũ (giữ lại 3 folder mới nhất)
uv run python -m scripts.clean --reports --keep 3

# Xóa tất cả
uv run python -m scripts.clean --all --execute
```

---

## 📊 Kết Quả

Sau khi train, bạn sẽ thấy:

**Metrics:**
- **MAE**: Sai số trung bình tuyệt đối (USD)
- **RMSE**: Căn bậc hai của sai số bình phương trung bình (USD)
- **MAPE**: Sai số phần trăm trung bình (%)
- **Direction Accuracy**: Độ chính xác xu hướng (tăng/giảm)

**Biểu đồ:**
- Training history (loss, val_loss, mae, val_mae)
- Predictions vs Actual
- All-in-one (tất cả trong 1 figure)

**Kết quả được tự động lưu vào:**
- `reports/cli/` - Khi chạy CLI
- `reports/notebook/` - Khi chạy notebook

Mỗi lần chạy tạo folder chứa:
- `results_BiLSTM_YYYYMMDD_HHMMSS.md` - Báo cáo tổng hợp
- `training_history_*.png` - Biểu đồ training history
- `predictions_*.png` - Biểu đồ dự đoán
- `all_in_one_*.png` - Biểu đồ tổng hợp
- `config.json` - Cấu hình
- `metrics.json` - Metrics

---

## 💡 Tips Cho Người ADHD

### 1. Làm theo flow - Don't jump around!
```
✅ ĐÚNG:
   1. Đọc START_HERE.md
   2. Chọn preset phù hợp
   3. Chạy CLI hoặc Notebook
   4. Đọc docs/CONCEPTS.md nếu không hiểu khái niệm

❌ SAI:
   - Nhảy lung tung → lạc lối nhanh!
```

### 2. Mỗi module 1 việc - Easy to find!

| Cần làm gì? | Mở file nào? |
|------------|--------------|
| Đổi config/preset? | `src/config.py` |
| Đổi cách xử lý data? | `src/core/preprocessing.py` |
| Đổi model? | `src/core/model.py` |
| Đổi cách train? | `src/pipeline.py` |
| Đổi CLI args? | `cli/main.py` |

### 3. Đọc comments - Analogies everywhere!

Tất cả file code có analogies (ví dụ đời sống) để dễ hiểu:
- BiLSTM = "nhìn bản đồ 2 chiều"
- Sliding Window = "nhìn qua cửa sổ lướt"
- Scaling = "đổi đơn vị đo"

---

## ⚙️ Cấu Trúc Mới (Refactored)

| Nguyên tắc | Nghĩa là gì? | Ví dụ đời sống |
|------------|--------------|-----------------|
| **KISS** | Keep It Simple, Stupid | "Làm đơn giản" - main.py từ 400 → 50 lines |
| **DRY** | Don't Repeat Yourself | "Không lặp lại" - config ở 1 file |
| **SoC** | Separation of Concerns | "Chia việc ra" - mỗi module 1 việc |

---

## 🆘 Nếu Bị Lạc

### Quên mình đang làm gì?
→ Đọc lại `START_HERE.md`

### Không hiểu khái niệm?
→ Đọc `docs/CONCEPTS.md`

### Không biết chọn preset nào?
→ Đọc `docs/HYPERPARAMETER_TUNING.md`

### Gặp lỗi?
→ Xem `docs/SURVIVAL_GUIDE.md`

---

## 📝 Lưu Ý Quan Trọng

- ✅ **Mỗi folder chỉ làm 1 việc** (SoC)
- ✅ **Config tập trung ở 1 nơi** (DRY)
- ✅ **Code đơn giản, rõ ràng** (KISS)
- ✅ **Comments bằng tiếng Việt** với analogies
- ✅ **Từng bước một** - đừng nhảy cóc!
- ✅ **Tập trung vào 15m timeframe** với data khổng lồ
- ✅ **Sử dụng preset có sẵn** - đừng cấu hình thủ công khi không cần

---

## 🎯 Bắt Đầu Ngay!

Chọn 1 trong 2 cách:

1. **Nếu bạn thích hướng dẫn chi tiết:**
   → Đọc `START_HERE.md`
   → Chọn preset phù hợp
   → Chạy notebook: `uv run jupyter notebook`

2. **Nếu bạn thích nhanh gọn:**
   → Chọn preset từ bảng bên trên
   → Chạy CLI: `uv run python -m cli.main --preset scalping-fast`
