# Mô hình dự báo giá Bitcoin với BiLSTM

**Được thiết kế đặc biệt cho người ADHD - cấu trúc rõ ràng, dễ hiểu!**

> [!NOTE]
> Project đã được refactor theo **KISS**, **DRY**, **SoC**. Xem cấu trúc chi tiết bên dưới.

---

## 📁 Cấu Trúc Project

```
deep_learning/
├── src/                        # ⭐ SOURCE CODE CHÍNH
│   ├── config.py               # ⚙️ Config tập trung (DRY)
│   ├── pipeline.py             # 🔄 Pipeline chính (SoC)
│   ├── results.py              # 💾 Lưu kết quả
│   ├── training.py             # 🏋️ Training logic
│   ├── core/                   # 🎯 Business logic
│   │   ├── data.py            # 📥 Đọc dữ liệu
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
| **[START_HERE.md](START_HERE.md)** | Hướng dẫn bắt đầu | **BÂY GIỜ - BƯỚC 1!** |
| [docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md) | Hướng dẫn sống còn | Khi gặp vấn đề |
| [docs/ANALOGIES.md](docs/ANALOGIES.md) | Giải thích bằng ví dụ đời sống | Khi không hiểu khái niệm |
| [docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md) | Sơ đồ flow của chương trình | Khi muốn hiểu quy trình |

---

## 🚀 Quick Start

### Option 1: Chạy CLI (Nhanh)

```bash
# Cài đặt dependencies
uv sync

# Chạy với config mặc định
uv run python -m cli.main

# Chạy với tham số tùy chỉnh
uv run python -m cli.main --epochs 20 --limit 1500
uv run python -m cli.main --timeframe 4h --window 30
uv run python -m cli.main --preset fast       # Config nhanh - test
uv run python -m cli.main --preset high-quality  # Config chất lượng cao
```

**Các tham số quan trọng:**
- `--data-path`: Đường dẫn file CSV
- `--timeframe`: `1d`, `4h` (mặc định: `1d`)
- `--limit`: Lấy N dòng cuối (mặc định: `1500`)
- `--window`: Số nến nhìn lại (mặc định: `60`)
- `--epochs`: Số epochs (mặc định: `20`)
- `--preset`: `default`, `fast`, `high-quality`

### Option 2: Chạy Notebook (Khuyến nghị cho người mới)

```bash
uv sync
uv run jupyter notebook
```

Mở file `notebooks/run_complete.ipynb` và chạy từng cell theo thứ tự.

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
   2. Chạy CLI hoặc Notebook
   3. Đọc docs/ANALOGIES.md nếu không hiểu khái niệm

❌ SAI:
   - Nhảy lung tung → lạc lối nhanh!
```

### 2. Mỗi module 1 việc - Easy to find!

| Cần làm gì? | Mở file nào? |
|------------|--------------|
| Đổi config? | `src/config.py` |
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
→ Đọc `docs/ANALOGIES.md`

### Gặp lỗi?
→ Xem `docs/SURVIVAL_GUIDE.md`

---

## 📝 Lưu Ý Quan Trọng

- ✅ **Mỗi folder chỉ làm 1 việc** (SoC)
- ✅ **Config tập trung ở 1 nơi** (DRY)
- ✅ **Code đơn giản, rõ ràng** (KISS)
- ✅ **Comments bằng tiếng Việt** với analogies
- ✅ **Từng bước một** - đừng nhảy cóc!

---

## 🎯 Bắt Đầu Ngay!

Chọn 1 trong 2 cách:

1. **Nếu bạn thích hướng dẫn chi tiết:**
   → Đọc `START_HERE.md`
   → Chạy notebook: `uv run jupyter notebook`

2. **Nếu bạn thích nhanh gọn:**
   → Chạy CLI: `uv run python -m cli.main --preset fast`
