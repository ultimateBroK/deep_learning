# ⭐ ĐỌC FILE NÀY TRƯỚC!

Chào mừng bạn đến với **Mô hình dự báo giá Bitcoin với BiLSTM**!

> Được thiết kế đặc biệt cho người ADHD - cấu trúc rõ ràng, dễ hiểu với analogies.

---

## 📋 CHỈ MỤC

- [Cấu trúc mới (Refactored)](#-cấu-trúc-mới-refactored)
- [Cách chạy](#-cách-chạy)
- [Tài liệu quan trọng](#-tài-liệu-quan-trọng)
- [Tips cho người ADHD](#-tips-cho-người-adhd)
- [Nếu bị lạc](#-nếu-bị-lạc)

---

## 🆕 Cấu Trúc Mới (Refactored)

Project đã được refactor theo 3 nguyên tắc quan trọng:

| Nguyên tắc | Nghĩa là gì? | Ví dụ đời sống |
|------------|--------------|-----------------|
| **KISS** | Keep It Simple, Stupid | "Làm đơn giản" - main.py từ 400 → 50 lines |
| **DRY** | Don't Repeat Yourself | "Không lặp lại" - config ở 1 file |
| **SoC** | Separation of Concerns | "Chia việc ra" - mỗi module 1 việc |

```
deep_learning/
├── src/                        # ⭐ SOURCE CODE CHÍNH
│   ├── config.py               # ⚙️ Config tập trung (DRY)
│   ├── pipeline.py             # 🔄 Pipeline chính (SoC)
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
└── docs/                       # 📚 Tài liệu
    ├── SURVIVAL_GUIDE.md        # Hướng dẫn sống còn
    ├── ANALOGIES.md             # Giải thích bằng ví dụ đời sống
    └── FLOW_DIAGRAM.md          # Sơ đồ flow của chương trình
```

---

## 🚀 Cách Chạy

### Option 1: Chạy CLI (Nhanh)

```bash
# Cài đặt dependencies
uv sync

# Chạy với config mặc định
uv run python -m cli.main

# Chạy với tham số tùy chỉnh
uv run python -m cli.main --epochs 20 --limit 1500
uv run python -m cli.main --timeframe 4h --window 30

# Dùng preset (config có sẵn)
uv run python -m cli.main --preset fast           # Nhanh - test
uv run python -m cli.main --preset high-quality  # Chất lượng cao - production
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

**⚠️ CẦN UPDATE IMPORTS (nếu notebook dùng cấu trúc cũ):**

| Import cũ | Import mới |
|-----------|-----------|
| `from step1_data import ...` | `from src.core import ...` |
| `from step2_preprocessing import ...` | `from src.core import ...` |
| `from step3_model import ...` | `from src.core import ...` |
| `from step4_training import ...` | `from src.training import ...` |
| `from step5_visualization import ...` | `from src.visualization import ...` |
| `from main.py import ...` | `from src import Config, run_pipeline` |

**Ví dụ:**
```python
# Cũ
from step1_data import fetch_binance_data
from step3_model import build_bilstm_model

# Mới
from src.core import fetch_binance_data, build_bilstm_model

# Hoặc đơn giản hơn:
from src import Config, run_pipeline
```

---

## 📚 Tài Liệu Quan Trọng

| Tài liệu | Nội dung | Khi nào đọc? |
|----------|---------|--------------|
| [docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md) | Hướng dẫn sống còn | Khi gặp vấn đề |
| [docs/ANALOGIES.md](docs/ANALOGIES.md) | Giải thích bằng ví dụ đời sống | Khi không hiểu khái niệm |
| [docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md) | Sơ đồ flow của chương trình | Khi muốn hiểu quy trình |

---

## 💡 Tips Cho Người ADHD

### 1. Làm theo flow - Don't jump around!

**Vấn đề:** Ng ADHD thường nhảy cóc → lạc lối

**Giải pháp:** Làm theo flow, từng bước một

```
✅ ĐÚNG:
   1. Đọc file này (START_HERE.md)
   2. Đọc docs/ANALOGIES.md → hiểu khái niệm
   3. Chạy CLI hoặc Notebook
   4. Đọc docs/SURVIVAL_GUIDE.md nếu gặp lỗi

❌ SAI:
   - Nhảy lung tung → lạc lối nhanh!
```

### 2. Mỗi module 1 việc - Easy to find!

**Vấn đề:** Code ở đâu?

**Giải pháp:** Tên module = chức năng

| Cần làm gì? | Mở file nào? |
|------------|--------------|
| Đổi config? | `src/config.py` |
| Đổi cách xử lý data? | `src/core/preprocessing.py` |
| Đổi model? | `src/core/model.py` |
| Đổi cách train? | `src/pipeline.py` |
| Đổi CLI args? | `cli/main.py` |

### 3. Đọc comments - Analogies everywhere!

**Vấn đề:** Code khó hiểu?

**Giải pháp:** Comments có analogies (ví dụ đời sống)

Ví dụ trong `src/core/model.py`:
```python
"""
Giải thích bằng ví dụ đời sống:
- BiLSTM giống như "nhìn bản đồ 2 chiều"
  - Trước → Sau (xu hướng tăng)
  - Sau → Trước (xu hướng giảm)
- Thấy rõ hơn so với LSTM thường!
"""
```

### 4. Dùng preset - Don't config everything!

**Vấn đề:** Quá nhiều options?

**Giải pháp:** Dùng preset (config có sẵn)

```bash
# Nhanh - test
uv run python -m cli.main --preset fast

# Mặc định - cân bằng
uv run python -m cli.main --preset default

# Chất lượng cao - production
uv run python -m cli.main --preset high-quality
```

---

## 🆘 Nếu Bị Lạc

### Quên mình đang làm gì?
→ Đọc lại file này (`START_HERE.md`)

### Không hiểu khái niệm?
→ Đọc `docs/ANALOGIES.md`

### Gặp lỗi?
→ Xem `docs/SURVIVAL_GUIDE.md`

### Muốn hiểu flow?
→ Xem `docs/FLOW_DIAGRAM.md`

### Không biết code ở đâu?
- Mỗi module chỉ có 1-2 files
- Tên module mô tả rõ ràng chức năng
- Xem table "Mỗi module 1 việc" ở trên

---

## 📝 Lưu Ý Quan Trọng

- ✅ **Cấu trúc mới** - đã refactor theo KISS, DRY, SoC
- ✅ **Config tập trung** - ở 1 file (`src/config.py`)
- ✅ **Mỗi module 1 việc** - dễ tìm, dễ sửa
- ✅ **Comments bằng tiếng Việt** với analogies
- ✅ **Từng bước một** - không nhảy cóc!

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

## 🎯 Bắt Đầu Ngay!

Chọn 1 trong 2 cách:

1. **Nếu bạn thích nhanh gọn:**
   → Chạy CLI: `uv run python -m cli.main --preset fast`

2. **Nếu bạn thích hướng dẫn chi tiết:**
   → Chạy notebook: `uv run jupyter notebook`
   → Mở `notebooks/run_complete.ipynb`
