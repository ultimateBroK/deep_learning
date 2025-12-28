# Hướng dẫn bắt đầu

Tài liệu này hướng dẫn chạy project end-to-end (CLI hoặc notebook) và tìm chỗ cần chỉnh khi muốn thử nghiệm thêm.

---

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cách chạy](#cách-chạy)
- [Tài liệu](#tài-liệu)
- [Nếu cần hỗ trợ](#nếu-cần-hỗ-trợ)

---

## Cấu trúc thư mục

```
deep_learning/
├── src/                        # Source code
│   ├── config.py               # Cấu hình tập trung (mặc định: 15m)
│   ├── pipeline.py             # Pipeline chính
│   ├── core/                   # Dữ liệu, preprocessing, model, metrics
│   ├── runtime/                # Runtime config
│   └── visualization/          # Vẽ biểu đồ
│
├── cli/                        # CLI
│   └── main.py                 # Entry point
│
├── scripts/                    # Tiện ích
│   └── clean.py                # Dọn dẹp project
│
└── docs/                       # Tài liệu
    ├── SURVIVAL_GUIDE.md        # Hướng dẫn sống còn
    ├── CONCEPTS.md              # Khái niệm (window, scaling, metrics, LSTM/BiLSTM)
    ├── FLOW_DIAGRAM.md          # Sơ đồ flow của chương trình
    ├── WORKFLOW.md              # Workflow 1 trang
    ├── README.md                # Index docs
    └── HYPERPARAMETER_TUNING.md # Presets & tuning
```

---

## Cách chạy

### Cách 1: CLI

```bash
# Cài dependencies
uv sync

# Chạy với cấu hình mặc định
uv run python -m cli.main

# Chạy với preset
uv run python -m cli.main --preset intraday-balanced

# Chạy với khung thời gian khác
uv run python -m cli.main --timeframe 1h --preset 1h-light
uv run python -m cli.main --timeframe 4h --preset 4h-balanced
uv run python -m cli.main --timeframe 1d --preset default

# Tuỳ chỉnh tham số
uv run python -m cli.main --epochs 20 --limit 15000
uv run python -m cli.main --timeframe 15m --window 240
uv run python -m cli.main --data-path data/btc_15m_data_2018_to_2025.csv
```

**Tham số chính:**
- `--data-path`: Đường dẫn file CSV (nếu không chỉ định → tự chọn theo timeframe)
- `--timeframe`: `15m`, `1h`, `4h`, `1d` (mặc định: `15m`)
- `--limit`: Lấy N dòng cuối (mặc định: `50000` cho 15m)
- `--window`: Số nến nhìn lại (mặc định: `240` cho 15m)
- `--epochs`: Số epochs (mặc định: `30`)
- `--preset`: Preset có sẵn

### Cách 2: Notebook

```bash
uv sync
uv run jupyter notebook
```

Mở file `notebooks/run_complete.ipynb` và chạy từng cell theo thứ tự.

---

## Workflow

Xem hướng dẫn 1 trang: `docs/WORKFLOW.md`

## Preset và tuning

Danh sách presets và gợi ý tuning (đã gom về 1 chỗ để tránh lặp):

- `docs/HYPERPARAMETER_TUNING.md`

**Nếu bạn có notebook/import theo cấu trúc cũ, cập nhật như sau:**

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

## Tài liệu

| Tài liệu | Nội dung | Khi nào đọc? |
|----------|---------|--------------|
| [docs/README.md](docs/README.md) | Index tài liệu | Khi cần overview |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Workflow (chạy → xem kết quả → so sánh) | Khi muốn chạy nhanh |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Khái niệm (window size, scaling, metrics, LSTM/BiLSTM) | Khi cần giải thích thuật ngữ |
| [docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md) | Sơ đồ luồng xử lý | Khi muốn hiểu pipeline |
| [docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md) | Hướng dẫn xử lý sự cố | Khi gặp lỗi |
| [docs/HYPERPARAMETER_TUNING.md](docs/HYPERPARAMETER_TUNING.md) | Preset và tuning | Khi cần chọn cấu hình |

---

## Nếu cần hỗ trợ

### Quên mình đang làm gì?
→ Đọc lại file này (`START_HERE.md`)

### Không hiểu khái niệm?
→ Đọc `docs/CONCEPTS.md`

### Muốn hiểu flow?
→ Xem `docs/FLOW_DIAGRAM.md`

### Gặp lỗi?
→ Xem `docs/SURVIVAL_GUIDE.md`

### Không biết code ở đâu?
- Mỗi module chỉ có 1-2 files
- Tên module mô tả rõ ràng chức năng
- Xem table "Mỗi module 1 việc" ở trên

---

## Ghi chú

- Cấu hình tập trung ở `src/config.py`.
- Pipeline chính ở `src/pipeline.py`.

---

## Dọn dẹp project

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

## Chạy nhanh

Chọn một trong hai cách:

1. CLI: `uv run python -m cli.main --preset intraday-balanced`
2. Notebook: `uv run jupyter notebook` → mở `notebooks/run_complete.ipynb`
