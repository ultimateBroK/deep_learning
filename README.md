# Dự báo giá Bitcoin với BiLSTM

Dự án huấn luyện và đánh giá mô hình BiLSTM để dự báo giá BTC (mặc định: khung thời gian 15m).

---

## Cấu trúc thư mục

```
deep_learning/
├── src/                        # Source code
│   ├── config.py               # Cấu hình tập trung (mặc định: 15m, 50K dòng)
│   ├── pipeline.py             # Pipeline chính
│   ├── results.py              # Lưu kết quả
│   ├── training.py             # Training
│   ├── core/                   # Xử lý dữ liệu, model, metrics
│   ├── runtime/                # Runtime config
│   └── visualization/          # Vẽ biểu đồ
│
├── cli/                        # Command line interface
│   └── main.py                 # Entry point
│
├── scripts/                    # Tiện ích
│   └── clean.py                # Dọn dẹp project
│
├── data/                       # Dữ liệu
├── reports/                    # Kết quả
├── docs/                       # Tài liệu
├── notebooks/                  # Notebook
├── START_HERE.md               # Hướng dẫn bắt đầu
└── pyproject.toml
```

---

## Tài liệu

| Tài liệu | Nội dung | Khi nào đọc? |
|----------|---------|--------------|
| **[START_HERE.md](START_HERE.md)** | Hướng dẫn bắt đầu | Bước 1 |
| **[docs/README.md](docs/README.md)** | Index tài liệu | Khi cần overview |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Workflow (chạy → xem kết quả → so sánh) | Khi muốn chạy nhanh |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Khái niệm (window size, scaling, metrics, LSTM/BiLSTM) | Khi cần giải thích thuật ngữ |
| [docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md) | Hướng dẫn xử lý sự cố | Khi gặp lỗi |
| [docs/HYPERPARAMETER_TUNING.md](docs/HYPERPARAMETER_TUNING.md) | Preset và tuning | Khi cần chọn cấu hình |
| [docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md) | Sơ đồ luồng xử lý | Khi muốn hiểu pipeline |

---

## Quick start

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

Danh sách presets và cách tuning được gom về 1 chỗ (tránh lặp): `docs/HYPERPARAMETER_TUNING.md`

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

## Kết quả

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

## Khi gặp vấn đề

- Không hiểu thuật ngữ: `docs/CONCEPTS.md`
- Lỗi khi chạy: `docs/SURVIVAL_GUIDE.md`
