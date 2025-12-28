# 🧭 Workflow 1 trang (ít rối, ít lặp)

## 0) Chuẩn bị

```bash
uv sync
```

## 1) Chọn cách chạy

### Cách A: CLI (nhanh)

```bash
# mặc định (15m)
uv run python -m cli.main

# chạy theo preset
uv run python -m cli.main --preset intraday-balanced
```

### Cách B: Notebook (dễ theo dõi)

```bash
uv run jupyter notebook
```

Mở `notebooks/run_complete.ipynb` và chạy từ trên xuống.

## 2) Xem output

Sau khi chạy xong sẽ có folder kết quả:

- CLI: `reports/cli/`
- Notebook: `reports/notebook/`

Trong mỗi folder kết quả thường có:

- `results_*.md`: report tổng hợp
- `metrics.json`: metrics
- `config.json`: cấu hình đã chạy
- `*.png`: biểu đồ

## 3) So sánh các lần chạy

- Nếu bạn chạy notebook nhiều lần: mở `reports/notebook/EVALUATION.md`
- Hoặc tự so: mở từng `metrics.json` và so các chỉ số (MAE/RMSE/MAPE/Direction Accuracy).

## 4) “Next step” gợi ý (khi muốn test cho ra kết luận)

- **Cố định dataset** (ví dụ `--limit 30000`) rồi sweep **window** (ví dụ 48/72/96/144).
- Mỗi lần chỉ đổi **1 biến** để biết cái gì đang ảnh hưởng kết quả.
- Đọc ý nghĩa metrics trong `docs/CONCEPTS.md` để tránh so sánh sai.

