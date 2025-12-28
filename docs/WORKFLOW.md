# Workflow (1 trang)

## 0) Chuẩn bị

```bash
uv sync
```

## 1) Chạy project

### Cách A: CLI

```bash
# mặc định (15m)
uv run python -m cli.main

# chạy theo preset
uv run python -m cli.main --preset intraday-balanced
```

### Cách B: Notebook

```bash
uv run jupyter notebook
```

Mở `notebooks/run_complete.ipynb` và chạy từ trên xuống.

## 2) Xem kết quả

Sau khi chạy xong sẽ có thư mục kết quả:

- CLI: `reports/cli/`
- Notebook: `reports/notebook/`

Trong mỗi thư mục kết quả thường có:

- `results_*.md`: báo cáo
- `metrics.json`: metrics
- `config.json`: cấu hình đã chạy
- `*.png`: biểu đồ

## 3) So sánh các lần chạy

- Nếu bạn chạy notebook nhiều lần: mở `reports/notebook/EVALUATION.md`
- Hoặc tự so: mở từng `metrics.json` và so các chỉ số (MAE/RMSE/MAPE/Direction Accuracy).

## 4) Gợi ý khi muốn thử nghiệm có kết luận

- Cố định dataset (ví dụ `--limit 30000`), sau đó thử nhiều `--window` (ví dụ 48/72/96/144).
- Mỗi lần chỉ đổi 1 biến để biết yếu tố nào ảnh hưởng kết quả.
- Tham khảo ý nghĩa metrics trong `docs/CONCEPTS.md`.

