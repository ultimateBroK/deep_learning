"""
UTILS: LƯU KẾT QUẢ
-------------------

Giải thích bằng ví dụ đời sống:
- Lưu kết quả giống như ghi lại bài làm
- Khi nào cần xem lại, mở file là được
- Không phải chạy lại code từ đầu

Kết quả được lưu:
1. File Markdown: Tổng hợp mọi thứ (metrics, config, links)
2. Biểu đồ: PNG với độ phân giải cao
3. Model: File .keras để load lại sau
"""

from pathlib import Path
from datetime import datetime
from typing import Dict
import json


def _to_jsonable(obj):
    """
    Convert các kiểu không JSON-serializable (numpy scalar/array, Path, ...) về kiểu cơ bản.
    """
    # Local import để tránh ép dependency nếu không cần
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # Thử cast số kiểu lạ (int64/float64 từ polars có thể rơi vào đây)
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass

    return obj


def _md_table_kv(rows: list[tuple[str, str]]) -> str:
    """Tạo markdown table dạng key/value."""
    out = ["| Tham số / Parameter | Giá trị / Value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out) + "\n"


def _fmt(v) -> str:
    if v is None:
        return "-"
    return str(v)


def _fmt_money(v) -> str:
    try:
        return f"${float(v):.2f}"
    except Exception:
        return _fmt(v)


def create_results_folder(base_path: str = None, run_type: str = "main") -> Path:
    """
    Tạo folder để lưu kết quả
    
    Args:
        base_path: Đường dẫn cơ sở (mặc định: reports/)
        run_type: "main" hoặc "notebook"
    
    Returns:
        Đường dẫn đến folder kết quả
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "reports"
    else:
        base_path = Path(base_path)
    
    # Tạo timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tên folder: run_type/BiLSTM_YYYYMMDD_HHMMSS
    folder_path = base_path / run_type / f"BiLSTM_{timestamp}"
    folder_path.mkdir(parents=True, exist_ok=True)
    
    return folder_path


def save_markdown_report(
    folder_path: Path,
    config: Dict,
    metrics: Dict,
    history: Dict = None,
    plots: Dict = None
):
    """
    Lưu báo cáo tổng hợp dưới dạng Markdown
    
    Args:
        folder_path: Thư mục lưu báo cáo
        config: Cấu hình chạy
        metrics: Kết quả đánh giá
        history: Training history
        plots: Dict chứa tên file của các plot
    """
    report_path = folder_path / f"results_BiLSTM_{folder_path.name.replace('BiLSTM_', '')}.md"
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"# Kết quả dự đoán giá Bitcoin - BiLSTM / Bitcoin Price Prediction Results (BiLSTM)\n\n**Timestamp:** {now_str}\n\n---\n\n"

    # =====================
    # Tổng quan / cấu hình
    # =====================
    content += "## ⚙️ Cấu hình & dữ liệu / Config & Data\n\n"
    kv_rows: list[tuple[str, str]] = [
        ("Source CSV", _fmt(config.get("data_path"))),
        ("Timeframe", _fmt(config.get("timeframe", "1d"))),
        ("Rows (tail)", _fmt(config.get("limit", 1500))),
        ("Data rows", _fmt(config.get("data_rows"))),
        ("Date range", f"{_fmt(config.get('data_start'))} → {_fmt(config.get('data_end'))}"),
        ("Features", _fmt(config.get("features"))),
        ("Scaler", _fmt(config.get("scaler_type"))),
        ("Window size", _fmt(config.get("window_size", 60))),
        ("Split sizes (train/val/test)", f"{_fmt(config.get('train_samples'))} / {_fmt(config.get('val_samples'))} / {_fmt(config.get('test_samples'))}"),
        ("Seed", _fmt(config.get("seed"))),
        ("LSTM units", _fmt(config.get("lstm_units", [64, 32]))),
        ("Dropout rate", _fmt(config.get("dropout_rate", 0.2))),
        ("Epochs", _fmt(config.get("epochs", 20))),
        ("Batch size", _fmt(config.get("batch_size", 32))),
        ("Best epoch", _fmt(config.get("best_epoch"))),
        ("Best val_loss", _fmt(config.get("best_val_loss"))),
        ("Training time (s)", _fmt(config.get("train_seconds"))),
    ]
    content += _md_table_kv(kv_rows)
    content += "\n---\n\n"

    # =========
    # Metrics
    # =========
    content += "## 📊 Metrics / Chỉ số\n\n"
    metric_rows: list[tuple[str, str]] = [
        ("MAE (Sai số trung bình)", _fmt_money(metrics.get("mae", 0))),
        ("RMSE (Căn bậc 2 sai số)", _fmt_money(metrics.get("rmse", 0))),
        ("MAPE (Sai số phần trăm)", f"{float(metrics.get('mape', 0)):.2f}%" if isinstance(metrics.get("mape", 0), (int, float)) else _fmt(metrics.get("mape"))),
    ]
    if "direction_accuracy" in metrics:
        try:
            metric_rows.append(("Độ chính xác xu hướng", f"{float(metrics['direction_accuracy']) * 100:.2f}%"))
        except Exception:
            metric_rows.append(("Độ chính xác xu hướng", _fmt(metrics.get("direction_accuracy"))))
    content += _md_table_kv(metric_rows)

    # ==================
    # Ví dụ dự đoán
    # ==================
    y_true = metrics.get("y_true")
    y_pred = metrics.get("predictions")
    if y_true is not None and y_pred is not None:
        try:
            import numpy as np

            y_true_arr = np.array(y_true).reshape(-1)
            y_pred_arr = np.array(y_pred).reshape(-1)
            n = int(min(10, len(y_true_arr), len(y_pred_arr)))

            content += "\n---\n\n## 🔍 Ví dụ dự đoán (10 mẫu đầu) / Sample predictions (first 10)\n\n"
            content += "| # | Thực tế / Actual | Dự đoán / Pred | Sai số / Error | % Sai số / %Err |\n|---:|---:|---:|---:|---:|\n"
            for i in range(n):
                t = float(y_true_arr[i])
                p = float(y_pred_arr[i])
                err = abs(t - p)
                pct = (err / (t + 1e-8)) * 100
                content += f"| {i+1} | ${t:.2f} | ${p:.2f} | ${err:.2f} | {pct:.2f}% |\n"
        except Exception:
            # Không làm report fail chỉ vì phần sample
            pass
    
    # Thêm training history nếu có
    if history:
        final_epoch = len(history.get('loss', []))
        content += """
---

## 📈 Training History / Lịch sử huấn luyện

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
|-------|------------|----------|-----------|---------|
"""
        for i in range(final_epoch):
            content += f"| {i+1} | {history['loss'][i]:.6f} | {history['val_loss'][i]:.6f} | {history['mae'][i]:.4f} | {history['val_mae'][i]:.4f} |\n"
    
    # Thêm plots nếu có
    if plots:
        content += "\n---\n\n## 📊 Biểu đồ / Plots\n\n"
        if 'training_history' in plots:
            content += f"- [Training History](training_history_{plots['training_history']}.png)\n"
        if 'predictions' in plots:
            content += f"- [Predictions vs Actual](predictions_{plots['predictions']}.png)\n"
        if 'all_in_one' in plots:
            content += f"- [All-in-one Summary](all_in_one_{plots['all_in_one']}.png)\n"
    
    content += "\n---\n\n*Generated by BiLSTM Bitcoin Price Prediction Model*"
    
    # Lưu file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"💾 Đã lưu báo cáo Markdown: {report_path}")
    return report_path


def save_config(folder_path: Path, config: Dict):
    """
    Lưu cấu hình vào file JSON
    
    Args:
        folder_path: Thư mục lưu file
        config: Dict cấu hình
    """
    config_path = folder_path / "config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(config), f, indent=2, ensure_ascii=False)
    
    print(f"💾 Đã lưu cấu hình: {config_path}")


def save_metrics(folder_path: Path, metrics: Dict):
    """
    Lưu metrics vào file JSON
    
    Args:
        folder_path: Thư mục lưu file
        metrics: Dict metrics (có thể chứa numpy arrays)
    """
    import numpy as np
    
    metrics_path = folder_path / "metrics.json"
    
    # Chuyển numpy arrays sang lists để JSON serialize
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"💾 Đã lưu metrics: {metrics_path}")


def clean_old_reports(base_path: str = None, keep: int = 5):
    """
    Xóa các báo cáo cũ, chỉ giữ lại `keep` folder mới nhất
    
    Args:
        base_path: Đường dẫn cơ sở
        keep: Số folder cần giữ lại
    
    Returns:
        Số folder đã xóa
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "reports"
    else:
        base_path = Path(base_path)
    
    deleted_count = 0
    
    # Duyệt qua các thư mục con (main, notebook)
    for run_type_dir in base_path.iterdir():
        if not run_type_dir.is_dir():
            continue
        
        # Lấy danh sách các folder kết quả, sắp xếp theo thời gian giảm dần
        result_folders = sorted(run_type_dir.glob("BiLSTM_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Xóa các folder cũ hơn `keep`
        for folder in result_folders[keep:]:
            import shutil
            shutil.rmtree(folder)
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"🗑️  Đã xóa {deleted_count} báo cáo cũ (giữ lại {keep} mới nhất)")
    else:
        print("✅ Không có báo cáo nào để xóa")
    
    return deleted_count


if __name__ == "__main__":
    # Test functions
    folder = create_results_folder()
    print(f"Đã tạo folder kết quả: {folder}")
    
    # Test save markdown
    config = {
        'data_path': 'data/btc_1d_data_2018_to_2025.csv',
        'timeframe': '1d',
        'limit': 1500,
        'window_size': 60,
        'epochs': 20
    }
    
    metrics = {
        'mae': 500.0,
        'rmse': 700.0,
        'mape': 1.5
    }
    
    history = {
        'loss': [0.1, 0.08, 0.06],
        'val_loss': [0.12, 0.09, 0.07],
        'mae': [100, 80, 60],
        'val_mae': [120, 90, 70]
    }
    
    save_markdown_report(folder, config, metrics, history)
    save_config(folder, config)
    save_metrics(folder, metrics)
