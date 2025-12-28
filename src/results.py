"""
RESULTS MODULE - LƯU KẾT QUẢ
----------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "file hồ sơ" - lưu tất cả kết quả
- Khi cần xem lại, mở file là được
- Không phải chạy lại code từ đầu

Kết quả được lưu:
1. File Markdown: Tổng hợp mọi thứ (metrics, config, links)
2. Biểu đồ: PNG với độ phân giải cao
3. Model: File .keras để load lại sau
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _to_jsonable(obj: Any) -> Any:
    """
    Convert các kiểu không JSON-serializable về kiểu cơ bản
    """
    try:
        import numpy as np
    except Exception:
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

    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass

    return obj


def create_results_folder(
    base_path: Optional[Path] = None,
    run_type: str = "main",
    config: Optional[Dict] = None
) -> Path:
    """
    Tạo folder để lưu kết quả với tên chuẩn hóa

    Giải thích bằng ví dụ đời sống:
    - Giống như "tạo folder hồ sơ mới" cho mỗi lần chạy
    - Không bị lẫn với kết quả lần trước
    - Tên folder chứa thông tin quan trọng để dễ phân biệt

    Format tên (đơn giản): BiLSTM_{timeframe}_w{window}_l{limit}_{timestamp}
    Ví dụ: BiLSTM_15m_w96_l30k_20251227_133014
    
    Format đầy đủ (nếu có thêm tham số): BiLSTM_{timeframe}_w{window}_l{limit}_e{epochs}_u{lstm_units}_d{dropout}_b{batch}_{scaler}_{timestamp}
    Ví dụ: BiLSTM_15m_w96_l30k_e20_u64-32_d20_b32_mm_20251227_133014

    Args:
        base_path: Đường dẫn cơ sở
        run_type: "main", "notebook", "test"
        config: Dict chứa config với các tham số:
            - timeframe: str (bắt buộc)
            - window_size: int (bắt buộc)
            - limit: int (tùy chọn) - số dòng dữ liệu (khuyến nghị dùng)
            - epochs: int (tùy chọn) - chỉ thêm nếu cần phân biệt
            - lstm_units: List[int] (tùy chọn) - chỉ thêm nếu cần phân biệt
            - dropout_rate: float (tùy chọn) - chỉ thêm nếu cần phân biệt
            - batch_size: int (tùy chọn) - chỉ thêm nếu cần phân biệt
            - scaler_type: str (tùy chọn) - chỉ thêm nếu cần phân biệt
    
    Note: Format mặc định chỉ dùng timeframe, window_size, limit (đồng bộ với notebook)

    Returns:
        Đường dẫn đến folder kết quả
    """
    from .config import Paths  # noqa: E402 - Import here to avoid circular dependency

    if base_path is None:
        base_path = Paths().reports_dir
    else:
        base_path = Path(base_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tạo tên folder chuẩn hóa với thông tin config
    if config and 'timeframe' in config and 'window_size' in config:
        timeframe = str(config['timeframe']).replace('/', '_')  # Xử lý ký tự đặc biệt
        window_size = config['window_size']
        
        # Bắt đầu với phần cơ bản
        parts = [f"BiLSTM_{timeframe}", f"w{window_size}"]
        
        # Thêm limit nếu có (rút gọn: 30000 -> l30k, 50000 -> l50k, 100000 -> l100k)
        if 'limit' in config and config['limit']:
            limit = config['limit']
            if limit >= 1000:
                # Rút gọn: 30000 -> l30k, 50000 -> l50k, 100000 -> l100k
                limit_k = limit // 1000
                parts.append(f"l{limit_k}k")
            else:
                parts.append(f"l{limit}")
        
        # Thêm epochs nếu có
        if 'epochs' in config and config['epochs']:
            parts.append(f"e{config['epochs']}")
        
        # Thêm LSTM units nếu có (rút gọn: 64-32 thay vì 64_32)
        if 'lstm_units' in config and config['lstm_units']:
            lstm_units = config['lstm_units']
            if isinstance(lstm_units, (list, tuple)):
                units_str = '-'.join(str(u) for u in lstm_units)
            else:
                units_str = str(lstm_units)
            parts.append(f"u{units_str}")
        
        # Thêm dropout rate nếu có (làm tròn 2 chữ số, bỏ dấu chấm: 0.2 -> d20, 0.05 -> d05)
        if 'dropout_rate' in config and config['dropout_rate'] is not None:
            dropout = config['dropout_rate']
            # Chuyển 0.2 thành "20", 0.05 thành "05", giữ nguyên số 0 đầu
            dropout_str = f"{int(dropout * 100):02d}"
            parts.append(f"d{dropout_str}")
        
        # Thêm batch size nếu có
        if 'batch_size' in config and config['batch_size']:
            parts.append(f"b{config['batch_size']}")
        
        # Thêm scaler type nếu có (rút gọn: minmax -> mm, standard -> std)
        if 'scaler_type' in config and config['scaler_type']:
            scaler = config['scaler_type']
            if scaler == 'minmax':
                parts.append('mm')
            elif scaler == 'standard':
                parts.append('std')
            else:
                parts.append(scaler[:2])  # Lấy 2 ký tự đầu
        
        # Thêm timestamp
        parts.append(timestamp)
        
        folder_name = '_'.join(parts)
    else:
        # Fallback nếu không có config đầy đủ
        folder_name = f"BiLSTM_{timestamp}"
    
    folder_path = base_path / run_type / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path


def save_config(
    folder_path: Path,
    config: Dict
) -> None:
    """
    Lưu cấu hình ra file JSON

    Args:
        folder_path: Thư mục lưu
        config: Cấu hình
    """
    config_path = folder_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(_to_jsonable(config), f, indent=2)
    print(f"Đã lưu config: {config_path}")


def save_metrics(
    folder_path: Path,
    metrics: Dict
) -> None:
    """
    Lưu metrics ra file JSON

    Args:
        folder_path: Thư mục lưu
        metrics: Dictionary metrics
    """
    metrics_path = folder_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(_to_jsonable(metrics), f, indent=2)
    print(f"Đã lưu metrics: {metrics_path}")


def save_markdown_report(
    folder_path: Path,
    config: Dict,
    metrics: Dict,
    history: Optional[Dict] = None,
    plots: Optional[Dict] = None
) -> None:
    """
    Lưu báo cáo tổng hợp dưới dạng Markdown

    Giải thích bằng ví dụ đời sống:
    - Giống như "báo cáo tổng hợp" - tất cả ở 1 file
    - Dễ chia sẻ, dễ xem, dễ tìm kiếm

    Args:
        folder_path: Thư mục lưu báo cáo
        config: Cấu hình chạy
        metrics: Kết quả đánh giá
        history: Training history
        plots: Dict chứa tên file của các plot
    """
    # Tạo tên file markdown tương ứng với tên folder
    # Format: results_{folder_name}.md
    report_path = folder_path / f"results_{folder_path.name}.md"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"# Kết quả dự đoán giá Bitcoin - BiLSTM\n\n**Timestamp:** {now_str}\n\n---\n\n"

    # Cấu hình
    content += "## Cấu hình & Dữ liệu\n\n"
    content += "| Tham số | Giá trị |\n|---|---|\n"
    for k, v in config.items():
        if v is not None and not k.startswith('_'):
            content += f"| {k} | {v} |\n"
    content += "\n"

    # Metrics
    if metrics:
        content += "## Kết quả Đánh Giá\n\n"
        content += "| Metric | Giá trị |\n|---|---|\n"
        if "mae" in metrics:
            content += f"| MAE | ${metrics['mae']:.2f} |\n"
        if "rmse" in metrics:
            content += f"| RMSE | ${metrics['rmse']:.2f} |\n"
        if "mape" in metrics:
            content += f"| MAPE | {metrics['mape']:.2f}% |\n"
        if "direction_accuracy" in metrics:
            content += f"| Direction Accuracy | {metrics['direction_accuracy']*100:.2f}% |\n"
        content += "\n"

    # Training history
    if history:
        content += "## Training History\n\n"
        content += "| Metric | Final |\n|---|---|\n"
        if 'loss' in history:
            content += f"| Train Loss | {history['loss'][-1]:.6f} |\n"
        if 'val_loss' in history:
            content += f"| Val Loss | {history['val_loss'][-1]:.6f} |\n"
        if 'mae' in history:
            content += f"| Train MAE | {history['mae'][-1]:.6f} |\n"
        if 'val_mae' in history:
            content += f"| Val MAE | {history['val_mae'][-1]:.6f} |\n"
        content += "\n"

    # Plots
    if plots:
        content += "## Biểu Đồ / Plots\n\n"
        for plot_name, plot_file in plots.items():
            content += f"- **{plot_name}**: `{plot_file}`\n"
        content += "\n"

    content += "---\n\n*Generated by BiLSTM Bitcoin Price Predictor*\n"

    with open(report_path, 'w') as f:
        f.write(content)

    print(f"Đã lưu báo cáo: {report_path}")


def clean_old_reports(
    base_path: Optional[Path] = None,
    run_type: str = "main",
    keep: int = 5
) -> int:
    """
    Xóa báo cáo cũ, chỉ giữ lại N báo cáo mới nhất

    Args:
        base_path: Thư mục cơ sở
        run_type: Loại chạy ("main", "notebook")
        keep: Số báo cáo cần giữ

    Returns:
        Số báo cáo đã xóa
    """
    import shutil

    from .config import Paths  # noqa: E402 - Import here to avoid circular dependency

    if base_path is None:
        base_path = Paths().reports_dir
    else:
        base_path = Path(base_path)

    run_dir = base_path / run_type
    if not run_dir.exists():
        return 0

    # Lấy danh sách folder, sắp theo thời gian giảm dần
    folders = sorted(run_dir.glob("BiLSTM_*"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Xóa các folder cũ
    deleted_count = 0
    for folder in folders[keep:]:
        shutil.rmtree(folder)
        deleted_count += 1

    if deleted_count > 0:
        print(f"Đã xóa {deleted_count} báo cáo cũ")
    else:
        print("Không có báo cáo cũ nào để xóa")

    return deleted_count


if __name__ == "__main__":
    # Test
    folder = create_results_folder()
    save_config(folder, {"test": "value"})
    save_metrics(folder, {"mae": 100, "rmse": 150})
    save_markdown_report(
        folder,
        {"window_size": 60},
        {"mae": 100, "rmse": 150},
        {"loss": [0.1, 0.08], "val_loss": [0.12, 0.09]},
        {"history": "test.png"}
    )
    print(f"Created: {folder}")
