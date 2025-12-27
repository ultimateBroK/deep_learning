"""
🎯 CORE BUSINESS LOGIC
-----------------------

Module này chứa logic chính của project:
- data.py: Đọc/ghi dữ liệu
- preprocessing.py: Xử lý dữ liệu (windowing, scaling)
- model.py: Xây dựng model BiLSTM
- metrics.py: Tính toán metrics

Giải thích bằng ví dụ đời sống:
- Giống như "bếp nhà hàng" - nơi nấu món chính
- Mỗi bếp chuyên biệt 1 việc (data, preprocessing, model, metrics)
- Tất cả hoạt động nhịp nhàng để tạo ra món ăn (model dự đoán)
"""

from .data import fetch_binance_data, clear_cache
from .preprocessing import (
    create_windows,
    split_data,
    DataScaler,
    prepare_data_for_lstm
)
from .model import build_bilstm_model, print_model_summary
from .metrics import (
    evaluate_model,
    print_sample_predictions,
    calculate_direction_accuracy
)

__all__ = [
    # Data
    "fetch_binance_data",
    "clear_cache",
    # Preprocessing
    "create_windows",
    "split_data",
    "DataScaler",
    "prepare_data_for_lstm",
    # Model
    "build_bilstm_model",
    "print_model_summary",
    # Metrics
    "evaluate_model",
    "print_sample_predictions",
    "calculate_direction_accuracy",
]
