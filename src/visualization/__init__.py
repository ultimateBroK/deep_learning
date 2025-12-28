"""
VISUALIZATION MODULE - VẼ BIỂU ĐỒ
--------------------------------------

Giải thích bằng ví dụ đời sống:
- Visual giúp "thấy" data bằng mắt thay vì đọc số
- Biểu đồ dễ hiểu hơn hàng trăm con số
- Dễ nhận pattern, outlier, trend

Các biểu đồ:
1. Training History: Loss và val_loss qua từng epoch
2. Predictions vs Actual: So sánh dự đoán và thực tế
3. Residuals: Sai số phân phối như thế nào
"""

from .plots import (
    plot_training_history,
    plot_predictions,
    plot_residuals,
    plot_price_history,
    plot_all_in_one
)

__all__ = [
    "plot_training_history",
    "plot_predictions",
    "plot_residuals",
    "plot_price_history",
    "plot_all_in_one",
]
