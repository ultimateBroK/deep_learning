"""
METRICS MODULE - TÍNH TOÁN METRICS
----------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "bảng điểm" - biết model học tốt như thế nào
- Sau mỗi bài kiểm tra, ta tính điểm

Các metrics:
1. MAE (Mean Absolute Error): Sai số trung bình tuyệt đối
2. RMSE (Root Mean Squared Error): Căn bậc 2 sai số bình phương
3. MAPE (Mean Absolute Percentage Error): Sai số phần trăm trung bình
4. Direction Accuracy: Độ chính xác khi dự đoán xu hướng (tăng/giảm)
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler=None,
    return_predictions: bool = False
) -> Dict:
    """
    Đánh giá model trên test set

    Giải thích bằng ví dụ đời sống:
    - Giống như "thi cuối kỳ" - model chưa từng thấy data này
    - Kết quả cho biết model "thực học" hay "thực chiến"

    Args:
        model: Model đã được train
        X_test: Dữ liệu test đầu vào
        y_test: Dữ liệu test mục tiêu (giá trị thật)
        scaler: Scaler để inverse transform
        return_predictions: Có trả về predictions không

    Returns:
        Dictionary chứa:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - mape: Mean Absolute Percentage Error
            - predictions: Dự đoán (nếu return_predictions=True)
            - y_true: Giá trị thật
    """
    # Dự đoán
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Flatten
    y_test_flat = y_test.flatten()
    y_pred_scaled_flat = y_pred_scaled.flatten()

    # Inverse transform nếu có scaler
    if scaler is not None:
        y_true = scaler.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled_flat.reshape(-1, 1)).flatten()
    else:
        y_true = y_test_flat
        y_pred = y_pred_scaled_flat

    # Tính metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 60)
    print(f"MAE:  ${mae:.2f}  (Sai số trung bình tuyệt đối)")
    print(f"RMSE: ${rmse:.2f}  (Căn bậc 2 sai số bình phương)")
    print(f"MAPE: {mape:.2f}%  (Sai số phần trăm trung bình)")
    print("=" * 60 + "\n")

    result = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "y_true": y_true,
        "predictions": y_pred
    }

    if return_predictions:
        result["predictions_scaled"] = y_pred_scaled_flat

    return result


def print_sample_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 10
) -> None:
    """
    In một số ví dụ dự đoán

    Giải thích bằng ví dụ đời sống:
    - Giống như "xem bài làm chi tiết" - biết sai ở đâu
    - Dễ thấy model dự đoán tốt/xấu ở trường hợp nào
    """
    n_samples = min(n_samples, len(y_true))

    print("\n" + "=" * 60)
    print(f"VÍ DỤ DỰ ĐOÁN (đầu {n_samples} mẫu)")
    print("=" * 60)
    print(f"{'STT':<5} {'Thực tế':<15} {'Dự đoán':<15} {'Sai số':<15} {'% Sai số':<12}")
    print("-" * 60)

    for i in range(n_samples):
        true_val = y_true[i]
        pred_val = y_pred[i]
        error = abs(true_val - pred_val)
        pct_error = (error / true_val) * 100

        print(f"{i+1:<5} ${true_val:<13.2f} ${pred_val:<13.2f} ${error:<13.2f} {pct_error:<9.2f}%")

    print("=" * 60 + "\n")


def calculate_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Tính độ chính xác khi dự đoán xu hướng (tăng/giảm)

    Giải thích bằng ví dụ đời sống:
    - Giống như "dự đoán đi lên hay đi xuống" - không cần đúng số liệu
    - Chỉ cần đúng xu hướng là được cộng điểm

    So sánh chiều xu hướng thực tế với dự đoán:
    - true_direction = actual[t+1] - actual[t] (xu hướng thực tế)
    - pred_direction = pred[t+1] - actual[t] (dự đoán đi từ actual[t])

    Args:
        y_true: Giá trị thật
        y_pred: Dự đoán
        threshold: Ngưỡng coi là "không đổi"

    Returns:
        Độ chính xác (0-1)
    """
    # Xu hướng thực tế
    true_change = np.diff(y_true)

    # Dự đoán xu hướng
    pred_change = y_pred[1:] - y_true[:-1]

    # Xác định xu hướng (tăng = 1, giảm = -1, không đổi = 0)
    true_direction = np.where(true_change > threshold, 1, np.where(true_change < -threshold, -1, 0))
    pred_direction = np.where(pred_change > threshold, 1, np.where(pred_change < -threshold, -1, 0))

    # Tính độ chính xác
    accuracy = np.mean(true_direction == pred_direction)

    print(f"Độ chính xác xu hướng: {accuracy*100:.2f}%")

    return accuracy


if __name__ == "__main__":
    # Test function
    y_true = np.array([50000, 51000, 49500, 52000, 52500])
    y_pred = np.array([50500, 50800, 49800, 51800, 52700])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    print_sample_predictions(y_true, y_pred)
    calculate_direction_accuracy(y_true, y_pred)
