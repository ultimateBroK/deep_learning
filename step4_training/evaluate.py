"""
BƯỚC 4.2: ĐÁNH GIÁ MODEL - EVALUATION
---------------------------------------

Giải thích bằng ví dụ đời sống:
- Evaluation giống như thi cuối kỳ
- Model chưa từng thấy data này (test set)
- Kết quả cho biết model "thực học" hay "thực chiến"

Các metrics:
1. MAE (Mean Absolute Error):
   - Sai số trung bình tuyệt đối
   - Ví dụ: MAE = $500 → Model dự đoán sai trung bình $500
   
2. RMSE (Root Mean Squared Error):
   - Căn bậc 2 của sai số bình phương trung bình
   - Nhấn mạnh vào các lỗi lớn (outliers)
   
3. MAPE (Mean Absolute Percentage Error):
   - Sai số phần trăm trung bình
   - Ví dụ: MAPE = 2% → Model sai trung bình 2%
"""

import numpy as np
from typing import Dict
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
            - y_true: Giá trị thật (đã inverse scale nếu có scaler)
            - predictions_scaled: Dự đoán (scaled, nếu return_predictions=True)
    """
    # Dự đoán trên test set
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Flatten nếu cần
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
    
    # MAPE (tránh chia cho 0)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET / TEST SET EVALUATION")
    print("=" * 60)
    print(f"MAE:  ${mae:.2f}  (Sai số trung bình tuyệt đối / Mean Absolute Error)")
    print(f"RMSE: ${rmse:.2f}  (Căn bậc 2 sai số bình phương / Root Mean Squared Error)")
    print(f"MAPE: {mape:.2f}%  (Sai số phần trăm trung bình / Mean Absolute Percentage Error)")
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
):
    """
    In một số ví dụ dự đoán
    
    Args:
        y_true: Giá trị thật
        y_pred: Dự đoán
        n_samples: Số mẫu in ra
    """
    n_samples = min(n_samples, len(y_true))
    
    print("\n" + "=" * 60)
    print(f"VÍ DỤ DỰ ĐOÁN (đầu {n_samples} mẫu) / SAMPLE PREDICTIONS (first {n_samples})")
    print("=" * 60)
    print(f"{'STT/#':<5} {'Thực tế/Actual':<15} {'Dự đoán/Pred':<15} {'Sai số/Error':<15} {'% Sai số/%Err':<12}")
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

    So sánh chiều xu hướng thực tế với dự đoán:
    - true_direction = actual[t+1] - actual[t] (xu hướng thực tế)
    - pred_direction = pred[t+1] - actual[t] (dự đoán đi từ actual[t])

    Args:
        y_true: Giá trị thật (đã inverse transform)
        y_pred: Dự đoán (đã inverse transform)
        threshold: Ngưỡng coi là "không đổi" (tương đương spread, phí)

    Returns:
        Độ chính xác (0-1)
    """
    # Xu hướng thực tế: actual[t] -> actual[t+1]
    true_change = np.diff(y_true)

    # Dự đoán xu hướng từ actual[t] đến pred[t+1]
    pred_change = y_pred[1:] - y_true[:-1]

    # Xác định xu hướng (tăng = 1, giảm = -1, không đổi = 0)
    true_direction = np.where(true_change > threshold, 1, np.where(true_change < -threshold, -1, 0))
    pred_direction = np.where(pred_change > threshold, 1, np.where(pred_change < -threshold, -1, 0))

    # Tính độ chính xác
    accuracy = np.mean(true_direction == pred_direction)

    print(f"📈 Độ chính xác xu hướng / Direction accuracy: {accuracy*100:.2f}%")

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
