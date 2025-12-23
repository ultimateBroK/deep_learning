"""
BƯỚC 2.1: TẠO CỬA SỔ (WINDOWS) - SLIDING WINDOW
------------------------------------------------

Giải thích bằng ví dụ đời sống:
- Sliding Window giống như bạn nhìn lại lịch sử giá của 60 ngày qua
- Để dự đoán giá ngày mai, bạn cần xem giá của 60 ngày trước đó
- Mỗi "window" là một chuỗi dữ liệu liên tục

Ví dụ:
- Window size = 60, thì mỗi sample là 60 ngày giá
- Sample 1: Day 0-59 → Dự đoán Day 60
- Sample 2: Day 1-60 → Dự đoán Day 61
- ...
"""

import numpy as np
import pandas as pd
from typing import Tuple


def create_windows(
    data: np.ndarray,
    window_size: int = 60,
    predict_steps: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo sliding windows từ dữ liệu
    
    Args:
        data: Dữ liệu đầu vào (shape: [n_samples, n_features])
        window_size: Số bước nhìn lại (past days)
        predict_steps: Số bước dự đoán (future days, thường = 1)
    
    Returns:
        X: Dữ liệu đầu vào (shape: [n_windows, window_size, n_features])
        y: Dữ liệu mục tiêu (shape: [n_windows, predict_steps])
    
    Ví dụ:
        data = [10, 20, 30, 40, 50, 60, 70]
        window_size = 3
        predict_steps = 1
        
        Kết quả:
        X = [[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]]
        y = [[40], [50], [60], [70]]
    """
    X, y = [], []
    
    # Tính số lượng windows có thể tạo
    n_windows = len(data) - window_size - predict_steps + 1
    
    for i in range(n_windows):
        # Window đầu vào: từ i đến i + window_size
        X.append(data[i:i + window_size])
        
        # Dữ liệu mục tiêu: từ i + window_size đến i + window_size + predict_steps
        y.append(data[i + window_size:i + window_size + predict_steps])
    
    # Chuyển sang numpy array
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Đã tạo {len(X)} windows:")
    print(f"   X shape: {X.shape} (samples, window_size, features)")
    print(f"   y shape: {y.shape} (samples, predict_steps)")
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chia dữ liệu thành train, validation, test
    
    Args:
        X: Dữ liệu đầu vào
        y: Dữ liệu mục tiêu
        train_ratio: Tỷ lệ dữ liệu train (mặc định 80%)
        val_ratio: Tỷ lệ dữ liệu validation (mặc định 10%)
    
    Returns:
        X_train, y_train: Dữ liệu train
        X_val, y_val: Dữ liệu validation
        X_test, y_test: Dữ liệu test
    
    Lưu ý:
        - Train: Dùng để huấn luyện model
        - Validation: Dùng để điều chỉnh hyperparameters
        - Test: Dùng để đánh giá cuối cùng (chỉ dùng 1 lần!)
    """
    n_samples = len(X)
    
    # Tính số lượng samples cho mỗi phần
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    print(f"📊 Chia dữ liệu:")
    print(f"   Train: {n_train} samples ({train_ratio*100}%)")
    print(f"   Val: {n_val} samples ({val_ratio*100}%)")
    print(f"   Test: {n_test} samples ({(1-train_ratio-val_ratio)*100}%)")
    
    # Chia dữ liệu theo thứ tự thời gian (không shuffle!)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Test function
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Tạo windows
    X, y = create_windows(data, window_size=3, predict_steps=1)
    print("\nDữ liệu gốc:", data)
    print("X:", X)
    print("y:", y)
    
    # Chia dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
