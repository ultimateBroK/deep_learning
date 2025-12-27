"""
🔧 PREPROCESSING MODULE - XỬ LÝ DỮ LIỆU
-----------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "bếp chuẩn bị nguyên liệu" - rửa, cắt, chia tỷ lệ
- Dữ liệu thô → Dữ liệu sạch để model dễ học

Các bước:
1. Windowing: Tạo sliding windows (nhìn lại lịch sử)
2. Scaling: Chuẩn hoá về range 0-1
3. Splitting: Chia train/val/test

Trách nhiệm (SoC - Separation of Concerns):
- Chỉ xử lý dữ liệu, không làm gì khác
"""

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ==================== WINDOWING ====================
def create_windows(
    data: np.ndarray,
    window_size: int = 60,
    predict_steps: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo sliding windows từ dữ liệu

    Giải thích bằng ví dụ đời sống:
    - Giống như "nhìn qua cửa sổ lướt"
    - Mỗi lần nhìn 60 ngày qua để dự đoán ngày mai
    - Cửa sổ trượt từ đầu đến cuối

    Args:
        data: Dữ liệu đầu vào (shape: [n_samples, n_features])
        window_size: Số bước nhìn lại (past days)
        predict_steps: Số bước dự đoán (future days)

    Returns:
        X: Dữ liệu đầu vào (shape: [n_windows, window_size, n_features])
        y: Dữ liệu mục tiêu (shape: [n_windows, predict_steps])

    Ví dụ:
        data = [10, 20, 30, 40, 50, 60, 70]
        window_size = 3, predict_steps = 1

        Kết quả:
        X = [[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]]
        y = [[40], [50], [60], [70]]
    """
    X, y = [], []
    n_windows = len(data) - window_size - predict_steps + 1

    for i in range(n_windows):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + predict_steps])

    X = np.array(X)
    y = np.array(y)

    return X, y


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chia dữ liệu thành train/val/test

    Giải thích bằng ví dụ đời sống:
    - Giống như "chia bài" cho các ván chơi khác nhau
    - Train: Đặt câu hỏi và đáp án (học)
    - Val: Kiểm tra sau mỗi ván để điều chỉnh
    - Test: Thi cuối kỳ, không được xem đáp án

    Args:
        data: Dữ liệu cần chia
        train_ratio: Tỷ lệ train (mặc định 0.7)
        val_ratio: Tỷ lệ validation (mặc định 0.15)
        test_ratio = 1 - train_ratio - val_ratio = 0.15

    Returns:
        train, val, test
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    print("✅ Chia dữ liệu:")
    print(f"   Train: {len(train)} mẫu ({train_ratio:.0%})")
    print(f"   Val:   {len(val)} mẫu ({val_ratio:.0%})")
    print(f"   Test:  {len(test)} mẫu ({(1-train_ratio-val_ratio):.0%})")

    return train, val, test


# ==================== SCALING ====================
class DataScaler:
    """
    Class để xử lý scaling dữ liệu

    Giải thích bằng ví dụ đời sống:
    - Giống như "đổi đơn vị đo" - $50,000 → 0.5 (nếu scale 0-1)
    - Model học nhanh hơn khi số nhỏ và đồng nhất
    """

    def __init__(self, scaler_type: str = "minmax"):
        """
        Args:
            scaler_type: "minmax" hoặc "standard"
        """
        self.scaler_type = scaler_type
        self.scaler = None

        if scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(
                f"Scaler type không hợp lệ: {scaler_type}. "
                "Chọn 'minmax' hoặc 'standard'."
            )

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit scaler và transform dữ liệu (dùng cho training)

        Args:
            data: Dữ liệu đầu vào (2D array: [n_samples, n_features])

        Returns:
            Dữ liệu đã được scale
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        scaled_data = self.scaler.fit_transform(data)

        print(f"✅ Đã fit và transform dữ liệu với {self.scaler_type} scaler")
        print(f"   Min: {scaled_data.min():.4f}, Max: {scaled_data.max():.4f}")

        return scaled_data

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform dữ liệu (dùng cho validation/test)

        Lưu ý: KHÔNG fit lại scaler!

        Args:
            data: Dữ liệu đầu vào

        Returns:
            Dữ liệu đã được scale
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return self.scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform ngược từ scaled data về giá trị gốc

        Args:
            data: Dữ liệu đã scale

        Returns:
            Dữ liệu gốc
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return self.scaler.inverse_transform(data).flatten()

    def get_params(self) -> Dict:
        """Lấy params của scaler"""
        if hasattr(self.scaler, 'data_min_'):
            return {
                "scaler_type": self.scaler_type,
                "data_min_": self.scaler.data_min_.tolist(),
                "data_max_": self.scaler.data_max_.tolist(),
                "data_range_": self.scaler.data_range_.tolist(),
            }
        elif hasattr(self.scaler, 'mean_'):
            return {
                "scaler_type": self.scaler_type,
                "mean_": self.scaler.mean_.tolist(),
                "scale_": self.scaler.scale_.tolist(),
            }
        return {"scaler_type": self.scaler_type}


# ==================== COMPLETE PIPELINE ====================
def prepare_data_for_lstm(
    df: pl.DataFrame,
    features: List[str] = None,
    window_size: int = 60,
    scaler_type: str = "minmax",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict:
    """
    Pipeline hoàn chỉnh để chuẩn bị dữ liệu cho LSTM

    Giải thích bằng ví dụ đời sống:
    - Giống như "assembly line" - quy trình hoàn chỉnh
    - Input: DataFrame thô → Output: X_train, X_val, X_test, y_train, y_val, y_test

    Args:
        df: DataFrame với các cột OHLCV
        features: List features sử dụng
        window_size: Số nến nhìn lại
        scaler_type: Loại scaler
        train_ratio: Tỷ lệ train
        val_ratio: Tỷ lệ validation

    Returns:
        Dictionary chứa tất cả dữ liệu đã chuẩn bị
    """
    if features is None:
        features = ["close"]
    
    print("\n" + "=" * 70)
    print("🔧 CHUẨN BỊ DỮ LIỆU CHO LSTM")
    print("=" * 70 + "\n")

    # 1. Lấy features từ DataFrame
    feature_data = df.select(features).to_numpy()

    print(f"📊 Shape dữ liệu gốc: {feature_data.shape}")
    print(f"   Features: {features}")

    # 2. Scaling
    scaler = DataScaler(scaler_type=scaler_type)
    scaled_data = scaler.fit_transform(feature_data)

    # 3. Chia train/val/test (TRƯỚC khi tạo windows!)
    train_data, val_data, test_data = split_data(scaled_data, train_ratio, val_ratio)

    # 4. Tạo windows
    X_train, y_train = create_windows(train_data, window_size)
    X_val, y_val = create_windows(val_data, window_size)
    X_test, y_test = create_windows(test_data, window_size)

    print("\n✅ Dữ liệu sau khi tạo windows:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"   X_test:  {X_test.shape}, y_test: {y_test.shape}")
    print("")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    # Test
    import polars as pl
    import numpy as np

    # Tạo data giả
    dates = pl.date_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2022, 1, 1),
        interval="1d",
        eager=True
    )
    df = pl.DataFrame({
        "datetime": dates,
        "close": np.random.randn(len(dates)).cumsum() + 10000
    })

    result = prepare_data_for_lstm(df, window_size=30)
    print("X_train shape:", result["X_train"].shape)
