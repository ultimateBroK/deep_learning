"""
BƯỚC 2.2: CHUẨN HÓA DỮ LIỆU - SCALING
----------------------------------------

Giải thích bằng ví dụ đời sống:
- Giá Bitcoin dao động từ $10,000 đến $100,000
- Nếu ta để số nguyên, model sẽ bị rối vì số quá lớn
- Scaling giống như "đơn vị hóa" - đưa tất cả về cùng mức (0-1 hoặc -1 đến 1)

Ví dụ:
- $50,000 → 0.5 (nếu scale về 0-1)
- $10,000 → 0.1
- $90,000 → 0.9

Lợi ích:
1. Model học nhanh hơn
2. Số học ổn định hơn
3. Không bị số quá lớn/nhỏ gây lỗi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataScaler:
    """
    Class để xử lý scaling dữ liệu
    
    Giải thích:
    - MinMaxScaler: Đưa dữ liệu về khoảng [0, 1]
    - StandardScaler: Đưa dữ liệu về mean=0, std=1
    
    Với dự đoán giá crypto, MinMaxScaler thường tốt hơn vì:
    1. Giá luôn > 0
    2. Ta biết range của giá (min, max)
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
            raise ValueError(f"Scaler type không hợp lệ: {scaler_type}. Chọn 'minmax' hoặc 'standard'.")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit scaler và transform dữ liệu (dùng cho training)
        
        Args:
            data: Dữ liệu đầu vào (2D array: [n_samples, n_features])
        
        Returns:
            Dữ liệu đã được scale
        """
        # Đảm bảo data là 2D array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Fit và transform
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
        # Đảm bảo data là 2D array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform ngược từ scaled data về giá trị gốc
        
        Args:
            data: Dữ liệu đã scale
        
        Returns:
            Dữ liệu gốc ( chưa scale)
        """
        # Đảm bảo data là 2D array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        return self.scaler.inverse_transform(data)


def prepare_data_for_lstm(
    df: pd.DataFrame,
    features: list = ["close"],
    window_size: int = 60,
    scaler_type: str = "minmax"
) -> dict:
    """
    Pipeline hoàn chỉnh: Lấy data → Scale → Tạo windows → Split
    
    Args:
        df: DataFrame chứa dữ liệu giá
        features: List các features dùng (["close"], ["open", "close"], v.v.)
        window_size: Số bước nhìn lại
        scaler_type: Loại scaler ("minmax" hoặc "standard")
    
    Returns:
        Dictionary chứa:
            - X_train, y_train, X_val, y_val, X_test, y_test
            - scaler: Dùng để inverse transform
            - original_data: Dữ liệu gốc (để vẽ chart)
    """
    from .create_windows import create_windows, split_data
    
    # 1. Chọn các features cần dùng
    data = df[features].values
    
    print(f"📦 Dữ liệu gốc shape: {data.shape}")
    
    # 2. Scale dữ liệu
    scaler = DataScaler(scaler_type=scaler_type)
    scaled_data = scaler.fit_transform(data)
    
    # 3. Tạo windows
    X, y = create_windows(scaled_data, window_size=window_size, predict_steps=1)
    
    # 4. Chia train/val/test
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "original_data": data,
        "features": features,
        "window_size": window_size
    }


if __name__ == "__main__":
    # Test function
    data = np.array([10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000])
    
    # Scale dữ liệu
    scaler = DataScaler(scaler_type="minmax")
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    print("\nDữ liệu gốc:", data)
    print("Dữ liệu đã scale:", scaled.flatten())
    
    # Inverse transform
    original = scaler.inverse_transform(scaled)
    print("Dữ liệu sau khi inverse:", original.flatten())
