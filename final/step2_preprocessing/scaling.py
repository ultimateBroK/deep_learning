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


def _scale_3d_windows(scaler: DataScaler, X: np.ndarray) -> np.ndarray:
    """
    Scale windows dạng 3D: (n_samples, window_size, n_features) bằng scaler đã fit.
    """
    if X.ndim != 3:
        raise ValueError(f"X phải là 3D (n_samples, window_size, n_features). Nhận: {X.shape}")

    n_samples, window_size, n_features = X.shape
    X2d = X.reshape(-1, n_features)
    X2d_scaled = scaler.transform(X2d)
    return X2d_scaled.reshape(n_samples, window_size, n_features)


def _scale_y(scaler: DataScaler, y: np.ndarray) -> np.ndarray:
    """
    Scale y theo cùng scaler (dùng chung scaler của feature/price).

    Hỗ trợ:
    - y shape (n_samples, predict_steps, n_features)
    - y shape (n_samples, predict_steps)
    - y shape (n_samples,)
    """
    if y.ndim == 3:
        n_samples, predict_steps, n_features = y.shape
        y2d = y.reshape(-1, n_features)
        y2d_scaled = scaler.transform(y2d)
        return y2d_scaled.reshape(n_samples, predict_steps, n_features)

    if y.ndim == 2:
        # (n_samples, predict_steps) -> (n_samples*predict_steps, 1)
        y2d = y.reshape(-1, 1)
        return scaler.transform(y2d).reshape(y.shape[0], y.shape[1])

    if y.ndim == 1:
        return scaler.transform(y.reshape(-1, 1)).reshape(-1)

    raise ValueError(f"y có ndim không hỗ trợ: {y.ndim} (shape={y.shape})")


def prepare_data_for_lstm(
    df: pd.DataFrame,
    features: list = ["close"],
    window_size: int = 60,
    scaler_type: str = "minmax"
) -> dict:
    """
    Pipeline hoàn chỉnh (CHUẨN, tránh leakage):
    Lấy data → Tạo windows (raw) → Split theo thời gian → Fit scaler chỉ trên TRAIN → Transform train/val/test

    Lưu ý quan trọng:
    - Nếu fit scaler trên toàn bộ data trước khi split, bạn sẽ bị data leakage (val/test "thấy" min/max tương lai).
    - Target (y) mặc định lấy feature đầu tiên trong `features` để khớp output model (1 giá trị).
    
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

    # 2. Tạo windows trên RAW data trước (để split đúng theo thời gian)
    X_raw, y_raw = create_windows(data, window_size=window_size, predict_steps=1)

    # 3. Chia train/val/test (không shuffle)
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = split_data(X_raw, y_raw)

    # 4. Fit scaler CHỈ trên TRAIN (gộp cả X_train và y_train để scaler "biết" range mục tiêu)
    n_features = X_train_raw.shape[-1]
    train_fit_matrix = np.concatenate(
        [
            X_train_raw.reshape(-1, n_features),
            y_train_raw.reshape(-1, n_features),
        ],
        axis=0,
    )

    scaler = DataScaler(scaler_type=scaler_type)
    _ = scaler.fit_transform(train_fit_matrix)  # fit scaler trên train בלבד

    # 5. Transform X/y cho train/val/test
    X_train = _scale_3d_windows(scaler, X_train_raw)
    X_val = _scale_3d_windows(scaler, X_val_raw)
    X_test = _scale_3d_windows(scaler, X_test_raw)

    y_train_scaled = _scale_y(scaler, y_train_raw)
    y_val_scaled = _scale_y(scaler, y_val_raw)
    y_test_scaled = _scale_y(scaler, y_test_raw)

    # 6. Chuẩn hoá shape y về (n_samples, 1) để khớp model output_units=1
    # y_raw ban đầu: (n_samples, predict_steps=1, n_features). Ta lấy feature đầu tiên làm target.
    if y_train_scaled.ndim == 3:
        y_train = y_train_scaled[:, :, 0]
        y_val = y_val_scaled[:, :, 0]
        y_test = y_test_scaled[:, :, 0]
    else:
        y_train, y_val, y_test = y_train_scaled, y_val_scaled, y_test_scaled

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        # đã đúng shape (n_samples, 1)
        pass
    elif y_train.ndim == 2:
        # predict_steps > 1 (tương lai): giữ nguyên 2D (n_samples, predict_steps)
        pass
    else:
        # fallback: đảm bảo ít nhất 2D cho Keras
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    
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
