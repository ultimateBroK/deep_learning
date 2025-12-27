"""
🧠 MODEL MODULE - XÂY DỰNG MODEL BiLSTM
------------------------------------------

Giải thích bằng ví dụ đời sống:
- LSTM giống như "bộ nhớ ngắn hạn" - nhớ thông tin quan trọng
- BiLSTM (Bidirectional) nhìn cả quá khứ VÀ tương lai
- Tại sao nhìn tương lai được? Vì khi train, ta có toàn bộ dữ liệu!

Ví dụ đời sống:
- LSTM thường: "Nhìn bản đồ từ điểm A đến điểm B"
- BiLSTM: "Nhìn bản đồ từ A đến B + B đến A" → hiểu rõ hơn

Lợi ích:
- Phát hiện pattern tốt hơn
- Hiểu context từ 2 phía
- Kết quả thường tốt hơn LSTM thường
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_bilstm_model(
    input_shape: Tuple[int, int],
    lstm_units: list = [64, 32],
    dropout_rate: float = 0.2,
    dense_units: list = [16],
    output_units: int = 1,
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Xây dựng model BiLSTM

    Args:
        input_shape: Shape của dữ liệu đầu vào (window_size, n_features)
        lstm_units: List số units cho mỗi LSTM layer
        dropout_rate: Tỷ lệ dropout (để tránh overfitting)
        dense_units: List số units cho mỗi Dense layer
        output_units: Số units ở output layer (thường = 1)
        learning_rate: Learning rate cho optimizer

    Returns:
        Model BiLSTM đã được compile

    Ví dụ:
        input_shape = (60, 1)  # 60 ngày, 1 feature (giá close)
        lstm_units = [64, 32]   # 2 LSTM layers với 64 và 32 units
        output_units = 1       # Dự đoán 1 giá trị (giá ngày mai)
    """
    model = models.Sequential(name="BiLSTM_Price_Prediction")

    # Input layer
    model.add(layers.Input(shape=input_shape, name="input"))

    # BiLSTM layers
    # Layer 1: return_sequences=True để pass cho LSTM layer tiếp theo
    model.add(layers.Bidirectional(
        layers.LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            name="bilstm_1"
        ),
        name="bidirectional_1"
    ))
    model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    # Các LSTM layers tiếp theo (nếu có)
    for i, units in enumerate(lstm_units[1:], start=2):
        is_last = i == len(lstm_units)
        model.add(layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=not is_last,
                name=f"bilstm_{i}"
            ),
            name=f"bidirectional_{i}"
        ))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i}"))

    # Dense layers
    for i, units in enumerate(dense_units, start=1):
        model.add(layers.Dense(units, activation='relu', name=f"dense_{i}"))
        model.add(layers.Dropout(dropout_rate * 0.5, name=f"dense_dropout_{i}"))

    # Output layer
    model.add(layers.Dense(output_units, name="output"))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error - tốt cho regression
        metrics=['mae']  # Mean Absolute Error - dễ hiểu hơn
    )

    print(f"✅ Đã build model BiLSTM với {len(lstm_units)} LSTM layers")

    return model


def print_model_summary(model: models.Sequential):
    """
    In thông tin chi tiết về model

    Giải thích: Giống như "thông số kỹ thuật" - biết model có bao nhiêu layers, parameters
    """
    print("\n" + "=" * 60)
    print("🧠 MODEL SUMMARY")
    print("=" * 60)
    model.summary()
    print("=" * 60)

    # Đếm parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print("\n📊 Thống kê:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Non-trainable: {non_trainable_params:,}")


if __name__ == "__main__":
    # Test function
    input_shape = (60, 1)
    model = build_bilstm_model(input_shape=input_shape)
    print_model_summary(model)
