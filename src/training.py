"""
🏋️ TRAINING MODULE - HUẤN LUYỆN MODEL
-----------------------------------------

Giải thích bằng ví dụ đời sống:
- Training giống như "tập làm bài tập"
- Model học từ các ví dụ (data) để tìm pattern
- Mỗi epoch = 1 lần học hết toàn bộ bài tập

Callback là gì?
- Giống như "người giám sát" trong quá trình training
- EarlyStopping: Dừng lại khi model không còn học được gì
- ModelCheckpoint: Lưu lại model tốt nhất
- ReduceLROnPlateau: Giảm learning rate khi model không còn tiến bộ

Trách nhiệm (SoC):
- Chỉ handle training logic
- Không chứa business logic khác
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import Config


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config
) -> Dict[str, Any]:
    """
    Huấn luyện model với callbacks

    Args:
        model: Model đã được build
        X_train, y_train: Dữ liệu train
        X_val, y_val: Dữ liệu validation
        config: Cấu hình

    Returns:
        Dictionary chứa:
            - history: Training history
            - best_epoch: Epoch có val_loss thấp nhất
            - train_seconds: Thời gian training
            - checkpoint_path: Đường dẫn checkpoint
    """
    # Tạo thư mục checkpoint
    checkpoint_dir = config.paths.models_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.keras"

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [checkpoint_callback, early_stop_callback, reduce_lr_callback]

    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU TRAINING")
    print("=" * 70)
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 70 + "\n")

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    train_seconds = time.perf_counter() - t0

    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])

    print("\n" + "=" * 70)
    print("✅ TRAINING HOÀN THÀNH")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}/{config.training.epochs}")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Best val_mae: {history.history['val_mae'][best_epoch-1]:.6f}")
    print(f"Training time: {train_seconds:.2f}s")
    print("=" * 70 + "\n")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_seconds": train_seconds,
        "checkpoint_path": str(checkpoint_path)
    }


def load_checkpoint(checkpoint_path: str) -> keras.Model:
    """
    Load model từ checkpoint

    Args:
        checkpoint_path: Đường dẫn đến file .keras hoặc .h5

    Returns:
        Model đã load
    """
    model = keras.models.load_model(checkpoint_path)
    print(f"✅ Đã load model từ: {checkpoint_path}")
    return model


def clean_checkpoints(
    config: Optional[Config] = None,
    keep_best: bool = True
) -> int:
    """
    Xóa các checkpoint cũ

    Args:
        config: Cấu hình
        keep_best: Có giữ lại checkpoint "best" không

    Returns:
        Số file đã xóa
    """
    if config is None:
        config = Config()

    checkpoint_dir = config.paths.models_dir / "checkpoints"

    if not checkpoint_dir.exists():
        return 0

    deleted_count = 0

    for file_path in checkpoint_dir.glob("*.keras"):
        if keep_best and "best" in file_path.name.lower():
            continue

        file_path.unlink()
        deleted_count += 1

    if deleted_count > 0:
        print(f"🗑️  Đã xóa {deleted_count} checkpoint")
    else:
        print("✅ Không có checkpoint nào để xóa")

    return deleted_count


if __name__ == "__main__":
    print("Training module loaded successfully!")
