"""
BƯỚC 4.1: TRAINING MODEL
--------------------------

Giải thích bằng ví dụ đời sống:
- Training giống như bạn tập làm bài tập
- Model học từ các ví dụ (data) để tìm pattern
- Mỗi epoch = 1 lần học hết toàn bộ bài tập

Ví dụ:
- Epoch 1: Model học từ data lần đầu tiên, chưa hiểu nhiều
- Epoch 2: Model học lại, hiểu rõ hơn
- Epoch 20: Model đã hiểu tốt pattern của dữ liệu

Callback là gì?
- Giống như "người giám sát" trong quá trình training
- EarlyStopping: Dừng lại khi model không còn học được gì
- ModelCheckpoint: Lưu lại model tốt nhất
- ReduceLROnPlateau: Giảm learning rate khi model không còn tiến bộ
"""

from pathlib import Path
import time
import numpy as np
from typing import Dict
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    checkpoint_dir: str = None,
    checkpoint_name: str = "best_model.keras",
    early_stopping_patience: int = 5
) -> Dict:
    """
    Huấn luyện model
    
    Args:
        model: Model đã được build
        X_train, y_train: Dữ liệu train
        X_val, y_val: Dữ liệu validation
        epochs: Số lần học qua toàn bộ data
        batch_size: Số samples mỗi lần tính gradient
        checkpoint_dir: Thư mục lưu model
        checkpoint_name: Tên file checkpoint
        early_stopping_patience: Số epochs chờ trước khi dừng
    
    Returns:
        Dictionary chứa:
            - history: Training history
            - best_epoch: Epoch có val_loss thấp nhất
            - callbacks: List callbacks đã dùng
    """
    # Tạo thư mục checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent.parent / "reports" / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # 1. ModelCheckpoint: Lưu lại model tốt nhất
    checkpoint_callback = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
    
    # 2. EarlyStopping: Dừng lại nếu val_loss không giảm
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    # 3. ReduceLROnPlateau: Giảm learning rate nếu val_loss không giảm
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # Giảm LR đi 50%
        patience=3,       # Chờ 3 epochs
        min_lr=1e-6,      # LR tối thiểu
        verbose=1
    )
    
    callbacks = [checkpoint_callback, early_stop_callback, reduce_lr_callback]
    
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60 + "\n")
    
    # Training
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    train_seconds = time.perf_counter() - t0
    
    # Tìm epoch có val_loss thấp nhất
    best_epoch = np.argmin(history.history['val_loss']) + 1  # +1 vì epoch bắt đầu từ 1
    best_val_loss = min(history.history['val_loss'])
    
    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN THÀNH")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}/{epochs}")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Best val_mae: {history.history['val_mae'][best_epoch-1]:.6f}")
    print(f"Training time: {train_seconds:.2f}s")
    print("=" * 60 + "\n")
    
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_seconds": train_seconds,
        "callbacks": callbacks,
        "checkpoint_path": checkpoint_path
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


def clean_checkpoints(checkpoint_dir: str = None, keep_best: bool = True) -> int:
    """
    Xóa các checkpoint cũ
    
    Args:
        checkpoint_dir: Thư mục checkpoint
        keep_best: Có giữ lại checkpoint "best" không
    
    Returns:
        Số file đã xóa
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent.parent / "reports" / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
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
    # Test function (cần có model và data)
    print("Testing train_model...")
    print("Cần tạo model và data trước khi chạy!")
