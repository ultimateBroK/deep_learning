"""
PLOTS MODULE - VẼ BIỂU ĐỒ
------------------------------

Giải thích bằng ví dụ đời sống:
- Visual giúp "thấy" data bằng mắt thay vì đọc số
- Biểu đồ dễ hiểu hơn hàng trăm con số
- Dễ nhận pattern, outlier, trend

Các biểu đồ:
1. Training History: Loss và val_loss qua từng epoch
2. Predictions vs Actual: So sánh dự đoán và thực tế
3. Residuals: Sai số phân phối như thế nào
4. All-in-One: Tất cả trong 1 figure
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Set style cho plots
plt.style.use('seaborn-v0_8-darkgrid')


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """
    Vẽ biểu đồ training history (loss và val_loss)

    Giải thích bằng ví dụ đời sống:
    - Giống như "điểm số qua các bài kiểm tra"
    - Thấy model có học được không

    Args:
        history: Training history từ model.fit()
        save_path: Đường dẫn để lưu plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # MAE plot
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_title('Training & Validation MAE', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu training history plot: {save_path}")

    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    n_samples: Optional[int] = None
) -> None:
    """
    Vẽ biểu đồ dự đoán vs thực tế

    Giải thích bằng ví dụ đời sống:
    - Giống như "so sánh đáp án với bài làm"
    - Thấy model dự đoán đúng/sai ở đâu

    Args:
        y_true: Giá trị thật
        y_pred: Dự đoán
        save_path: Đường dẫn để lưu plot
        n_samples: Số mẫu vẽ (None = vẽ tất cả)
    """
    if n_samples is not None:
        y_true = y_true[:n_samples]
        y_pred = y_pred[:n_samples]

    plt.figure(figsize=(14, 6))

    # Vẽ thực tế
    plt.plot(y_true, label='Thực tế', linewidth=2, alpha=0.8)

    # Vẽ dự đoán
    plt.plot(y_pred, label='Dự đoán', linewidth=2, alpha=0.8)

    plt.title('Dự đoán giá Bitcoin vs Thực tế', fontsize=16, fontweight='bold')
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Giá (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Thêm text với metrics
    mae = np.mean(np.abs(y_true - y_pred))
    pct_error = (mae / np.mean(y_true)) * 100
    plt.text(0.02, 0.98, f'MAE: ${mae:.2f}\nSai số: {pct_error:.2f}%',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu predictions plot: {save_path}")

    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ biểu đồ residuals (sai số)

    Giải thích bằng ví dụ đời sống:
    - Giống như "xem bài làm chi tiết" - sai ở đâu, nhiều hay ít
    - Residuals = Thực tế - Dự đoán

    Args:
        y_true: Giá trị thật
        y_pred: Dự đoán
        save_path: Đường dẫn để lưu plot
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals over time
    axes[0].plot(residuals, linewidth=1)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_title('Sai số (Residuals) theo thời gian', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Thời gian', fontsize=12)
    axes[0].set_ylabel('Sai số (USD)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_title('Phân phối sai số', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sai số (USD)', fontsize=12)
    axes[1].set_ylabel('Tần suất', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu residuals plot: {save_path}")

    plt.show()


def plot_price_history(
    df,
    price_column: str = "close",
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ lịch sử giá

    Args:
        df: DataFrame chứa dữ liệu
        price_column: Cột giá muốn vẽ
        save_path: Đường dẫn để lưu plot
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df['datetime'], df[price_column], linewidth=2, color='#2E86AB')

    plt.title(f'Lịch sử giá Bitcoin ({price_column})', fontsize=16, fontweight='bold')
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Giá (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu price history plot: {save_path}")

    plt.show()


def plot_all_in_one(
    history,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ tất cả plots vào một figure

    Giải thích bằng ví dụ đời sống:
    - Giống như "poster tổng hợp" - thấy tất cả ở 1 nơi
    - 4 plots: history, predictions, residuals, histogram

    Args:
        history: Training history
        y_true: Giá trị thật
        y_pred: Dự đoán
        save_path: Đường dẫn để lưu plot
    """
    plt.figure(figsize=(16, 10))

    # 1. Training History (top left)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Predictions (top right)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(y_true, label='Thực tế', linewidth=2, alpha=0.8)
    ax2.plot(y_pred, label='Dự đoán', linewidth=2, alpha=0.8)
    ax2.set_title('Dự đoán vs Thực tế', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Thời gian')
    ax2.set_ylabel('Giá (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residuals (bottom left)
    residuals = y_true - y_pred
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(residuals, linewidth=1)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_title('Sai số (Residuals)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Thời gian')
    ax3.set_ylabel('Sai số (USD)')
    ax3.grid(True, alpha=0.3)

    # 4. Histogram (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=1)
    ax4.set_title('Phân phối sai số', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sai số (USD)')
    ax4.set_ylabel('Tần suất')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Tổng hợp kết quả dự đoán giá Bitcoin', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu all-in-one plot: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test function
    print("Testing plot functions...")

    # Test plot_training_history
    class MockHistory:
        history = {
            'loss': [0.1, 0.08, 0.06, 0.05, 0.04],
            'val_loss': [0.12, 0.09, 0.07, 0.06, 0.05],
            'mae': [100, 80, 60, 50, 40],
            'val_mae': [120, 90, 70, 60, 50]
        }

    plot_training_history(MockHistory())

    # Test plot_predictions
    y_true = np.array([50000, 51000, 50500, 52000, 52500, 53000, 52800, 53500])
    y_pred = np.array([50500, 50800, 50200, 51800, 52200, 53200, 53000, 53300])
    plot_predictions(y_true, y_pred)
