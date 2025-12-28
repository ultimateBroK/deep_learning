"""
CONFIG TẬP TRUNG - CENTRALIZED CONFIGURATION
--------------------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "menu nhà hàng" - tất cả lựa chọn ở 1 nơi
- Thay đổi ở đây → thay đổi toàn bộ project
- Không cần sửa code ở nhiều nơi (DRY - Don't Repeat Yourself)

Ví dụ:
- Muốn thay số epochs → sửa ở đây, không sửa trong main.py, notebook, test...
- Muốn thay timeframe → sửa ở đây, tất cả tự động cập nhật

TẬP TRUNG VÀO 15m TIMEFRAME (280K DÒNG DATA)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ==================== PROJECT PATHS ====================
# Giống như "bản đồ" - biết mọi thứ nằm ở đâu
@dataclass
class Paths:
    """Đường dẫn các thư mục trong project"""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.reports_dir = self.project_root / "reports"
        self.models_dir = self.project_root / "models"
        self.cache_dir = self.data_dir / "cache"  # Cache trong thư mục data

        # Tạo các thư mục nếu chưa có
        for p in [self.data_dir, self.reports_dir, self.models_dir, self.cache_dir]:
            p.mkdir(parents=True, exist_ok=True)


# ==================== DATA CONFIG ====================
# Giống như "đặc tả nguyên liệu" - dùng loại dữ liệu nào
@dataclass
class DataConfig:
    """Cấu hình cho dữ liệu"""

    # File dữ liệu
    data_path: str = None  # None = tự chọn theo timeframe
    timeframe: str = "15m"  # Default là 15m để tận dụng data khủng

    # Giới hạn dữ liệu
    limit: int = 50000  # Default là 50k dòng cho 15m

    # Features dùng để dự đoán
    features: List[str] = field(default_factory=lambda: ["close"])

    # Có refresh cache không
    refresh_cache: bool = False

    def get_data_file(self) -> Path:
        """Lấy đường dẫn file CSV theo timeframe"""
        if self.data_path:
            return Path(self.data_path)

        tf = self.timeframe.lower()
        paths = Paths()

        if tf == "15m":
            return paths.data_dir / "btc_15m_data_2018_to_2025.csv"
        if tf == "1h":
            return paths.data_dir / "btc_1h_data_2018_to_2025.csv"
        if tf == "4h":
            return paths.data_dir / "btc_4h_data_2018_to_2025.csv"
        return paths.data_dir / "btc_1d_data_2018_to_2025.csv"


# ==================== PREPROCESSING CONFIG ====================
# Giống như "công thức chế biến" - xử lý dữ liệu thế nào
@dataclass
class PreprocessingConfig:
    """Cấu hình cho tiền xử lý dữ liệu"""

    # Sliding Window
    window_size: int = 240  # Default: 4 ngày 15m
    predict_steps: int = 1  # Số bước dự đoán (thường = 1)

    # Scaling
    scaler_type: str = "minmax"  # minmax hoặc standard

    # Train/Val/Test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio = 0.15


# ==================== MODEL CONFIG ====================
# Giống như "thiết kế kiến trúc" - model có cấu trúc nào
@dataclass
class ModelConfig:
    """Cấu hình cho model BiLSTM"""

    # LSTM layers
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])

    # Dropout
    dropout_rate: float = 0.2

    # Dense layers
    dense_units: List[int] = field(default_factory=lambda: [32])

    # Output
    output_units: int = 1

    def get_input_shape(self, window_size: int, n_features: int) -> Tuple[int, int]:
        """Lấy shape đầu vào cho model"""
        return (window_size, n_features)


# ==================== TRAINING CONFIG ====================
# Giống như "lịch học tập" - học thế nào
@dataclass
class TrainingConfig:
    """Cấu hình cho training"""

    # Training parameters
    epochs: int = 30
    batch_size: int = 32

    # Early stopping
    early_stopping_patience: int = 10

    # Learning rate
    learning_rate: float = 0.001

    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = None  # None = auto

    def get_checkpoint_dir(self, paths: Paths) -> Path:
        """Lấy thư mục lưu checkpoint"""
        if self.checkpoint_dir:
            return Path(self.checkpoint_dir)
        return paths.models_dir / "checkpoints"


# ==================== RUNTIME CONFIG ====================
# Giống như "cấu hình máy tính" - chạy thế nào
@dataclass
class RuntimeConfig:
    """Cấu hình cho runtime TensorFlow"""

    # CPU threads (tối ưu cho CPU AMD)
    intra_op_threads: int = 12  # Số core vật lý
    inter_op_threads: int = 2

    # XLA optimization
    enable_xla: bool = True

    # Random seed
    seed: int = 42

    # GPU settings
    use_gpu: bool = False  # False = chỉ dùng CPU


# ==================== VISUALIZATION CONFIG ====================
# Giống như "thiết kế slide" - hiển thị thế nào
@dataclass
class VisualizationConfig:
    """Cấu hình cho visualization"""

    # Plot style
    style: str = "seaborn-v0_8-darkgrid"

    # DPI (độ phân giải)
    dpi: int = 300

    # Figure size
    default_figsize: Tuple[int, int] = (14, 5)


# ==================== MASTER CONFIG ====================
# Giống như "menu chính" - tất cả config ở 1 nơi
@dataclass
class Config:
    """Config tổng hợp cho toàn bộ project"""

    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_args(cls, **kwargs) -> "Config":
        """Tạo config từ CLI arguments"""
        config = cls()

        # Data args
        if "data_path" in kwargs:
            config.data.data_path = kwargs["data_path"]
        if "timeframe" in kwargs:
            config.data.timeframe = kwargs["timeframe"]
        if "limit" in kwargs:
            config.data.limit = kwargs["limit"]
        if "refresh_cache" in kwargs:
            config.data.refresh_cache = kwargs["refresh_cache"]
        if "features" in kwargs:
            config.data.features = kwargs["features"]

        # Preprocessing args
        if "window" in kwargs:
            config.preprocessing.window_size = kwargs["window"]

        # Model args
        if "lstm_units" in kwargs:
            config.model.lstm_units = kwargs["lstm_units"]
        if "dropout" in kwargs:
            config.model.dropout_rate = kwargs["dropout"]

        # Training args
        if "epochs" in kwargs:
            config.training.epochs = kwargs["epochs"]
        if "batch_size" in kwargs:
            config.training.batch_size = kwargs["batch_size"]

        # Runtime args
        if "intra_threads" in kwargs:
            config.runtime.intra_op_threads = kwargs["intra_threads"]
        if "seed" in kwargs:
            config.runtime.seed = kwargs["seed"]

        return config

    def summary(self) -> str:
        """In tóm tắt config"""
        lines = [
            "=" * 70,
            "CONFIG SUMMARY",
            "=" * 70,
            "",
            "DATA:",
            f"  File: {self.data.get_data_file()}",
            f"  Timeframe: {self.data.timeframe}",
            f"  Limit: {self.data.limit} lines",
            f"  Features: {self.data.features}",
            f"  Refresh cache: {self.data.refresh_cache}",
            "",
            "PREPROCESSING:",
            f"  Window size: {self.preprocessing.window_size}",
            f"  Scaler: {self.preprocessing.scaler_type}",
            f"  Train/Val/Test: {self.preprocessing.train_ratio:.0%}/{self.preprocessing.val_ratio:.0%}/{(1-self.preprocessing.train_ratio-self.preprocessing.val_ratio):.0%}",
            "",
            "MODEL:",
            f"  LSTM units: {self.model.lstm_units}",
            f"  Dropout: {self.model.dropout_rate}",
            f"  Dense units: {self.model.dense_units}",
            "",
            "TRAINING:",
            f"  Epochs: {self.training.epochs}",
            f"  Batch size: {self.training.batch_size}",
            f"  Learning rate: {self.training.learning_rate}",
            "",
            "RUNTIME:",
            f"  Intra-op threads: {self.runtime.intra_op_threads}",
            f"  Inter-op threads: {self.runtime.inter_op_threads}",
            f"  XLA: {self.runtime.enable_xla}",
            f"  Seed: {self.runtime.seed}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


# ==================== PRESET CONFIGS ====================
# Giống như "combo menu" - config có sẵn cho từng mục đích
# TẬP TRUNG VÀO 15m TIMEFRAME (280K DÒNG DATA)

# ==================== SCALPING PRESETS (15m - Ngắn hạn cực nhanh) ====================
def get_scalping_ultra_fast_config() -> Config:
    """Scalping siêu nhanh - cho 15m, dự đoán 6 tiếng tới"""
    config = Config()
    config.data.limit = 10000
    config.preprocessing.window_size = 24  # 6 tiếng 15m
    config.training.epochs = 5
    config.model.lstm_units = [16]
    config.model.dense_units = []
    config.runtime.intra_op_threads = 4
    return config


def get_scalping_fast_config() -> Config:
    """Scalping nhanh - cho 15m, dự đoán 12 tiếng tới"""
    config = Config()
    config.data.limit = 20000
    config.preprocessing.window_size = 48  # 12 tiếng 15m
    config.training.epochs = 10
    config.model.lstm_units = [32, 16]
    config.model.dense_units = [16]
    config.runtime.intra_op_threads = 6
    return config


# ==================== INTRADAY PRESETS (15m - Ngắn hạn) ====================
def get_intraday_light_config() -> Config:
    """Intraday nhẹ - cho 15m, dự đoán 1 ngày tới"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 96  # 1 ngày 15m
    config.training.epochs = 15
    config.training.early_stopping_patience = 5
    config.model.lstm_units = [32, 16]
    config.model.dense_units = [16]
    config.runtime.intra_op_threads = 8
    return config


def get_intraday_balanced_config() -> Config:
    """Intraday cân bằng - cho 15m, dự đoán 1.5 ngày tới"""
    config = Config()
    config.data.limit = 50000
    config.preprocessing.window_size = 144  # 1.5 ngày 15m
    config.training.epochs = 25
    config.training.early_stopping_patience = 8
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 10
    return config


# ==================== SWING PRESETS (15m - Trung hạn) ====================
def get_swing_fast_config() -> Config:
    """Swing nhanh - cho 15m, dự đoán 2.5 ngày tới"""
    config = Config()
    config.data.limit = 70000
    config.preprocessing.window_size = 240  # 2.5 ngày 15m
    config.training.epochs = 30
    config.training.early_stopping_patience = 10
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 12
    return config


def get_swing_balanced_config() -> Config:
    """Swing cân bằng - cho 15m, dự đoán 4 ngày tới"""
    config = Config()
    config.data.limit = 100000
    config.preprocessing.window_size = 384  # 4 ngày 15m
    config.training.epochs = 50
    config.training.early_stopping_patience = 12
    config.model.lstm_units = [128, 64, 32]
    config.model.dense_units = [64, 32]
    config.runtime.intra_op_threads = 12
    return config


# ==================== LONG-TERM PRESETS (15m - Dài hạn) ====================
def get_long_term_config() -> Config:
    """Long-term - cho 15m, dự đoán 6 ngày tới"""
    config = Config()
    config.data.limit = 150000
    config.preprocessing.window_size = 576  # 6 ngày 15m
    config.training.epochs = 80
    config.training.early_stopping_patience = 15
    config.model.lstm_units = [256, 128, 64, 32]
    config.model.dense_units = [128, 64]
    config.runtime.intra_op_threads = 12
    return config


# ==================== PRODUCTION PRESETS (15m - Chất lượng cao nhất) ====================
def get_production_config() -> Config:
    """Production - cho 15m, dự đoán 8 ngày tới, chất lượng cao nhất"""
    config = Config()
    config.data.limit = 200000
    config.preprocessing.window_size = 768  # 8 ngày 15m
    config.training.epochs = 100
    config.training.early_stopping_patience = 20
    config.model.lstm_units = [256, 128, 64, 32]
    config.model.dense_units = [128, 64, 32]
    config.runtime.intra_op_threads = 12
    return config


# ==================== PRESETS 30K DATASET (15m - Window size từ ngắn đến dài hạn) ====================
# Tất cả preset này dùng dataset 30000 với intra_op_threads=12
# Window size trải dài từ ngắn hạn đến dài hạn để test

def get_30k_w24_config() -> Config:
    """30k dataset - Window 24 (6h) - Ngắn hạn cực nhanh"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 24  # 6 tiếng 15m
    config.training.epochs = 15
    config.training.early_stopping_patience = 5
    config.model.lstm_units = [32, 16]
    config.model.dense_units = [16]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w48_config() -> Config:
    """30k dataset - Window 48 (12h) - Ngắn hạn nhanh"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 48  # 12 tiếng 15m
    config.training.epochs = 15
    config.training.early_stopping_patience = 5
    config.model.lstm_units = [32, 16]
    config.model.dense_units = [16]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w72_config() -> Config:
    """30k dataset - Window 72 (18h) - Ngắn hạn"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 72  # 18 tiếng 15m
    config.training.epochs = 20
    config.training.early_stopping_patience = 6
    config.model.lstm_units = [32, 16]
    config.model.dense_units = [16]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w96_config() -> Config:
    """30k dataset - Window 96 (1 ngày) - Ngắn hạn cân bằng"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 96  # 1 ngày 15m
    config.training.epochs = 20
    config.training.early_stopping_patience = 6
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w144_config() -> Config:
    """30k dataset - Window 144 (1.5 ngày) - Trung hạn ngắn"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 144  # 1.5 ngày 15m
    config.training.epochs = 25
    config.training.early_stopping_patience = 8
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w192_config() -> Config:
    """30k dataset - Window 192 (2 ngày) - Trung hạn"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 192  # 2 ngày 15m
    config.training.epochs = 25
    config.training.early_stopping_patience = 8
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w240_config() -> Config:
    """30k dataset - Window 240 (2.5 ngày) - Trung hạn cân bằng"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 240  # 2.5 ngày 15m
    config.training.epochs = 30
    config.training.early_stopping_patience = 10
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w336_config() -> Config:
    """30k dataset - Window 336 (3.5 ngày) - Trung hạn dài"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 336  # 3.5 ngày 15m
    config.training.epochs = 30
    config.training.early_stopping_patience = 10
    config.model.lstm_units = [128, 64, 32]
    config.model.dense_units = [64, 32]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w480_config() -> Config:
    """30k dataset - Window 480 (5 ngày) - Dài hạn ngắn"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 480  # 5 ngày 15m
    config.training.epochs = 40
    config.training.early_stopping_patience = 12
    config.model.lstm_units = [128, 64, 32]
    config.model.dense_units = [64, 32]
    config.runtime.intra_op_threads = 12
    return config


def get_30k_w672_config() -> Config:
    """30k dataset - Window 672 (7 ngày) - Dài hạn"""
    config = Config()
    config.data.limit = 30000
    config.preprocessing.window_size = 672  # 7 ngày 15m
    config.training.epochs = 40
    config.training.early_stopping_patience = 12
    config.model.lstm_units = [128, 64, 32]
    config.model.dense_units = [64, 32]
    config.runtime.intra_op_threads = 12
    return config


# ==================== LEGACY PRESETS (Cho các timeframe khác) ====================
def get_default_config() -> Config:
    """Config mặc định - dành cho 15m timeframe (default)"""
    return Config()


def get_fast_config() -> Config:
    """Config nhanh - dùng cho test/development với 15m"""
    config = Config()
    config.data.limit = 20000
    config.preprocessing.window_size = 48
    config.training.epochs = 10
    config.model.lstm_units = [32, 16]
    config.runtime.intra_op_threads = 6
    return config


def get_1h_light_config() -> Config:
    """1h nhẹ - dùng cho testing với 1h timeframe"""
    config = Config()
    config.data.timeframe = "1h"  # Set timeframe cho 1h
    config.data.limit = 10000
    config.preprocessing.window_size = 48  # 2 ngày 1h
    config.training.epochs = 15
    config.model.lstm_units = [32, 16]
    config.model.dense_units = [16]
    config.runtime.intra_op_threads = 8
    return config


def get_4h_balanced_config() -> Config:
    """4h cân bằng - dùng cho 4h timeframe"""
    config = Config()
    config.data.timeframe = "4h"  # Set timeframe cho 4h
    config.data.limit = 2000
    config.preprocessing.window_size = 24  # 4 ngày 4h
    config.training.epochs = 30
    config.training.early_stopping_patience = 8
    config.model.lstm_units = [64, 32]
    config.model.dense_units = [32]
    config.runtime.intra_op_threads = 10
    return config
