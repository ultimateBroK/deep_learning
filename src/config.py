"""
⚙️ CONFIG TẬP TRUNG - CENTRALIZED CONFIGURATION
--------------------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "menu nhà hàng" - tất cả lựa chọn ở 1 nơi
- Thay đổi ở đây → thay đổi toàn bộ project
- Không cần sửa code ở nhiều nơi (DRY - Don't Repeat Yourself)

Ví dụ:
- Muốn thay số epochs → sửa ở đây, không sửa trong main.py, notebook, test...
- Muốn thay timeframe → sửa ở đây, tất cả tự động cập nhật
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


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
    timeframe: str = "1d"  # 1d hoặc 4h

    # Giới hạn dữ liệu
    limit: int = 1500  # Lấy N dòng cuối (<=0 = lấy tất cả)

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

        if tf == "4h":
            return paths.data_dir / "btc_4h_data_2018_to_2025.csv"
        return paths.data_dir / "btc_1d_data_2018_to_2025.csv"


# ==================== PREPROCESSING CONFIG ====================
# Giống như "công thức chế biến" - xử lý dữ liệu thế nào
@dataclass
class PreprocessingConfig:
    """Cấu hình cho tiền xử lý dữ liệu"""

    # Sliding Window
    window_size: int = 60  # Số nến nhìn lại
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
    dense_units: List[int] = field(default_factory=lambda: [16])

    # Output
    output_units: int = 1

    def get_input_shape(self, window_size: int, n_features: int) -> tuple:
        """Lấy shape đầu vào cho model"""
        return (window_size, n_features)


# ==================== TRAINING CONFIG ====================
# Giống như "lịch học tập" - học thế nào
@dataclass
class TrainingConfig:
    """Cấu hình cho training"""

    # Training parameters
    epochs: int = 20
    batch_size: int = 32

    # Early stopping
    early_stopping_patience: int = 5

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
    default_figsize: tuple = (14, 5)


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
            "⚙️  CONFIG SUMMARY",
            "=" * 70,
            "",
            "📁 DATA:",
            f"  File: {self.data.get_data_file()}",
            f"  Timeframe: {self.data.timeframe}",
            f"  Limit: {self.data.limit} lines",
            f"  Features: {self.data.features}",
            f"  Refresh cache: {self.data.refresh_cache}",
            "",
            "🔧 PREPROCESSING:",
            f"  Window size: {self.preprocessing.window_size}",
            f"  Scaler: {self.preprocessing.scaler_type}",
            f"  Train/Val/Test: {self.preprocessing.train_ratio:.0%}/{self.preprocessing.val_ratio:.0%}/{(1-self.preprocessing.train_ratio-self.preprocessing.val_ratio):.0%}",
            "",
            "🧠 MODEL:",
            f"  LSTM units: {self.model.lstm_units}",
            f"  Dropout: {self.model.dropout_rate}",
            f"  Dense units: {self.model.dense_units}",
            "",
            "🏋️  TRAINING:",
            f"  Epochs: {self.training.epochs}",
            f"  Batch size: {self.training.batch_size}",
            f"  Learning rate: {self.training.learning_rate}",
            "",
            "⚡ RUNTIME:",
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
def get_default_config() -> Config:
    """Config mặc định - cân bằng giữa tốc độ và chất lượng"""
    return Config()


def get_fast_config() -> Config:
    """Config nhanh - dùng cho test/development"""
    config = Config()
    config.data.limit = 500
    config.preprocessing.window_size = 30
    config.training.epochs = 5
    config.model.lstm_units = [32, 16]
    return config


def get_high_quality_config() -> Config:
    """Config chất lượng cao - dùng cho production"""
    config = Config()
    config.data.limit = 3000
    config.preprocessing.window_size = 90
    config.training.epochs = 50
    config.training.early_stopping_patience = 10
    config.model.lstm_units = [128, 64, 32]
    config.model.dense_units = [64, 32]
    return config
