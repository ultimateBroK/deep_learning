"""
🎯 SOURCE PACKAGE
------------------

Package chính của project.

Modules:
- config: Cấu hình tập trung (DRY)
- core: Business logic chính (data, preprocessing, model, metrics)
- runtime: TensorFlow runtime configuration
- visualization: Vẽ biểu đồ
- results: Lưu kết quả
- pipeline: Pipeline chính (SoC)

Giải thích bằng ví dụ đời sống:
- Giống như "tầng trệt" của tòa nhà
- Tất cả các phòng (modules) đều ở đây
- CLI và Notebook chỉ cần import từ đây
"""

from .config import (
    Config,
    Paths,
    DataConfig,
    PreprocessingConfig,
    ModelConfig,
    TrainingConfig,
    RuntimeConfig,
    VisualizationConfig,
    # Scalping presets (15m)
    get_scalping_ultra_fast_config,
    get_scalping_fast_config,
    # Intraday presets (15m)
    get_intraday_light_config,
    get_intraday_balanced_config,
    # Swing presets (15m)
    get_swing_fast_config,
    get_swing_balanced_config,
    # Long-term preset (15m)
    get_long_term_config,
    # Production preset (15m)
    get_production_config,
    # 30k dataset presets (15m - Window size từ ngắn đến dài hạn)
    get_30k_w24_config,
    get_30k_w48_config,
    get_30k_w72_config,
    get_30k_w96_config,
    get_30k_w144_config,
    get_30k_w192_config,
    get_30k_w240_config,
    get_30k_w336_config,
    get_30k_w480_config,
    get_30k_w672_config,
    # Legacy presets (other timeframes)
    get_default_config,
    get_fast_config,
    get_1h_light_config,
    get_4h_balanced_config,
)
from .pipeline import run_pipeline

__all__ = [
    # Config classes
    "Config",
    "Paths",
    "DataConfig",
    "PreprocessingConfig",
    "ModelConfig",
    "TrainingConfig",
    "RuntimeConfig",
    "VisualizationConfig",
    # Scalping presets (15m)
    "get_scalping_ultra_fast_config",
    "get_scalping_fast_config",
    # Intraday presets (15m)
    "get_intraday_light_config",
    "get_intraday_balanced_config",
    # Swing presets (15m)
    "get_swing_fast_config",
    "get_swing_balanced_config",
    # Long-term preset (15m)
    "get_long_term_config",
    # Production preset (15m)
    "get_production_config",
    # 30k dataset presets (15m - Window size từ ngắn đến dài hạn)
    "get_30k_w24_config",
    "get_30k_w48_config",
    "get_30k_w72_config",
    "get_30k_w96_config",
    "get_30k_w144_config",
    "get_30k_w192_config",
    "get_30k_w240_config",
    "get_30k_w336_config",
    "get_30k_w480_config",
    "get_30k_w672_config",
    # Legacy presets (other timeframes)
    "get_default_config",
    "get_fast_config",
    "get_1h_light_config",
    "get_4h_balanced_config",
    # Pipeline
    "run_pipeline",
]
