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
    get_default_config,
    get_fast_config,
    get_high_quality_config,
)
from .pipeline import run_pipeline

__all__ = [
    # Config
    "Config",
    "Paths",
    "DataConfig",
    "PreprocessingConfig",
    "ModelConfig",
    "TrainingConfig",
    "RuntimeConfig",
    "VisualizationConfig",
    "get_default_config",
    "get_fast_config",
    "get_high_quality_config",
    # Pipeline
    "run_pipeline",
]
