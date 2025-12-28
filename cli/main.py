"""
ENTRY POINT - CLI MAIN
---------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "cửa chính" vào nhà
- User mở cửa → CLI chào đón → Gọi pipeline

KISS (Keep It Simple, Stupid):
- Chỉ parse arguments
- Chỉ gọi pipeline
- Không chứa business logic

Usage:
    python -m cli.main --epochs 20 --limit 1500
    python -m cli.main --help
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings TRƯỚC khi import bất kỳ thứ gì
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Chỉ hiển thị ERROR (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt oneDNN warnings

# Suppress Python warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*np.object.*')
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*Could not find cuda.*')
warnings.filterwarnings('ignore', message='.*cuda drivers.*')
warnings.filterwarnings('ignore', message='.*GPU will not be used.*')

# Thêm src vào path để import được
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import và suppress TensorFlow logger ngay lập tức
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    # Suppress stderr output từ TensorFlow
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except ImportError:
    pass  # TensorFlow chưa được cài đặt

from src import Config, run_pipeline, get_default_config, get_fast_config, get_1h_light_config, get_4h_balanced_config, get_scalping_ultra_fast_config, get_scalping_fast_config, get_intraday_light_config, get_intraday_balanced_config, get_swing_fast_config, get_swing_balanced_config, get_long_term_config, get_production_config, get_30k_w24_config, get_30k_w48_config, get_30k_w72_config, get_30k_w96_config, get_30k_w144_config, get_30k_w192_config, get_30k_w240_config, get_30k_w336_config, get_30k_w480_config, get_30k_w672_config   # noqa: E402
from src.core.data import _infer_timeframe_from_filename   # noqa: E402


def parse_args():
    """
    Parse command line arguments

    Giải thích: Giống như "lắng nghe yêu cầu" từ user
    """
    parser = argparse.ArgumentParser(
        description="Dự báo giá Bitcoin với BiLSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m cli.main --epochs 20 --limit 1500
  python -m cli.main --timeframe 4h --window 30
  python -m cli.main --refresh-cache
        """
    )

    # ==================== DATA ARGS ====================
    data_group = parser.add_argument_group("Data", "Dữ liệu đầu vào")

    data_group.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Đường dẫn file CSV (nếu bỏ trống sẽ chọn theo --timeframe)'
    )
    data_group.add_argument(
        '--timeframe',
        type=str,
        default=None,
        choices=['1d', '4h', '1h', '15m'],
        help='Timeframe (mặc định theo preset/config; preset default = 15m)'
    )
    data_group.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Lấy N dòng cuối trong file CSV (mặc định theo preset/config; <=0 = lấy tất cả)'
    )
    data_group.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Đọc lại từ CSV gốc (bỏ qua cache đã chuẩn hoá)'
    )
    data_group.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=None,
        help='Features sử dụng (mặc định: close)'
    )

    # ==================== PREPROCESSING ARGS ====================
    prep_group = parser.add_argument_group("Preprocessing", "Xử lý dữ liệu")

    prep_group.add_argument(
        '--window',
        type=int,
        default=None,
        help='Số nến nhìn lại (mặc định theo preset/config; preset default = 240)'
    )
    prep_group.add_argument(
        '--scaler-type',
        type=str,
        default=None,
        choices=['minmax', 'standard'],
        help='Loại scaler (mặc định: minmax)'
    )

    # ==================== MODEL ARGS ====================
    model_group = parser.add_argument_group("Model", "Cấu hình model")

    model_group.add_argument(
        '--lstm-units',
        type=int,
        nargs='+',
        default=None,
        help='Số units cho mỗi LSTM layer (mặc định: 64 32)'
    )
    model_group.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Dropout rate (mặc định: 0.2)'
    )

    # ==================== TRAINING ARGS ====================
    train_group = parser.add_argument_group("Training", "Huấn luyện model")

    train_group.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Số epochs (mặc định theo preset/config; preset default = 30)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (mặc định: 32)'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (mặc định: 0.001)'
    )
    train_group.add_argument(
        '--early-stopping-patience',
        type=int,
        default=None,
        help='Số epochs chờ trước khi dừng (mặc định theo preset/config; preset default = 10)'
    )

    # ==================== RUNTIME ARGS ====================
    runtime_group = parser.add_argument_group("Runtime", "Cấu hình runtime")

    runtime_group.add_argument(
        '--intra-threads',
        type=int,
        default=None,
        help='CPU threads cho operations trong cùng op (mặc định theo preset/config; default = 12)'
    )
    runtime_group.add_argument(
        '--inter-threads',
        type=int,
        default=None,
        help='CPU threads cho operations khác nhau (mặc định theo preset/config; default = 2)'
    )
    runtime_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Cố định ngẫu nhiên để tái lập kết quả (mặc định theo preset/config; default = 42, <0 = không set)'
    )

    # ==================== PRESET ====================
    preset_group = parser.add_argument_group("Preset", "Cấu hình có sẵn")

    preset_group.add_argument(
        '--preset',
        type=str,
        choices=['default', 'fast', '1h-light', '4h-balanced', 'scalping-ultra-fast', 'scalping-fast', 'intraday-light', 'intraday-balanced', 'swing-fast', 'swing-balanced', 'long-term', 'production', '30k-w24', '30k-w48', '30k-w72', '30k-w96', '30k-w144', '30k-w192', '30k-w240', '30k-w336', '30k-w480', '30k-w672'],
        default='default',
        help='Preset config (mặc định: default)'
    )

    return parser.parse_args()


def main():
    """
    Main entry point

    Giải thích: Giống như "nhân viên lễ tân" - tiếp nhận, chuyển tiếp
    """
    # Parse args
    args = parse_args()

    # Chọn preset
    preset_map = {
        'default': get_default_config,
        'fast': get_fast_config,
        '1h-light': get_1h_light_config,
        '4h-balanced': get_4h_balanced_config,
        'scalping-ultra-fast': get_scalping_ultra_fast_config,
        'scalping-fast': get_scalping_fast_config,
        'intraday-light': get_intraday_light_config,
        'intraday-balanced': get_intraday_balanced_config,
        'swing-fast': get_swing_fast_config,
        'swing-balanced': get_swing_balanced_config,
        'long-term': get_long_term_config,
        'production': get_production_config,
        # 30k dataset presets (15m - fixed limit=30000)
        '30k-w24': get_30k_w24_config,
        '30k-w48': get_30k_w48_config,
        '30k-w72': get_30k_w72_config,
        '30k-w96': get_30k_w96_config,
        '30k-w144': get_30k_w144_config,
        '30k-w192': get_30k_w192_config,
        '30k-w240': get_30k_w240_config,
        '30k-w336': get_30k_w336_config,
        '30k-w480': get_30k_w480_config,
        '30k-w672': get_30k_w672_config,
    }

    config = preset_map[args.preset]()

    # Override config với CLI args (chỉ khi user truyền vào)
    if args.data_path is not None:
        config.data.data_path = args.data_path
        # Infer timeframe từ filename nếu user không chỉ định --timeframe
        if args.timeframe is None:
            inferred_tf = _infer_timeframe_from_filename(Path(args.data_path))
            if inferred_tf:
                config.data.timeframe = inferred_tf

    if args.timeframe is not None:
        config.data.timeframe = args.timeframe
    if args.limit is not None:
        config.data.limit = args.limit
    if args.refresh_cache:
        config.data.refresh_cache = args.refresh_cache
    if args.features is not None:
        config.data.features = args.features
    if args.window is not None:
        config.preprocessing.window_size = args.window
    if args.scaler_type is not None:
        config.preprocessing.scaler_type = args.scaler_type
    if args.lstm_units is not None:
        config.model.lstm_units = args.lstm_units
    if args.dropout is not None:
        config.model.dropout_rate = args.dropout
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience
    if args.intra_threads is not None:
        config.runtime.intra_op_threads = args.intra_threads
    if args.inter_threads is not None:
        config.runtime.inter_op_threads = args.inter_threads
    if args.seed is not None:
        config.runtime.seed = args.seed

    # In header
    print("\n" + "=" * 70)
    print(" " * 15 + "DỰ BÁO GIÁ BITCOIN VỚI BiLSTM")
    print("=" * 70)
    print(f"Preset: {args.preset}")
    print("=" * 70)

    # In config summary
    print(config.summary())

    # Chạy pipeline
    run_pipeline(config, run_type="cli")


if __name__ == "__main__":
    main()
