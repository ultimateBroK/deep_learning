"""
🎯 ENTRY POINT - CLI MAIN
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
import sys
from pathlib import Path

# Thêm src vào path để import được
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import Config, run_pipeline


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
    data_group = parser.add_argument_group("📥 Data", "Dữ liệu đầu vào")

    data_group.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Đường dẫn file CSV (nếu bỏ trống sẽ chọn theo --timeframe)'
    )
    data_group.add_argument(
        '--timeframe',
        type=str,
        default='1d',
        choices=['1d', '4h'],
        help='Timeframe (mặc định: 1d)'
    )
    data_group.add_argument(
        '--limit',
        type=int,
        default=1500,
        help='Lấy N dòng cuối trong file CSV (mặc định: 1500, <=0 = lấy tất cả)'
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
        default=['close'],
        help='Features sử dụng (mặc định: close)'
    )

    # ==================== PREPROCESSING ARGS ====================
    prep_group = parser.add_argument_group("🔧 Preprocessing", "Xử lý dữ liệu")

    prep_group.add_argument(
        '--window',
        type=int,
        default=60,
        help='Số nến nhìn lại (mặc định: 60)'
    )

    # ==================== MODEL ARGS ====================
    model_group = parser.add_argument_group("🧠 Model", "Cấu hình model")

    model_group.add_argument(
        '--lstm-units',
        type=int,
        nargs='+',
        default=[64, 32],
        help='Số units cho mỗi LSTM layer (mặc định: 64 32)'
    )
    model_group.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (mặc định: 0.2)'
    )

    # ==================== TRAINING ARGS ====================
    train_group = parser.add_argument_group("🏋️ Training", "Huấn luyện model")

    train_group.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Số epochs (mặc định: 20)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (mặc định: 32)'
    )

    # ==================== RUNTIME ARGS ====================
    runtime_group = parser.add_argument_group("⚡ Runtime", "Cấu hình runtime")

    runtime_group.add_argument(
        '--intra-threads',
        type=int,
        default=12,
        help='CPU threads cho operations (mặc định: 12)'
    )
    runtime_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Cố định ngẫu nhiên để tái lập kết quả (mặc định: 42, <0 = không set)'
    )

    # ==================== PRESET ====================
    preset_group = parser.add_argument_group("📦 Preset", "Cấu hình có sẵn")

    preset_group.add_argument(
        '--preset',
        type=str,
        choices=['default', 'fast', 'high-quality'],
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
    if args.preset == 'fast':
        config = Config.from_args(
            limit=500,
            window=30,
            epochs=5,
            lstm_units=[32, 16],
            intra_threads=6,
            seed=args.seed
        )
    elif args.preset == 'high-quality':
        config = Config.from_args(
            limit=3000,
            window=90,
            epochs=50,
            lstm_units=[128, 64, 32],
            intra_threads=args.intra_threads,
            seed=args.seed
        )
    else:
        config = Config()

    # Override config với CLI args
    if args.data_path:
        config.data.data_path = args.data_path
    if args.timeframe:
        config.data.timeframe = args.timeframe
    if args.limit:
        config.data.limit = args.limit
    if args.refresh_cache:
        config.data.refresh_cache = args.refresh_cache
    if args.features:
        config.data.features = args.features
    if args.window:
        config.preprocessing.window_size = args.window
    if args.lstm_units:
        config.model.lstm_units = args.lstm_units
    if args.dropout:
        config.model.dropout_rate = args.dropout
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.intra_threads:
        config.runtime.intra_op_threads = args.intra_threads
    if args.seed:
        config.runtime.seed = args.seed

    # In header
    print("\n" + "=" * 70)
    print(" " * 15 + "DỰ BÁO GIÁ BITCOIN VỚI BiLSTM")
    print("=" * 70)
    print(f"📦 Preset: {args.preset}")
    print("=" * 70)

    # In config summary
    print(config.summary())

    # Chạy pipeline
    run_pipeline(config, run_type="cli")


if __name__ == "__main__":
    main()
