#!/usr/bin/env python3
"""
🎯 ENTRY POINT: CHAY PROJECT CLI
---------------------------------

Giải thích:
- File này là "cửa chính" để chạy toàn bộ project
- Chạy từ terminal với các tham số
- Tự động chạy qua tất cả các bước

Cách dùng:
    python main.py --epochs 20 --limit 1500
"""

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))

from step1_data import fetch_binance_data
from step2_preprocessing import prepare_data_for_lstm
from step3_model import build_bilstm_model, print_model_summary
from step4_training import train_model, evaluate_model, print_sample_predictions, calculate_direction_accuracy
from step5_visualization import plot_training_history, plot_predictions, plot_all_in_one
from utils import configure_tensorflow_runtime, print_tensorflow_info, create_results_folder, save_markdown_report, save_config, save_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Dự báo giá Bitcoin với BiLSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --epochs 20 --limit 1500
  python main.py --timeframe 4h --window 30
  python main.py --refresh-cache
        """
    )
    
    # Data args
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Cặp giao dịch (mặc định: BTC/USDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1d',
        choices=['1d', '4h', '1h'],
        help='Khung thời gian (mặc định: 1d)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1500,
        help='Số nến lấy từ Binance (mặc định: 1500)'
    )
    parser.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Tải lại dữ liệu từ Binance (không dùng cache)'
    )
    
    # Preprocessing args
    parser.add_argument(
        '--window',
        type=int,
        default=60,
        help='Số nến nhìn lại (mặc định: 60)'
    )
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=['close'],
        help='Features sử dụng (mặc định: close)'
    )
    
    # Model args
    parser.add_argument(
        '--lstm-units',
        type=int,
        nargs='+',
        default=[64, 32],
        help='Số units cho mỗi LSTM layer (mặc định: 64 32)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (mặc định: 0.2)'
    )
    
    # Training args
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Số epochs (mặc định: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (mặc định: 32)'
    )
    
    # Runtime args
    parser.add_argument(
        '--intra-threads',
        type=int,
        default=12,
        help='CPU threads cho operations (mặc định: 12)'
    )
    
    return parser.parse_args()


def main():
    """Hàm chính để chạy project"""
    # Parse args
    args = parse_args()
    
    print("\n" + "="*70)
    print(" " * 15 + "DỰ BÁO GIÁ BITCOIN VỚI BiLSTM")
    print("="*70)
    
    # Cấu hình TensorFlow
    configure_tensorflow_runtime(
        intra_op_threads=args.intra_threads,
        inter_op_threads=2,
        enable_xla=True
    )
    print_tensorflow_info()
    
    # ========================================
    # BƯỚC 1: LẤY DỮ LIỆU
    # ========================================
    print("\n" + "="*70)
    print("BƯỚC 1: LẤY DỮ LIỆU TỪ BINANCE")
    print("="*70 + "\n")
    
    df = fetch_binance_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        save_cache=not args.refresh_cache
    )
    
    # ========================================
    # BƯỚC 2: XỬ LÝ DỮ LIỆU
    # ========================================
    print("\n" + "="*70)
    print("BƯỚC 2: XỬ LÝ DỮ LIỆU")
    print("="*70 + "\n")
    
    data_dict = prepare_data_for_lstm(
        df=df,
        features=args.features,
        window_size=args.window,
        scaler_type='minmax'
    )
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    scaler = data_dict['scaler']
    
    # ========================================
    # BƯỚC 3: XÂY DỰNG MODEL
    # ========================================
    print("\n" + "="*70)
    print("BƯỚC 3: XÂY DỰNG MODEL BiLSTM")
    print("="*70 + "\n")
    
    input_shape = (args.window, len(args.features))
    model = build_bilstm_model(
        input_shape=input_shape,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        dense_units=[16],
        output_units=1
    )
    print_model_summary(model)
    
    # ========================================
    # BƯỚC 4: TRAINING
    # ========================================
    print("\n" + "="*70)
    print("BƯỚC 4: TRAINING MODEL")
    print("="*70 + "\n")
    
    train_result = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=5
    )
    
    history = train_result['history']
    
    # ========================================
    # BƯỚC 5: ĐÁNH GIÁ & VẼ BIỂU ĐỒ
    # ========================================
    print("\n" + "="*70)
    print("BƯỚC 5: ĐÁNH GIÁ & VẼ BIỂU ĐỒ")
    print("="*70 + "\n")
    
    # Đánh giá trên test set
    eval_result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        return_predictions=True
    )
    
    y_true = eval_result['y_true']
    y_pred = eval_result['predictions']
    
    # In một số ví dụ dự đoán
    print_sample_predictions(y_true, y_pred, n_samples=10)
    
    # Tính độ chính xác xu hướng
    calculate_direction_accuracy(y_true, y_pred)
    
    # ========================================
    # LƯU KẾT QUẢ
    # ========================================
    print("\n" + "="*70)
    print("LƯU KẾT QUẢ")
    print("="*70 + "\n")
    
    # Tạo folder kết quả
    results_folder = create_results_folder(run_type="main")
    print(f"\n📁 Folder kết quả: {results_folder}\n")
    
    # Vẽ và lưu biểu đồ
    timestamp_suffix = results_folder.name.replace('BiLSTM_', '')
    
    plot_history_file = results_folder / f"training_history_{timestamp_suffix}.png"
    plot_predictions_file = results_folder / f"predictions_{timestamp_suffix}.png"
    plot_all_in_one_file = results_folder / f"all_in_one_{timestamp_suffix}.png"
    
    plot_training_history(history, save_path=str(plot_history_file))
    plot_predictions(y_true, y_pred, save_path=str(plot_predictions_file))
    plot_all_in_one(history, y_true, y_pred, save_path=str(plot_all_in_one_file))
    
    # Lưu báo cáo
    config_dict = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'limit': args.limit,
        'window_size': args.window,
        'features': args.features,
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'intra_threads': args.intra_threads
    }
    
    plots_dict = {
        'training_history': timestamp_suffix,
        'predictions': timestamp_suffix,
        'all_in_one': timestamp_suffix
    }
    
    save_markdown_report(
        folder_path=results_folder,
        config=config_dict,
        metrics=eval_result,
        history=history.history,
        plots=plots_dict
    )
    save_config(results_folder, config_dict)
    save_metrics(results_folder, eval_result)
    
    # ========================================
    # HOÀN THÀNH
    # ========================================
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"📊 Báo cáo: {results_folder / f'results_{timestamp_suffix}.md'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
