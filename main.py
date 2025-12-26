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
import re

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))

# Lưu ý: không import các module "nặng" ở top-level để `python main.py --help` chạy gọn và nhanh.


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
        '--data-path',
        type=str,
        default=None,
        help='Đường dẫn file CSV (nếu bỏ trống sẽ chọn theo --timeframe trong thư mục data/)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1d',
        choices=['1d', '4h'],
        help='Timeframe (dùng để chọn file mặc định nếu không set --data-path) (mặc định: 1d)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1500,
        help='Lấy N dòng cuối trong file CSV (mặc định: 1500, <=0 = lấy tất cả)'
    )
    parser.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Đọc lại từ CSV gốc (bỏ qua cache đã chuẩn hoá)'
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

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Cố định ngẫu nhiên để tái lập kết quả (mặc định: 42, <0 = không set)'
    )
    
    return parser.parse_args()


def _infer_timeframe_from_filename(path_str: str | None) -> str | None:
    """
    Infer timeframe dựa vào tên file, ví dụ:
    - btc_1d_data_2018_to_2025.csv -> 1d
    - btc_4h_data_2018_to_2025.csv -> 4h
    """
    if not path_str:
        return None
    name = Path(path_str).name.lower()
    if re.search(r"(?:^|_)4h(?:_|\\.)", name) or "4h" in name:
        return "4h"
    if re.search(r"(?:^|_)1d(?:_|\\.)", name) or "1d" in name:
        return "1d"
    return None


def _default_data_path_from_timeframe(timeframe: str) -> str:
    tf = (timeframe or "1d").lower()
    base = Path(__file__).parent / "data"
    if tf == "4h":
        return str(base / "btc_4h_data_2018_to_2025.csv")
    return str(base / "btc_1d_data_2018_to_2025.csv")


def main():
    """Hàm chính để chạy project"""
    # Parse args
    args = parse_args()

    # Import các module "nặng" sau khi parse args để:
    # - `python main.py --help` chạy nhanh và không in log TensorFlow
    from step1_data import fetch_binance_data
    from step2_preprocessing import prepare_data_for_lstm
    from step3_model import build_bilstm_model, print_model_summary
    from step4_training import (
        train_model,
        evaluate_model,
        print_sample_predictions,
        calculate_direction_accuracy,
    )
    from step5_visualization import plot_training_history, plot_predictions, plot_all_in_one
    from utils import (
        configure_tensorflow_runtime,
        print_tensorflow_info,
        create_results_folder,
        save_markdown_report,
        save_config,
        save_metrics,
        set_random_seed,
    )
    
    print("\n" + "="*70)
    print(" " * 15 + "DỰ BÁO GIÁ BITCOIN VỚI BiLSTM")
    print("="*70)
    
    # Cấu hình TensorFlow
    set_random_seed(args.seed, deterministic=True)
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
    print("BƯỚC 1: ĐỌC DỮ LIỆU CSV (LOCAL)")
    print("="*70 + "\n")

    data_path = args.data_path or _default_data_path_from_timeframe(args.timeframe)
    inferred_tf = _infer_timeframe_from_filename(data_path)
    effective_tf = inferred_tf or args.timeframe
    print(f"📄 Data file: {data_path}")
    print(f"🕒 Timeframe (từ tên file): {effective_tf}\n")
    
    df = fetch_binance_data(
        data_path=data_path,
        timeframe=effective_tf,
        limit=args.limit,
        save_cache=not args.refresh_cache
    )

    # Thông tin dữ liệu (đưa vào report)
    data_rows = len(df)
    try:
        data_start = str(df["datetime"].iloc[0])
        data_end = str(df["datetime"].iloc[-1])
    except Exception:
        data_start, data_end = None, None
    
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

    # Thông tin split (đưa vào report)
    train_samples = len(X_train)
    val_samples = len(X_val)
    test_samples = len(X_test)
    scaler_type = "minmax"
    
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
    best_epoch = train_result.get("best_epoch")
    best_val_loss = train_result.get("best_val_loss")
    train_seconds = train_result.get("train_seconds")
    checkpoint_path = str(train_result.get("checkpoint_path")) if train_result.get("checkpoint_path") is not None else None
    
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
    direction_accuracy = calculate_direction_accuracy(y_true, y_pred)
    eval_result["direction_accuracy"] = float(direction_accuracy)
    
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
        'data_path': data_path,
        'timeframe': effective_tf,
        'limit': args.limit,
        'data_rows': data_rows,
        'data_start': data_start,
        'data_end': data_end,
        'window_size': args.window,
        'features': args.features,
        'scaler_type': scaler_type,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'intra_threads': args.intra_threads,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_seconds': train_seconds,
        'checkpoint_path': checkpoint_path,
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
    print(f"📊 Báo cáo: {results_folder / f'results_BiLSTM_{timestamp_suffix}.md'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
