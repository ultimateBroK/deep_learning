"""
PIPELINE MODULE - PIPELINE CHÍNH
------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "assembly line" trong nhà máy
- Mỗi công đoạn làm 1 việc riêng biệt (SoC - Separation of Concerns)
- Data → Preprocessing → Model → Training → Evaluation → Results

Flow:
1. STEP 1: Load data
2. STEP 2: Preprocess data
3. STEP 3: Build model
4. STEP 4: Train model
5. STEP 5: Evaluate & visualize
6. STEP 6: Save results

Trách nhiệm (SoC):
- Orchestrate toàn bộ process
- Không chứa logic cụ thể (logic ở các module khác)
"""

from typing import Dict, Optional

# Import từ các module khác
from .config import Config
from .runtime import configure_tensorflow_runtime, print_tensorflow_info, set_random_seed
from .core import (
    fetch_binance_data,
    prepare_data_for_lstm,
    build_bilstm_model,
    print_model_summary,
    evaluate_model,
    print_sample_predictions,
    calculate_direction_accuracy,
)
from .training import train_model
from .visualization import (
    plot_training_history,
    plot_predictions,
    plot_all_in_one,
)
from .results import (
    create_results_folder,
    save_config,
    save_metrics,
    save_markdown_report,
)


# ==================== MAIN PIPELINE ====================
def run_pipeline(config: Optional[Config] = None, run_type: str = "main") -> Dict:
    """
    Chạy toàn bộ pipeline

    Giải thích bằng ví dụ đời sống:
    - Giống như "điều phối viên" - điều phối toàn bộ quy trình
    - Mỗi step gọi đúng module chuyên biệt (SoC)
    - Config từ một nơi (DRY)

    Args:
        config: Cấu hình (None = dùng default)
        run_type: Loại chạy ("main", "notebook", "test")

    Returns:
        Dictionary chứa tất cả kết quả
    """
    # 1. Setup config
    if config is None:
        config = Config()

    # 2. Setup runtime
    set_random_seed(config.runtime.seed, deterministic=True)
    configure_tensorflow_runtime(
        intra_op_threads=config.runtime.intra_op_threads,
        inter_op_threads=config.runtime.inter_op_threads,
        enable_xla=config.runtime.enable_xla,
        use_gpu=config.runtime.use_gpu
    )
    print_tensorflow_info()

    # 3. STEP 1: LOAD DATA
    print("\n" + "=" * 70)
    print("BƯỚC 1: LOAD DỮ LIỆU")
    print("=" * 70 + "\n")

    data_file = config.data.get_data_file()
    print(f"Data file: {data_file}")
    print(f"Timeframe: {config.data.timeframe}\n")

    df = fetch_binance_data(
        data_path=str(data_file),
        timeframe=config.data.timeframe,
        limit=config.data.limit,
        save_cache=not config.data.refresh_cache,
        cache_dir=config.paths.cache_dir
    )

    data_rows = len(df)
    try:
        if data_rows > 0:
            data_start = str(df.select("datetime").row(0)[0])
            data_end = str(df.select("datetime").row(-1)[0])
        else:
            data_start, data_end = None, None
    except Exception:
        data_start, data_end = None, None

    # 4. STEP 2: PREPROCESS DATA
    print("\n" + "=" * 70)
    print("BƯỚC 2: XỬ LÝ DỮ LIỆU")
    print("=" * 70 + "\n")

    data_dict = prepare_data_for_lstm(
        df=df,
        features=config.data.features,
        window_size=config.preprocessing.window_size,
        scaler_type=config.preprocessing.scaler_type,
        train_ratio=config.preprocessing.train_ratio,
        val_ratio=config.preprocessing.val_ratio
    )

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    scaler = data_dict['scaler']

    # 5. STEP 3: BUILD MODEL
    print("\n" + "=" * 70)
    print("BƯỚC 3: XÂY DỰNG MODEL BiLSTM")
    print("=" * 70 + "\n")

    input_shape = config.model.get_input_shape(
        config.preprocessing.window_size,
        len(config.data.features)
    )
    model = build_bilstm_model(
        input_shape=input_shape,
        lstm_units=config.model.lstm_units,
        dropout_rate=config.model.dropout_rate,
        dense_units=config.model.dense_units,
        output_units=config.model.output_units,
        learning_rate=config.training.learning_rate
    )
    print_model_summary(model)

    # 6. STEP 4: TRAIN MODEL
    print("\n" + "=" * 70)
    print("BƯỚC 4: TRAINING MODEL")
    print("=" * 70 + "\n")

    train_result = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config
    )

    history = train_result['history']

    # 7. STEP 5: EVALUATE & VISUALIZE
    print("\n" + "=" * 70)
    print("BƯỚC 5: ĐÁNH GIÁ & VẼ BIỂU ĐỒ")
    print("=" * 70 + "\n")

    eval_result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        return_predictions=True
    )

    y_true = eval_result['y_true']
    y_pred = eval_result['predictions']

    print_sample_predictions(y_true, y_pred, n_samples=10)

    direction_accuracy = calculate_direction_accuracy(y_true, y_pred)
    eval_result["direction_accuracy"] = float(direction_accuracy)

    # 8. STEP 6: SAVE RESULTS
    print("\n" + "=" * 70)
    print("LƯU KẾT QUẢ")
    print("=" * 70 + "\n")

    # Tạo config dict để đặt tên folder (đồng bộ với notebook: chỉ dùng 3 tham số cơ bản)
    folder_config = {
        'timeframe': config.data.timeframe,
        'window_size': config.preprocessing.window_size,
        'limit': config.data.limit,
    }
    results_folder = create_results_folder(run_type=run_type, config=folder_config)
    print(f"\nFolder kết quả: {results_folder}\n")

    # Vẽ và lưu biểu đồ
    # Lấy timestamp từ tên folder (2 phần cuối: YYYYMMDD_HHMMSS)
    folder_parts = results_folder.name.split('_')
    timestamp_suffix = '_'.join(folder_parts[-2:])  # Lấy 2 phần cuối

    plot_history_file = results_folder / f"training_history_{timestamp_suffix}.png"
    plot_predictions_file = results_folder / f"predictions_{timestamp_suffix}.png"
    plot_all_in_one_file = results_folder / f"all_in_one_{timestamp_suffix}.png"

    plot_training_history(history, save_path=str(plot_history_file))
    plot_predictions(y_true, y_pred, save_path=str(plot_predictions_file))
    plot_all_in_one(history, y_true, y_pred, save_path=str(plot_all_in_one_file))

    # Tạo config dict để lưu
    config_dict = {
        'data_path': str(data_file),
        'symbol': 'BTC/USDT',  # Default symbol
        'timeframe': config.data.timeframe,
        'limit': config.data.limit,
        'data_rows': data_rows,
        'data_start': data_start,
        'data_end': data_end,
        'window_size': config.preprocessing.window_size,
        'features': config.data.features,
        'scaler_type': config.preprocessing.scaler_type,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'seed': config.runtime.seed,
        'lstm_units': config.model.lstm_units,
        'dropout_rate': config.model.dropout_rate,
        'epochs': config.training.epochs,
        'batch_size': config.training.batch_size,
        'early_stopping_patience': config.training.early_stopping_patience,
        'learning_rate': config.training.learning_rate,
        'intra_threads': config.runtime.intra_op_threads,
        'best_epoch': train_result['best_epoch'],
        'best_val_loss': train_result['best_val_loss'],
        'train_seconds': train_result['train_seconds'],
        'checkpoint_path': train_result['checkpoint_path'],
    }

    plots_dict = {
        'training_history': f"training_history_{timestamp_suffix}.png",
        'predictions': f"predictions_{timestamp_suffix}.png",
        'all_in_one': f"all_in_one_{timestamp_suffix}.png"
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

    # 9. SUMMARY
    print("\n" + "=" * 70)
    print("HOÀN THÀNH")
    print("=" * 70)
    print(f"Báo cáo: {results_folder / f'results_BiLSTM_{timestamp_suffix}.md'}")
    print("=" * 70 + "\n")

    return {
        "config": config_dict,
        "metrics": eval_result,
        "history": history.history,
        "plots": plots_dict,
        "results_folder": str(results_folder),
    }


if __name__ == "__main__":
    # Test pipeline
    config = Config.from_args(limit=100, epochs=2)
    results = run_pipeline(config)
    print("Pipeline completed!")
