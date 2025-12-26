"""
UTILS: CẤU HÌNH RUNTIME TENSORFLOW
------------------------------------

Giải thích bằng ví dụ đời sống:
- TensorFlow có nhiều cách chạy (CPU, GPU, TPU)
- Với CPU AMD, cần cấu hình số threads để tối ưu
- Giống như bạn có 12 nhân CPU, nên tận dụng hết

Lưu ý:
- intra_op_parallelism_threads: Số thread cho các operations song song
- inter_op_parallelism_threads: Số thread cho các operations song song khác
- enable_xla: Tối ưu code bằng XLA (Accelerated Linear Algebra)
"""

import os
import random
import warnings

import numpy as np

# Suppress warnings trước khi import TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Chỉ hiển thị ERROR và WARNING nghiêm trọng
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt oneDNN warnings

# Suppress FutureWarning về np.object
warnings.filterwarnings('ignore', category=FutureWarning, message='.*np.object.*')
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*CUDA.*')

import tensorflow as tf

# Suppress TensorFlow warnings sau khi import
tf.get_logger().setLevel('ERROR')


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """
    Cố định ngẫu nhiên để kết quả chạy có thể tái lập.

    Args:
        seed: Số seed (ví dụ 42). Nếu seed < 0 thì không làm gì.
        deterministic: Cố gắng bật deterministic ops (best-effort, tuỳ môi trường/TF version).
    """
    if seed is None or seed < 0:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)

    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def configure_tensorflow_runtime(
    intra_op_threads: int = 12,
    inter_op_threads: int = 2,
    enable_xla: bool = True
):
    """
    Cấu hình runtime TensorFlow cho CPU AMD
    
    Args:
        intra_op_threads: Số thread cho operations song song (số core vật lý)
        inter_op_threads: Số thread cho operations song song khác
        enable_xla: Bật XLA optimization
    """
    # Suppress warnings khi cấu hình
    old_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Cấu hình số threads
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        
        # Bật XLA optimization
        if enable_xla:
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        
        # Chỉ chạy trên CPU (nếu có GPU thì ẩn đi; nếu không có GPU thì bỏ qua)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            # Không để lỗi runtime nhỏ làm hỏng toàn pipeline
            pass  # Suppress error message
    finally:
        tf.get_logger().setLevel(old_level)
    
    # In thông tin cấu hình
    print("=" * 60)
    print("⚙️  CẤU HÌNH TENSORFLOW RUNTIME")
    print("=" * 60)
    print(f"Intra-op threads: {intra_op_threads}")
    print(f"Inter-op threads: {inter_op_threads}")
    print(f"XLA enabled: {enable_xla}")
    print("CPU only: True")
    print("=" * 60 + "\n")


def get_gpu_info():
    """
    Kiểm tra GPU có sẵn không
    
    Returns:
        True nếu có GPU, False nếu không
    """
    # Suppress warnings khi check GPU
    old_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
    finally:
        tf.get_logger().setLevel(old_level)
    
    if gpus:
        print(f"✅ Tìm thấy {len(gpus)} GPU:")
        for gpu in gpus:
            print(f"   - {gpu.name}")
        return True
    else:
        print("ℹ️  Không tìm thấy GPU, sẽ dùng CPU")
        return False


def set_memory_growth():
    """
    Cho phép GPU tự tăng bộ nhớ khi cần (tránh chiếm hết VRAM)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Đã bật memory growth cho GPU")
        except RuntimeError as e:
            print(f"❌ Lỗi khi cấu hình GPU: {e}")


def print_tensorflow_info():
    """
    In thông tin về TensorFlow và runtime
    """
    print("\n" + "=" * 60)
    print("📋 THÔNG TIN TENSORFLOW")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {get_gpu_info()}")
    
    # CPU threads
    print(f"Intra-op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(f"Inter-op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test functions
    configure_tensorflow_runtime()
    print_tensorflow_info()
