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
import tensorflow as tf


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
    # Cấu hình số threads
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    
    # Bật XLA optimization
    if enable_xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    # Cấu hình CPU affinity (chỉ chạy trên CPU)
    tf.config.set_visible_devices([], 'GPU')
    
    # In thông tin cấu hình
    print(f"{'='*60}")
    print(f"⚙️  CẤU HÌNH TENSORFLOW RUNTIME")
    print(f"{'='*60}")
    print(f"Intra-op threads: {intra_op_threads}")
    print(f"Inter-op threads: {inter_op_threads}")
    print(f"XLA enabled: {enable_xla}")
    print(f"CPU only: True")
    print(f"{'='*60}\n")


def get_gpu_info():
    """
    Kiểm tra GPU có sẵn không
    
    Returns:
        True nếu có GPU, False nếu không
    """
    gpus = tf.config.list_physical_devices('GPU')
    
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
    print(f"\n{'='*60}")
    print(f"📋 THÔNG TIN TENSORFLOW")
    print(f"{'='*60}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {get_gpu_info()}")
    
    # CPU threads
    print(f"Intra-op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(f"Inter-op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test functions
    configure_tensorflow_runtime()
    print_tensorflow_info()
