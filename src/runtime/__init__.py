"""
RUNTIME MODULE - CẤU HÌNH RUNTIME
------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "cấu hình máy tính" - chạy như thế nào
- Tối ưu cho CPU AMD (tăng performance)
- Suppress warnings để output sạch

Trách nhiệm (SoC - Separation of Concerns):
- Chỉ cấu hình runtime, không làm gì khác
"""

from .tensorflow import (
    configure_tensorflow_runtime,
    set_random_seed,
    get_gpu_info,
    set_memory_growth,
    print_tensorflow_info
)

__all__ = [
    "configure_tensorflow_runtime",
    "set_random_seed",
    "get_gpu_info",
    "set_memory_growth",
    "print_tensorflow_info",
]
