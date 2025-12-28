"""
CLI MODULE - COMMAND LINE INTERFACE
-----------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "receptionist" - tiếp nhận yêu cầu từ user
- User nhập câu lệnh → CLI hiểu → Gọi pipeline
- Simple, focused (KISS - Keep It Simple, Stupid)

Trách nhiệm (SoC):
- Chỉ parse arguments và gọi pipeline
- Không chứa business logic
"""

from .main import main

__all__ = ["main"]
