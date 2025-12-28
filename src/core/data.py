"""
DATA MODULE - ĐỌC DỮ LIỆU
--------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "nhân viên kho" - chịu trách nhiệm lấy dữ liệu
- Dữ liệu có sẵn trong file CSV, không cần gọi API
- Chuẩn hoá để các bộ phận khác dễ sử dụng

Trách nhiệm (SoC - Separation of Concerns):
- Đọc file CSV
- Chuẩn hoá format
- Cache để lần sau đọc nhanh
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl


def _infer_timeframe_from_filename(path: Path) -> Optional[str]:
    """
    Infer timeframe từ tên file
    Ví dụ: btc_15m_data_2018_to_2025.csv → 15m
            btc_4h_data_2018_to_2025.csv → 4h
            btc_1d_data_2018_to_2025.csv → 1d
    """
    name = path.name.lower()
    # Thử theo thứ tự từ dài đến ngắn để tránh ghi đè không chính xác
    if "15m" in name:
        return "15m"
    if "1h" in name:
        return "1h"
    if "4h" in name:
        return "4h"
    if "1d" in name:
        return "1d"
    return None


def _normalize_binance_export_csv(df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Chuẩn hoá CSV kiểu "Binance export" về schema thống nhất:
    datetime/open/high/low/close/volume

    Giải thích: Giống như "đóng gói" - đưa tất cả về cùng format
    """
    if df_raw is None or len(df_raw) == 0:
        return pl.DataFrame(
            schema=["datetime", "open", "high", "low", "close", "volume"]
        )

    # Kiểm tra cột bắt buộc
    required = ["Open time", "Open", "High", "Low", "Close", "Volume"]
    columns = df_raw.columns
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"CSV thiếu cột bắt buộc: {missing}. "
            f"Hiện có: {list(columns)}"
        )

    # Strip khoảng trắng và parse datetime
    df_raw = df_raw.with_columns([
        pl.col("Open time").str.strip_chars()
    ])

    first_sample = df_raw.select(pl.col("Open time")).row(0)[0]
    has_utc = " UTC" in first_sample

    if has_utc:
        df_parsed = df_raw.with_columns([
            pl.col("Open time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f UTC", strict=False)
        ])
    else:
        df_parsed = df_raw.with_columns([
            pl.col("Open time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        ])

    # Chuẩn hoá cột
    out = (
        df_parsed
        .rename({
            "Open time": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        .with_columns([
            pl.col("datetime").dt.replace_time_zone(None),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64)
        ])
        .filter(pl.col("datetime").is_not_null() & pl.col("close").is_not_null())
        .sort("datetime")
    )

    return out.select(["datetime", "open", "high", "low", "close", "volume"])


def fetch_binance_data(
    data_path: Optional[str] = None,
    data_dir: Optional[Path] = None,
    timeframe: str = "15m",
    limit: int = 50000,
    save_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> pl.DataFrame:
    """
    Đọc dữ liệu giá từ file CSV local

    Args:
        data_path: Đường dẫn CSV. Nếu None → chọn theo timeframe
        data_dir: Thư mục chứa file data (mặc định: project/data/)
        timeframe: Timeframe để chọn file (1d/4h)
        limit: Lấy N dòng cuối (<=0 → lấy tất cả)
        save_cache: Lưu cache để lần sau đọc nhanh hơn
        cache_dir: Thư mục cache

    Returns:
        DataFrame với các cột: datetime, open, high, low, close, volume
    """
    # Xác định thư mục data và cache
    from ..config import Paths  # noqa: E402 - Import here to avoid circular dependency

    if data_dir is None:
        data_dir = Paths().data_dir
    else:
        data_dir = Path(data_dir)

    if cache_dir is None:
        cache_dir = Paths().cache_dir
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Xác định file dữ liệu
    if data_path is None:
        tf = (timeframe or "15m").lower()
        # Chọn file tương ứng với timeframe
        if tf == "15m":
            data_file = data_dir / "btc_15m_data_2018_to_2025.csv"
        elif tf == "1h":
            data_file = data_dir / "btc_1h_data_2018_to_2025.csv"
        elif tf == "4h":
            data_file = data_dir / "btc_4h_data_2018_to_2025.csv"
        else:
            data_file = data_dir / "btc_1d_data_2018_to_2025.csv"
    else:
        data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file data: {data_file}")

    inferred_tf = _infer_timeframe_from_filename(data_file) or (timeframe or "15m")

    # Tên file cache
    stem = data_file.stem
    lim = int(limit) if isinstance(limit, int) else limit
    cache_filename = f"{stem}_{inferred_tf}_{lim if lim and lim > 0 else 'all'}.normalized.csv"
    cache_path = cache_dir / cache_filename

    # Đọc từ cache nếu có
    if save_cache and cache_path.exists():
        print(f"Đang đọc dữ liệu từ cache: {cache_path}")
        df = pl.read_csv(cache_path, try_parse_dates=True)
        return df

    print(f"Đang đọc dữ liệu từ CSV: {data_file}")
    print(f"Timeframe (từ tên file): {inferred_tf}")

    raw = pl.read_csv(data_file)
    df = _normalize_binance_export_csv(raw)

    if len(df) == 0:
        raise ValueError(
            f"DataFrame rỗng sau khi normalize. "
            f"Vui lòng kiểm tra file: {data_file}"
        )

    if isinstance(limit, int) and limit > 0 and len(df) > limit:
        df = df.tail(limit)

    # Lưu cache
    if save_cache:
        df.write_csv(cache_path)
        print(f"Đã lưu cache vào: {cache_path}")

    print(f"Đã tải {len(df)} dòng dữ liệu")
    try:
        print(f"   Thời gian: {df.select('datetime').row(0)[0]} đến {df.select('datetime').row(-1)[0]}")
    except Exception:
        pass

    return df


def clear_cache(cache_dir: Optional[Path] = None, older_than_days: Optional[int] = None) -> int:
    """
    Xóa cache dữ liệu

    Args:
        cache_dir: Thư mục cache
        older_than_days: Chỉ xóa file cũ hơn số ngày này (None = xóa tất cả)

    Returns:
        Số file đã xóa
    """
    from ..config import Paths  # noqa: E402 - Import here to avoid circular dependency

    if cache_dir is None:
        cache_dir = Paths().cache_dir
    else:
        cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return 0

    deleted_count = 0
    current_time = datetime.now().timestamp()

    for file_path in cache_dir.glob("*.csv"):
        if older_than_days is None:
            file_path.unlink()
            deleted_count += 1
        else:
            file_age_days = (current_time - file_path.stat().st_mtime) / 86400
            if file_age_days > older_than_days:
                file_path.unlink()
                deleted_count += 1

    if deleted_count > 0:
        print(f"Đã xóa {deleted_count} file cache")
    else:
        print("Không có file cache nào để xóa")

    return deleted_count


if __name__ == "__main__":
    # Test
    df = fetch_binance_data(timeframe="1d", limit=100)
    print(df.head())
