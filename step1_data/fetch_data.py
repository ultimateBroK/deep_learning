"""
BƯỚC 1: ĐỌC DỮ LIỆU TỪ FILE CSV (LOCAL)
---------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như bạn đã tải sẵn một file lịch sử giá Bitcoin về máy
- Thay vì gọi API (CCXT/Binance), ta đọc trực tiếp file CSV
- Sau đó chuẩn hoá cột để pipeline phía sau dùng thống nhất
"""

from datetime import datetime
from pathlib import Path
import polars as pl


def _project_root() -> Path:
    # step1_data/ nằm ngay dưới root project
    return Path(__file__).resolve().parents[1]


def _default_data_path(timeframe: str) -> Path:
    """
    Chọn file data mặc định theo timeframe.
    - 1d  -> data/btc_1d_data_2018_to_2025.csv
    - 4h  -> data/btc_4h_data_2018_to_2025.csv
    """
    data_dir = _project_root() / "data"
    tf = (timeframe or "1d").lower()
    if tf == "4h":
        return data_dir / "btc_4h_data_2018_to_2025.csv"
    return data_dir / "btc_1d_data_2018_to_2025.csv"


def _infer_timeframe_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    if "4h" in name:
        return "4h"
    if "1d" in name:
        return "1d"
    return None


def _normalize_binance_export_csv(df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Chuẩn hoá CSV kiểu "Binance export" về schema thống nhất:
    datetime/open/high/low/close/volume
    """
    if df_raw is None or len(df_raw) == 0:
        return pl.DataFrame(
            schema=["datetime", "open", "high", "low", "close", "volume"]
        )

    # Map cột (CSV của bạn có format: Open time, Open, High, Low, Close, Volume, ...)
    required = ["Open time", "Open", "High", "Low", "Close", "Volume"]
    columns = df_raw.columns
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"CSV thiếu cột bắt buộc: {missing}. "
            f"Hiện có: {list(columns)}"
        )

    # Tạo DataFrame mới với schema chuẩn
    # Strip khoảng trắng ở cột Open time trước khi parse
    df_raw = df_raw.with_columns([
        pl.col("Open time").str.strip_chars()
    ])

    # Kiểm tra xem cột Open time có chứa " UTC" không bằng cách lấy 1 mẫu đầu tiên
    first_sample = df_raw.select(pl.col("Open time")).row(0)[0]
    has_utc = " UTC" in first_sample

    # Parse datetime theo format phù hợp
    if has_utc:
        df_parsed = df_raw.with_columns([
            pl.col("Open time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f UTC", strict=False)
        ])
    else:
        df_parsed = df_raw.with_columns([
            pl.col("Open time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        ])

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
    data_path: str | None = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1500,
    save_cache: bool = True,
    cache_dir: str = None
) -> pl.DataFrame:
    """
    Đọc dữ liệu giá từ file CSV local (mặc định: `data/btc_1d_data_2018_to_2025.csv`)
    
    Args:
        data_path: Đường dẫn CSV. Nếu None -> chọn mặc định theo timeframe.
        symbol: (DEPRECATED) giữ lại để tương thích notebook cũ, không còn dùng để fetch API.
        timeframe: Dùng để chọn file mặc định khi data_path=None (1d/4h).
        limit: Lấy N dòng cuối (None hoặc <=0 -> lấy toàn bộ).
        save_cache: Có lưu cache (CSV đã chuẩn hoá) để lần sau đọc nhanh hơn không.
        cache_dir: Thư mục cache (mặc định: step1_data/cache)
    
    Returns:
        DataFrame với các cột: open, high, low, close, volume, datetime
    """
    # Xác định thư mục cache
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Xác định file dữ liệu
    if data_path is None:
        data_file = _default_data_path(timeframe)
    else:
        data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file data: {data_file}")

    inferred_tf = _infer_timeframe_from_filename(data_file) or (timeframe or "1d")

    # Tên file cache dựa trên file data + timeframe + limit
    stem = data_file.stem
    lim = int(limit) if isinstance(limit, int) else limit
    cache_filename = f"{stem}_{inferred_tf}_{lim if lim and lim > 0 else 'all'}.normalized.csv"
    cache_path = cache_dir / cache_filename
    
    # Nếu cache đã tồn tại và save_cache=True, đọc từ cache
    if save_cache and cache_path.exists():
        print(f"📂 Đang đọc dữ liệu từ cache: {cache_path}")
        df = pl.read_csv(cache_path, try_parse_dates=True)
        return df

    print(f"📥 Đang đọc dữ liệu từ CSV: {data_file}")
    print(f"🕒 Timeframe (từ tên file): {inferred_tf}")
    if symbol and symbol != "BTC/USDT":
        # Chỉ cảnh báo nhẹ để không làm hỏng notebook cũ
        print(f"ℹ️  (Bỏ qua) symbol={symbol} — hiện đang dùng dữ liệu từ file CSV local.")

    raw = pl.read_csv(data_file)
    df = _normalize_binance_export_csv(raw)

    if len(df) == 0:
        raise ValueError(
            f"DataFrame rỗng sau khi normalize. Có thể do format datetime không hợp lệ "
            f"hoặc tất cả dòng bị filter. Vui lòng kiểm tra file: {data_file}"
        )

    if isinstance(limit, int) and limit > 0 and len(df) > limit:
        df = df.tail(limit)

    # Lưu vào cache nếu save_cache=True
    if save_cache:
        df.write_csv(cache_path)
        print(f"💾 Đã lưu cache vào: {cache_path}")

    print(f"✅ Đã tải {len(df)} dòng dữ liệu")
    print(f"   Thời gian: {df.select('datetime').row(0)[0]} đến {df.select('datetime').row(-1)[0]}")
    
    return df


def clear_cache(cache_dir: str = None, older_than_days: int = None) -> int:
    """
    Xóa cache dữ liệu
    
    Args:
        cache_dir: Thư mục cache
        older_than_days: Chỉ xóa file cũ hơn số ngày này (None = xóa tất cả)
    
    Returns:
        Số file đã xóa
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    else:
        cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return 0
    
    deleted_count = 0
    current_time = datetime.now().timestamp()
    
    for file_path in cache_dir.glob("*.csv"):
        if older_than_days is None:
            # Xóa tất cả
            file_path.unlink()
            deleted_count += 1
        else:
            # Chỉ xóa file cũ hơn số ngày quy định
            file_age_days = (current_time - file_path.stat().st_mtime) / 86400
            if file_age_days > older_than_days:
                file_path.unlink()
                deleted_count += 1
    
    if deleted_count > 0:
        print(f"🗑️  Đã xóa {deleted_count} file cache")
    else:
        print("✅ Không có file cache nào để xóa")
    
    return deleted_count


if __name__ == "__main__":
    # Test function
    df = fetch_binance_data(timeframe="1d", limit=100)
    print(df.head())
