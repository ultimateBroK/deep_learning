"""
BƯỚC 1: LẤY DỮ LIỆU TỪ BINANCE
---------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như bạn lên website xem lịch sử giá Bitcoin
- Binance là một cái "kho dữ liệu" chứa tất cả giá giao dịch crypto
- Chúng ta sẽ kéo dữ liệu về máy để phân tích
"""

import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import ccxt


def fetch_binance_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1500,
    save_cache: bool = True,
    cache_dir: str = None
) -> pd.DataFrame:
    """
    Lấy dữ liệu giá từ Binance
    
    Args:
        symbol: Cặp giao dịch (BTC/USDT, ETH/USDT, v.v.)
        timeframe: Khung thời gian (1d = 1 ngày, 4h = 4 giờ, 1h = 1 giờ)
        limit: Số lượng nến (candles) muốn lấy
        save_cache: Có lưu vào cache không (để lần sau không phải tải lại)
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
    
    # Tên file cache dựa trên symbol, timeframe, limit
    cache_filename = f"{symbol.replace('/', '_')}_{timeframe}_{limit}.csv"
    cache_path = cache_dir / cache_filename
    
    # Nếu cache đã tồn tại và save_cache=True, đọc từ cache
    if save_cache and cache_path.exists():
        print(f"📂 Đang đọc dữ liệu từ cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    # Tạo client Binance (không cần API key cho public data)
    print(f"📥 Đang tải dữ liệu từ Binance: {symbol}, timeframe: {timeframe}, limit: {limit}")
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Tự động điều chỉnh tốc độ request
    })
    
    # Lấy dữ liệu OHLCV (Open, High, Low, Close, Volume)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # Chuyển sang DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Chuyển timestamp sang datetime (dễ đọc hơn)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Sắp xếp theo thời gian tăng dần
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Chỉ giữ lại các cột cần thiết
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    # Lưu vào cache nếu save_cache=True
    if save_cache:
        df.to_csv(cache_path, index=False)
        print(f"💾 Đã lưu cache vào: {cache_path}")
    
    print(f"✅ Đã tải {len(df)} dòng dữ liệu")
    print(f"   Thời gian: {df['datetime'].iloc[0]} đến {df['datetime'].iloc[-1]}")
    
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
    df = fetch_binance_data(symbol="BTC/USDT", timeframe="1d", limit=100)
    print(df.head())
