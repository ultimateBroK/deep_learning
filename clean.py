#!/usr/bin/env python3
"""
🧹 CLEAN SCRIPT - DỌN DẸP PROJECT
------------------------------------

Giải thích:
- Dùng để xóa các file cache, checkpoint, báo cáo cũ
- Giúp project gọn gàng, tiết kiệm disk space

Cách dùng:
    python clean.py                    # Dọn tất cả
    python clean.py --cache             # Chỉ dọn cache
    python clean.py --reports --keep 10 # Giữ lại 10 báo cáo mới nhất
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Dọn dẹp project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Xóa cache dữ liệu'
    )
    parser.add_argument(
        '--reports',
        action='store_true',
        help='Xóa báo cáo cũ'
    )
    parser.add_argument(
        '--keep-reports',
        type=int,
        default=5,
        help='Số báo cáo mới nhất cần giữ (mặc định: 5)'
    )
    parser.add_argument(
        '--data-cache',
        action='store_true',
        help='Xóa cache dữ liệu cũ (> 30 ngày)'
    )
    parser.add_argument(
        '--data-cache-force',
        action='store_true',
        help='Xóa TẤT CẢ cache dữ liệu'
    )
    parser.add_argument(
        '--checkpoints',
        action='store_true',
        help='Xóa checkpoints model'
    )
    
    return parser.parse_args()


def clean_data_cache(force: bool = False):
    """Xóa cache dữ liệu"""
    cache_dir = Path(__file__).parent / "step1_data" / "cache"
    
    if not cache_dir.exists():
        print("✅ Không có thư mục cache")
        return 0
    
    deleted_count = 0
    current_time = datetime.now().timestamp()
    max_age_days = 30
    
    for file_path in cache_dir.glob("*.csv"):
        if force:
            file_path.unlink()
            deleted_count += 1
        else:
            file_age_days = (current_time - file_path.stat().st_mtime) / 86400
            if file_age_days > max_age_days:
                file_path.unlink()
                deleted_count += 1
    
    if deleted_count > 0:
        print(f"🗑️  Đã xóa {deleted_count} file cache dữ liệu")
    else:
        print("✅ Không có file cache dữ liệu nào để xóa")
    
    return deleted_count


def clean_reports(keep: int = 5):
    """Xóa báo cáo cũ, chỉ giữ lại `keep` folder mới nhất"""
    reports_dir = Path(__file__).parent / "reports"
    
    if not reports_dir.exists():
        print("✅ Không có thư mục reports")
        return 0
    
    deleted_count = 0
    
    # Duyệt qua các thư mục con (main, notebook)
    for run_type_dir in reports_dir.iterdir():
        if not run_type_dir.is_dir():
            continue
        
        # Lấy danh sách các folder kết quả, sắp xếp theo thời gian giảm dần
        result_folders = sorted(
            run_type_dir.glob("BiLSTM_*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Xóa các folder cũ hơn `keep`
        for folder in result_folders[keep:]:
            shutil.rmtree(folder)
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"🗑️  Đã xóa {deleted_count} báo cáo cũ (giữ lại {keep} mới nhất)")
    else:
        print("✅ Không có báo cáo nào để xóa")
    
    return deleted_count


def clean_checkpoints():
    """Xóa checkpoints model"""
    checkpoint_dir = Path(__file__).parent / "reports" / "checkpoints"
    
    if not checkpoint_dir.exists():
        print("✅ Không có thư mục checkpoints")
        return 0
    
    deleted_count = 0
    
    for file_path in checkpoint_dir.glob("*.keras"):
        if "best" not in file_path.name.lower():
            file_path.unlink()
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"🗑️  Đã xóa {deleted_count} checkpoint")
    else:
        print("✅ Không có checkpoint nào để xóa")
    
    return deleted_count


def main():
    """Hàm chính"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("🧹 DỌN DẸP PROJECT")
    print("="*60 + "\n")
    
    total_deleted = 0
    
    # Nếu không có tham số nào, dọn tất cả
    if not any([args.cache, args.reports, args.data_cache, 
                args.data_cache_force, args.checkpoints]):
        print("🔧 Dọn dẹp tất cả...\n")
        total_deleted += clean_data_cache(force=True)
        total_deleted += clean_reports(keep=args.keep_reports)
        total_deleted += clean_checkpoints()
    else:
        # Xóa cache dữ liệu (force)
        if args.cache:
            print("🔧 Dọn cache dữ liệu...\n")
            total_deleted += clean_data_cache(force=True)
        
        # Xóa cache dữ liệu (chỉ file cũ)
        if args.data_cache:
            print("🔧 Dọn cache dữ liệu cũ (> 30 ngày)...\n")
            total_deleted += clean_data_cache(force=False)
        
        # Xóa TẤT CẢ cache dữ liệu
        if args.data_cache_force:
            print("🔧 XÓA TẤT CẢ cache dữ liệu...\n")
            total_deleted += clean_data_cache(force=True)
        
        # Xóa báo cáo cũ
        if args.reports:
            print(f"🔧 Dọn báo cáo cũ (giữ lại {args.keep_reports})...\n")
            total_deleted += clean_reports(keep=args.keep_reports)
        
        # Xóa checkpoints
        if args.checkpoints:
            print("🔧 Dọn checkpoints...\n")
            total_deleted += clean_checkpoints()
    
    print("\n" + "="*60)
    print(f"✅ Tổng cộng đã xóa {total_deleted} file/folder")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
