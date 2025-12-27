#!/usr/bin/env python3
"""
🧹 CLEAN SCRIPT - DỌN DẸP PROJECT
-------------------------------------

Giải thích bằng ví dụ đời sống:
- Giống như "dọn dẹp phòng" - xóa bớt file không cần
- Project nhẹ hơn, dễ quản lý hơn

Usage:
    python -m scripts.clean                    # Xem trước (dry-run)
    python -m scripts.clean --execute          # Thực sự xóa
    python -m scripts.clean --cache --days 7   # Xóa cache > 7 ngày
    python -m scripts.clean --reports --keep 3 # Giữ lại 3 báo cáo
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Base directory - từ vị trí file này
BASE = Path(__file__).resolve().parent.parent


def get_size_mb(p: Path) -> float:
    """Lấy kích thước file/folder (MB)"""
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / (1024 * 1024)


def get_age_days(p: Path) -> float:
    """Lấy tuổi file/folder (ngày)"""
    return (datetime.now().timestamp() - p.stat().st_mtime) / 86400


def print_header(title: str, rel_path: Path):
    """In header cho từng loại dọn dẹp"""
    print(f"\n{'='*60}")
    print(f"📁 {title} ({rel_path})")
    print(f"{'='*60}")


def print_item(p: Path, size_mb: float, age_days: float, indent: str = "  "):
    """In thông tin item cần xóa"""
    print(f"{indent}- {p.name if p.is_file() else str(p.relative_to(BASE)) + '/'}")
    print(f"{indent}    Size: {size_mb:.2f} MB | Age: {age_days:.1f} ngày")


def print_summary(count: int, size_mb: float, item_type: str = "items", dry_run: bool = True):
    """In tổng kết"""
    print("=" * 60)
    print(f"🗑️  Tổng: {count} {item_type}, {size_mb:.2f} MB")
    if dry_run:
        print("💡 DRY-RUN: Sử dụng --execute để thực sự xóa")
    print("=" * 60 + "\n")


def clean_cache(*, force: bool = False, older_than_days: int = 30, dry_run: bool = True) -> tuple[int, float]:
    """Xóa cache dữ liệu"""
    from src.config import Paths
    cache_dir = Paths().cache_dir

    if not cache_dir.exists():
        print("✅ Không có thư mục cache")
        return 0, 0.0

    candidates: list[Path] = []
    for file_path in cache_dir.glob("*.csv"):
        age_days = get_age_days(file_path)
        if force or age_days > older_than_days:
            candidates.append(file_path)

    if not candidates:
        print("✅ Không có file cache nào để xóa")
        return 0, 0.0

    print_header("CACHE DATA", cache_dir.relative_to(BASE))

    removed_count = 0
    total_size_mb = 0.0

    for file_path in sorted(candidates):
        age_days = get_age_days(file_path)
        size_mb = get_size_mb(file_path)
        print_item(file_path, size_mb, age_days)

        if not dry_run:
            try:
                file_path.unlink()
            except Exception as e:
                print(f"      ⚠️  Lỗi: {e}")
                continue

        removed_count += 1
        total_size_mb += size_mb

    print_summary(removed_count, total_size_mb, "files", dry_run)
    return removed_count, total_size_mb


def clean_reports(*, keep: int = 5, dry_run: bool = True) -> tuple[int, float]:
    """Xóa báo cáo cũ"""
    reports_dir = BASE / "reports"

    if not reports_dir.exists():
        print("✅ Không có thư mục reports")
        return 0, 0.0

    removed_count = 0
    total_size_mb = 0.0
    folders_to_remove = []

    for run_type_dir in reports_dir.iterdir():
        if not run_type_dir.is_dir():
            continue

        result_folders = sorted(
            run_type_dir.glob("BiLSTM_*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for folder in result_folders[keep:]:
            folders_to_remove.append(folder)
            removed_count += 1

    if removed_count > 0:
        print_header("REPORTS", reports_dir.relative_to(BASE))

        for folder in sorted(folders_to_remove):
            size_mb = get_size_mb(folder)
            age_days = get_age_days(folder)
            print_item(folder, size_mb, age_days)

            if not dry_run:
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    print(f"      ⚠️  Lỗi: {e}")
                    removed_count -= 1
                    total_size_mb -= size_mb
                    continue

            total_size_mb += size_mb

        print_summary(removed_count, total_size_mb, "folders", dry_run)
    else:
        print("✅ Không có báo cáo nào để xóa")

    return removed_count, total_size_mb


def clean_checkpoints(*, keep_best: bool = True, dry_run: bool = True) -> tuple[int, float]:
    """Xóa checkpoints"""
    checkpoint_dir = BASE / "models" / "checkpoints"

    if not checkpoint_dir.exists():
        print("✅ Không có thư mục checkpoints")
        return 0, 0.0

    removed_count = 0
    total_size_mb = 0.0
    files_to_remove = []

    for file_path in checkpoint_dir.glob("*.keras"):
        if keep_best and "best" in file_path.name.lower():
            continue

        files_to_remove.append(file_path)
        removed_count += 1

    if removed_count > 0:
        print_header("CHECKPOINTS", checkpoint_dir.relative_to(BASE))

        for file_path in sorted(files_to_remove):
            size_mb = get_size_mb(file_path)
            age_days = get_age_days(file_path)
            print_item(file_path, size_mb, age_days)

            if not dry_run:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"      ⚠️  Lỗi: {e}")
                    removed_count -= 1
                    total_size_mb -= size_mb
                    continue

            total_size_mb += size_mb

        print_summary(removed_count, total_size_mb, "files", dry_run)
    else:
        print("✅ Không có checkpoint nào để xóa")

    return removed_count, total_size_mb


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Dọn dẹp project: cache, reports, checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m scripts.clean                    # Xem trước (dry-run)
  python -m scripts.clean --execute          # Thực sự xóa
  python -m scripts.clean --cache --days 7   # Xóa cache > 7 ngày
  python -m scripts.clean --reports --keep 3 # Giữ lại 3 báo cáo
        """
    )

    parser.add_argument('--execute', '-e', action='store_true', help='Thực sự xóa (mặc định: dry-run)')
    parser.add_argument('--all', '-a', action='store_true', help='Dọn tất cả loại (cache cũ + reports cũ + checkpoints)')

    # Cache flags
    parser.add_argument('--cache', action='store_true', help='Xóa TẤT CẢ cache')
    parser.add_argument('--data-cache', action='store_true', help='Xóa cache cũ (> --days)')
    parser.add_argument('--data-cache-force', action='store_true', help='Xóa TẤT CẢ cache dữ liệu')
    parser.add_argument('--days', type=int, default=30, help='Số ngày cache cũ (mặc định: 30)')

    # Reports flags
    parser.add_argument('--reports', action='store_true', help='Xóa báo cáo cũ')
    parser.add_argument('--keep', '-k', type=int, default=5, help='Giữ lại N báo cáo (mặc định: 5)')
    parser.add_argument('--keep-reports', type=int, help='Alias của --keep')

    # Checkpoints
    parser.add_argument('--checkpoints', action='store_true', help='Xóa checkpoints')
    parser.add_argument('--no-keep-best', action='store_true', help='Xóa cả checkpoint "best"')

    return parser.parse_args()


def main():
    """Hàm chính"""
    args = parse_args()
    dry_run = not args.execute

    do_all = args.all or not any([
        args.cache, args.data_cache, args.data_cache_force, args.reports, args.checkpoints, args.keep_reports
    ])

    if args.keep_reports is not None:
        args.keep = args.keep_reports

    print("\n" + "="*60)
    print("🧹 DỌN DẸP PROJECT")
    print("="*60)
    print(f"Mode: {'DRY-RUN (chỉ xem)' if dry_run else 'EXECUTE (thực sự xóa)'}")
    if dry_run:
        print("💡 Sử dụng --execute để thực sự xóa")
    print("="*60 + "\n")

    total_count = 0
    total_size_mb = 0.0

    # Cache
    if do_all or args.cache or args.data_cache or args.data_cache_force:
        force = args.cache or args.data_cache_force
        count, size = clean_cache(force=force, older_than_days=args.days, dry_run=dry_run)
        total_count += count
        total_size_mb += size

    # Reports
    if do_all or args.reports:
        count, size = clean_reports(keep=args.keep, dry_run=dry_run)
        total_count += count
        total_size_mb += size

    # Checkpoints
    if do_all or args.checkpoints:
        count, size = clean_checkpoints(keep_best=not args.no_keep_best, dry_run=dry_run)
        total_count += count
        total_size_mb += size

    # Tổng kết
    print("\n" + "="*60)
    print("📊 TỔNG KẾT QUẢ")
    print("="*60)
    print(f"Total: {total_count} items, {total_size_mb:.2f} MB")

    if total_size_mb > 100:
        print(f"       (~{total_size_mb / 1024:.2f} GB)")

    if dry_run and total_count > 0:
        print("\n💡 Sử dụng --execute để thực sự xóa")
    elif total_count == 0:
        print("\n✅ Không có gì để xóa!")
    else:
        print(f"\n✅ Đã xóa {total_count} items!")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
