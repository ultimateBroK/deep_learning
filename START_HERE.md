# ⭐ ĐỌC FILE NÀY TRƯỚC!

Chào mừng bạn đến với **Mô hình dự báo giá Bitcoin với BiLSTM**!

Được thiết kế đặc biệt cho người ADHD - mọi thứ được chia thành từng bước rõ ràng, có giải thích bằng ví dụ đời sống.

---

## 📋 CHỈ MỤC

- [Quick Start](#-quick-start)
- [Cấu trúc project](#-cấu-trúc-project)
- [Cách chạy](#-cách-chạy)
- [Tài liệu quan trọng](#-tài-liệu-quan-trọng)
- [Tips cho người ADHD](#-tips-cho-người-adhd)
- [Nếu bị lạc](#-nếu-bị-lạc)

---

## 🚀 Quick Start

### Option 1: Chạy Notebook (Khuyến nghị cho người mới)

```bash
# Cài đặt dependencies
uv sync

# Chạy Jupyter Notebook
uv run jupyter notebook
```

Mở file `notebooks/run_complete.ipynb` và chạy từng cell theo thứ tự.

**Notebook có:**
- Markdown giải thích từng bước
- Checklist để đánh dấu tiến độ
- Analogies để dễ hiểu

### Option 2: Chạy CLI (Nhanh hơn)

```bash
# Cài đặt dependencies
uv sync

# Chạy với cấu hình mặc định
uv run python main.py

# Chạy với tham số tùy chỉnh
uv run python main.py --epochs 20 --limit 1500
```

---

## 📁 Cấu trúc Project

```
Deep_learning/
├── START_HERE.md              # ⭐ ĐỌC FILE NÀY TRƯỚC!
│
├── step1_data/                # BƯỚC 1: Lấy dữ liệu
│   ├── fetch_data.py          # Tải dữ liệu từ Binance
│   └── cache/                 # Dữ liệu đã tải (CSV)
│
├── step2_preprocessing/        # BƯỚC 2: Xử lý dữ liệu
│   ├── create_windows.py      # Tạo windows (sequences)
│   └── scaling.py             # Chuẩn hóa dữ liệu
│
├── step3_model/               # BƯỚC 3: Xây dựng model
│   └── bilstm.py               # Model BiLSTM
│
├── step4_training/            # BƯỚC 4: Training
│   ├── train.py               # Hàm train model
│   └── evaluate.py             # Đánh giá kết quả
│
├── step5_visualization/        # BƯỚC 5: Vẽ biểu đồ
│   └── plots.py                # Các hàm vẽ biểu đồ
│
├── docs/                      # 📚 Tài liệu giải thích
│   ├── SURVIVAL_GUIDE.md       # Hướng dẫn sống còn
│   ├── ANALOGIES.md            # Giải thích bằng ví dụ đời sống
│   └── FLOW_DIAGRAM.md         # Sơ đồ flow của chương trình
│
├── notebooks/                 # 📓 Notebook để chạy
│   └── run_complete.ipynb      # Notebook chính (flow rõ ràng)
│
├── utils/                     # 🔧 Utilities
│   ├── runtime.py              # Config TensorFlow
│   └── save_results.py         # Lưu kết quả (metrics, plots)
│
├── reports/                   # 📊 Kết quả đã lưu
│   ├── main/                   # Kết quả từ main.py
│   └── notebook/               # Kết quả từ notebook
│
├── main.py                    # 🎯 Entry point (CLI)
└── clean.py                   # 🧹 Dọn dẹp project
```

**Mỗi folder chỉ làm 1 việc duy nhất, rõ ràng!**

---

## 🎮 Cách Chạy

### Chạy từ Notebook

```bash
uv run jupyter notebook
```

Sau đó mở `notebooks/run_complete.ipynb`

**Notebook có:**
- ✅ Checklist để đánh dấu tiến độ
- ✅ Giải thích từng bước
- ✅ Code sẵn sàng chạy

### Chạy từ CLI

```bash
# Cấu hình mặc định
uv run python main.py

# Tùy chỉnh tham số
uv run python main.py --epochs 30 --limit 2000
uv run python main.py --timeframe 4h --window 30
uv run python main.py --refresh-cache
```

**Các tham số quan trọng:**
- `--timeframe`: `1d`, `4h`, `1h` (mặc định: `1d`)
- `--limit`: Số nến lấy từ Binance (mặc định: `1500`)
- `--window`: Số nến nhìn lại (mặc định: `60`)
- `--epochs`: Số epochs (mặc định: `20`)
- `--refresh-cache`: Tải lại dữ liệu từ Binance

---

## 📚 Tài Liệu Quan Trọng

| Tài liệu | Nội dung | Khi nào đọc? |
|----------|---------|--------------|
| [START_HERE.md](START_HERE.md) | Hướng dẫn bắt đầu | **ĐÂY - BÂY GIỜ!** |
| [docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md) | Hướng dẫn sống còn - giải thích từng bước, troubleshooting | Khi gặp vấn đề |
| [docs/ANALOGIES.md](docs/ANALOGIES.md) | Giải thích các khái niệm bằng ví dụ đời sống | Khi không hiểu khái niệm |
| [docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md) | Sơ đồ flow của toàn bộ chương trình | Khi muốn hiểu quy trình tổng thể |

---

## 💡 Tips Cho Người ADHD

### 1. Làm từng bước một
- Đừng nhảy cóc, làm xong bước này mới sang bước kia
- Mỗi folder chỉ làm 1 việc, dễ theo dõi

### 2. Đánh dấu checklist
- Trong notebook có checklist để đánh dấu tiến độ
- Tích vào checkbox khi làm xong mỗi bước

### 3. Đọc comments
- Code có comments bằng tiếng Việt
- Giải thích từng hàm, biến, tham số

### 4. Nghỉ giải lao
- Nếu cảm thấy ngợp, nghỉ 5-10 phút rồi quay lại
- Không cần hiểu hết ngay, cứ làm từng bước

### 5. Đọc ANALOGIES.md
- Giúp hiểu các khái niệm bằng ví dụ đời sống
- BiLSTM, LSTM, Sliding Window... đều có analogies

---

## 🆘 Nếu Bị Lạc

### Quên mình đang làm gì?
→ Đọc lại `START_HERE.md` (file này!)

### Không hiểu code?
→ Đọc `docs/ANALOGIES.md` để hiểu khái niệm bằng ví dụ đời sống

### Gặp lỗi?
→ Xem phần Troubleshooting trong `docs/SURVIVAL_GUIDE.md`

### Muốn hiểu flow?
→ Xem `docs/FLOW_DIAGRAM.md` để xem sơ đồ luồng

### Không biết code ở đâu?
- Mỗi folder chỉ có 1-2 files
- Tên folder mô tả rõ ràng chức năng
- Tên file cũng mô tả chức năng

---

## 📝 Lưu Ý Quan Trọng

- ✅ **Mỗi folder chỉ làm 1 việc** - đừng lo lắng về việc code ở đâu
- ✅ **Comments bằng tiếng Việt** - đọc comments để hiểu code
- ✅ **Từng bước một** - không cần hiểu hết ngay, cứ làm từng bước
- ✅ **Kết quả được tự động lưu** vào `reports/`
- ✅ **Có dọn dẹp project** với `clean.py`

---

## 🧹 Dọn Dẹp Project

Nếu project có quá nhiều file cache hoặc reports cũ:

```bash
# Dọn tất cả (cache + reports cũ, giữ lại 5 file reports mới nhất)
uv run python clean.py

# Chỉ dọn cache và checkpoint
uv run python clean.py --cache

# Chỉ dọn reports cũ (giữ lại 10 file mới nhất)
uv run python clean.py --reports --keep-reports 10

# Xóa cache dữ liệu (chỉ file cũ > 30 ngày)
uv run python clean.py --data-cache

# Xóa TẤT CẢ cache dữ liệu
uv run python clean.py --data-cache-force
```

---

## 🎯 Bắt Đầu Ngay!

Chọn 1 trong 2 cách:

1. **Nếu bạn thích hướng dẫn chi tiết, từng bước:**
   → Chạy notebook: `uv run jupyter notebook`
   → Mở `notebooks/run_complete.ipynb`

2. **Nếu bạn thích nhanh gọn:**
   → Chạy CLI: `uv run python main.py --epochs 20 --limit 1500`

**Chúc bạn thành công! 🚀**
