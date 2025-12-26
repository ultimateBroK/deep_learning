# Mô hình dự báo giá Bitcoin với BiLSTM

Project đơn giản để học và thực hành dự báo giá Bitcoin (BTC/USDT) bằng mô hình **BiLSTM** (Bidirectional LSTM).

**Được thiết kế đặc biệt cho người ADHD - mọi thứ được chia thành từng bước rõ ràng, có giải thích bằng ví dụ đời sống.**

## 📁 Cấu trúc project (Rõ Ràng Từng Bước)

```
Deep_learning/
├── START_HERE.md             # ⭐ ĐỌC FILE NÀY TRƯỚC!
│
├── step1_data/               # BƯỚC 1: Lấy dữ liệu
│   ├── fetch_data.py         # Tải dữ liệu từ Binance
│   └── cache/                # Dữ liệu đã tải (CSV)
│
├── step2_preprocessing/      # BƯỚC 2: Xử lý dữ liệu
│   ├── create_windows.py     # Tạo windows (sequences)
│   └── scaling.py            # Chuẩn hóa dữ liệu
│
├── step3_model/             # BƯỚC 3: Xây dựng model
│   └── bilstm.py             # Model BiLSTM
│
├── step4_training/          # BƯỚC 4: Training
│   ├── train.py              # Hàm train model
│   └── evaluate.py          # Đánh giá kết quả
│
├── step5_visualization/      # BƯỚC 5: Vẽ biểu đồ
│   └── plots.py              # Các hàm vẽ biểu đồ
│
├── docs/                     # 📚 Tài liệu giải thích
│   ├── SURVIVAL_GUIDE.md     # Hướng dẫn sống còn
│   ├── ANALOGIES.md          # Giải thích bằng ví dụ đời sống
│   └── FLOW_DIAGRAM.md       # Sơ đồ flow của chương trình
│
├── notebooks/                # 📓 Notebook để chạy
│   └── run_complete.ipynb    # Notebook chính (flow rõ ràng)
│
├── utils/                    # 🔧 Utilities
│   ├── runtime.py            # Config TensorFlow
│   └── save_results.py      # Lưu kết quả (metrics, plots)
│
├── reports/                  # 📊 Kết quả đã lưu
│   ├── main/                 # Kết quả từ main.py
│   └── notebook/             # Kết quả từ notebook
│
└── main.py                   # 🎯 Entry point (CLI)
```

**Mỗi folder chỉ làm 1 việc duy nhất, rõ ràng!**

## 📚 Tài Liệu Quan Trọng

- **[START_HERE.md](START_HERE.md)**: Hướng dẫn bắt đầu - **ĐỌC FILE NÀY TRƯỚC!**
- **[docs/SURVIVAL_GUIDE.md](docs/SURVIVAL_GUIDE.md)**: Hướng dẫn sống còn - giải thích từng bước, troubleshooting
- **[docs/ANALOGIES.md](docs/ANALOGIES.md)**: Giải thích các khái niệm bằng ví dụ đời sống
- **[docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md)**: Sơ đồ flow của toàn bộ chương trình

## 🛠️ Cài đặt

```bash
uv sync
```

## 🧹 Dọn dẹp Project

Nếu project có quá nhiều file cache hoặc reports cũ, dùng script `clean.py`:

```bash
# Dọn tất cả (cache + reports cũ, giữ lại 5 file reports mới nhất)
uv run clean.py

# Chỉ dọn cache và checkpoint
uv run clean.py --cache

# Chỉ dọn reports cũ (giữ lại 10 folder mới nhất)
uv run clean.py --reports --keep 10

# Xóa cache dữ liệu (chỉ file cũ > 30 ngày)
uv run clean.py --data-cache

# Xóa TẤT CẢ cache dữ liệu
uv run clean.py --data-cache-force
```

## 🎯 Cách sử dụng

### Option 1: Chạy Notebook (Khuyến nghị cho người mới)

```bash
uv run jupyter notebook
```

Mở file `notebooks/run_complete.ipynb` và chạy từng cell theo thứ tự.

**Notebook có:**
- Markdown giải thích từng bước
- Checklist để đánh dấu tiến độ
- Analogies để dễ hiểu

### Option 2: Chạy CLI (Nhanh hơn)

```bash
uv run main.py --epochs 20 --limit 1500
```

**Các tham số:**
- `--timeframe`: `1d`, `4h`, `1h` (mặc định: `1d`)
- `--limit`: Số nến lấy từ Binance (mặc định: `1500`)
- `--window`: Số nến nhìn lại (mặc định: `60`)
- `--epochs`: Số epochs (mặc định: `20`)
- `--intra-threads`: CPU threads (mặc định: `12`)
- `--refresh-cache`: Tải lại dữ liệu từ Binance

## ⚙️ Tối ưu cho CPU AMD

Project đã được tối ưu cho CPU AMD với cấu hình mặc định:
- `intra_op_threads=12` (số core vật lý)
- `inter_op_threads=2`
- `enable_xla=True`

Bạn có thể điều chỉnh trong notebook hoặc CLI.

## 📊 Kết quả

Sau khi train, bạn sẽ thấy:

**Metrics:**
- **MAE**: Sai số trung bình tuyệt đối (USD)
- **RMSE**: Căn bậc hai của sai số bình phương trung bình (USD)
- **MAPE**: Sai số phần trăm trung bình (%)

**Biểu đồ:**
- Giá Bitcoin theo thời gian
- Training history (loss, val_loss)
- So sánh dự đoán vs thực tế

**Kết quả được tự động lưu vào:**
- `reports/main/` - Khi chạy `main.py`
- `reports/notebook/` - Khi chạy notebook

Mỗi lần chạy sẽ tạo folder chứa các file:
- `results_BiLSTM_YYYYMMDD_HHMMSS.md` - **File chính** (chứa tất cả: metrics, config, training history, links đến biểu đồ)
- `training_history_BiLSTM_YYYYMMDD_HHMMSS.png` - Biểu đồ training history
- `predictions_BiLSTM_YYYYMMDD_HHMMSS.png` - Biểu đồ dự đoán vs thực tế

> 💡 **Lưu ý:** Tất cả kết quả được tổng hợp trong file `.md` duy nhất để dễ đọc, không bị phân tán!

## 💡 Tips Cho Người ADHD

1. **Làm từng bước một**: Đừng nhảy cóc, làm xong bước này mới sang bước kia
2. **Đánh dấu checklist**: Tích vào checklist khi làm xong để biết tiến độ
3. **Đọc comments**: Comments giải thích rõ ràng bằng tiếng Việt
4. **Nghỉ giải lao**: Nếu cảm thấy ngợp, nghỉ 5 phút rồi quay lại
5. **Đọc ANALOGIES.md**: Giúp hiểu các khái niệm bằng ví dụ đời sống

## 🆘 Nếu Bị Lạc

1. **Quên mình đang làm gì?** → Đọc lại `START_HERE.md`
2. **Không hiểu code?** → Đọc `docs/ANALOGIES.md`
3. **Gặp lỗi?** → Xem phần Troubleshooting trong `docs/SURVIVAL_GUIDE.md`
4. **Muốn hiểu flow?** → Xem `docs/FLOW_DIAGRAM.md`

## 📝 Lưu Ý

- **Mỗi folder chỉ làm 1 việc** - đừng lo lắng về việc code ở đâu
- **Comments bằng tiếng Việt** - đọc comments để hiểu code
- **Từng bước một** - không cần hiểu hết ngay, cứ làm từng bước