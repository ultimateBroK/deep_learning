<div align="center">

# BÁO CÁO DỰ ÁN: DỰ BÁO GIÁ BITCOIN (BTC/USDT) KHUNG THỜI GIAN 15 PHÚT SỬ DỤNG MÔ HÌNH BiLSTM (Long Short-Term Memory hai chiều)

</div>

- **Sinh viên:** Nguyễn Đức Hiếu  
- **Môn học:** Học sâu (Deep Learning)  
- **Giảng viên:** Lê Văn Hùng  
- **Ngày nộp:** 28/12/2025  

---

## MỤC LỤC

1. [Tổng quan dự án](#sec-01)  
2. [Mục tiêu và phạm vi](#sec-02)  
3. [Cơ sở lý thuyết và khái niệm](#sec-03)  
4. [Thiết kế pipeline (dòng xử lý)](#sec-04)  
5. [Dữ liệu và tiền xử lý](#sec-05)  
6. [Kiến trúc mô hình và cấu hình huấn luyện](#sec-06)  
7. [Chiến lược thử nghiệm và tinh chỉnh siêu tham số (hyperparameter tuning)](#sec-07)  
8. [Kết quả thực nghiệm](#sec-08)  
9. [Đánh giá, so sánh và thảo luận](#sec-09)  
10. [Hướng dẫn chạy dự án (CLI - giao diện dòng lệnh / Notebook) và cách đọc kết quả](#sec-10)   
11. [Kết luận và hướng phát triển](#sec-11)  
12. [Cấu trúc dự án](#sec-12)  
13. [Tài liệu tham khảo](#sec-13)  
14. [Phụ lục](#sec-14)

---

<a id="sec-01"></a>
## 1. TỔNG QUAN DỰ ÁN

### 1.1. Giới thiệu

Dự án triển khai mô hình **BiLSTM** (*Bidirectional Long Short-Term Memory* - LSTM hai chiều) để dự báo **giá đóng cửa** (close price - giá đóng cửa) của Bitcoin (BTC/USDT) trên chuỗi thời gian (time series), với **khung thời gian 15 phút**. Hệ thống được tổ chức theo một *pipeline* (dòng xử lý chuẩn): đọc dữ liệu → chuẩn hóa & tạo chuỗi (*sliding window* - cửa sổ trượt) → huấn luyện (*training*) → đánh giá (*evaluation*) → lưu báo cáo/biểu đồ.

### 1.2. Vấn đề nghiên cứu

Dự báo giá tiền mã hóa (*cryptocurrency*) là bài toán khó do:

- Biến động mạnh (*volatility* cao) và nhiều nhiễu
- Quan hệ phi tuyến và phụ thuộc theo thời gian
- Thị trường bị tác động bởi nhiều yếu tố ngoại sinh (tin tức, cảm xúc thị trường (*sentiment*), thanh khoản, v.v.)

### 1.3. Giải pháp đề xuất

Áp dụng **BiLSTM** để học các quan hệ dài hạn trong chuỗi thời gian và tận dụng ngữ cảnh “hai chiều” bên trong cửa sổ đầu vào (*input window*).

> **Lưu ý:** BiLSTM **không nhìn tương lai ngoài điểm dự đoán**. “Hai chiều” ở đây nghĩa là xử lý hai hướng **trong cùng một cửa sổ đầu vào** (forward + backward).

---

<a id="sec-02"></a>
## 2. MỤC TIÊU VÀ PHẠM VI

### 2.1. Mục tiêu chính

1. Xây dựng *pipeline* (dòng xử lý) hoàn chỉnh để dự báo giá Bitcoin theo chuỗi thời gian.
2. Thiết kế và huấn luyện mô hình BiLSTM, áp dụng các *callback* (dừng sớm (*EarlyStopping*), lưu trọng số tốt nhất (*Checkpoint*)).
3. Thử nghiệm với nhiều cấu hình (đặc biệt *window size* - kích thước cửa sổ và số lượng dữ liệu) và so sánh kết quả.
4. Đánh giá dự báo qua các chỉ số: MAE (Sai số tuyệt đối trung bình), RMSE (Căn phương sai lớn nhất bình), MAPE (Sai số phần trăm trung bình tuyệt đối), Độ chính xác hướng (*Direction Accuracy*).

### 2.2. Phạm vi nghiên cứu

- **Dữ liệu**: BTC/USDT từ năm 2018–2025 (dạng CSV, xuất từ Binance).  
- **Khung thời gian trọng tâm**: **15 phút** (các khung khác có thể chạy tham khảo, nhưng báo cáo tập trung 15 phút).  
- **Thuộc tính (*feature*) sử dụng chính trong thực nghiệm**: **close** (giá đóng cửa).  
- **Số lượng thử nghiệm**: khoảng **10 cấu hình** (window size, giới hạn dòng, các biến thể preset/dataset 30k...).  

---

<a id="sec-03"></a>
## 3. CƠ SỞ LÝ THUYẾT VÀ KHÁI NIỆM

### 3.1. *Sliding window* (cửa sổ trượt, window size)

Bài toán dự báo một bước xây dựng như sau:

- **Input (đầu vào)**: lấy *window* (cửa sổ) gồm các điểm giá liên tiếp gần nhất (ví dụ 72 cây nến gần nhất), tức là đoạn lịch sử ngay trước thời điểm dự báo.
- **Target (đích/giá trị cần dự báo)**: giá của **bước kế tiếp** ngay sau đoạn lịch sử đó (dự báo nến tiếp theo).

Ví dụ với timeframe **15 phút** và **window=72**:

- Input = giá close của **72 cây nến gần nhất** (tương đương 18 giờ)
- Target = giá close của **nến tiếp theo** (15 phút sau)

### 3.2. Chuẩn hóa dữ liệu (*Scaling*)

Giá BTC biên độ rất lớn (10,000–100,000+), nên cần *scaling* (chuẩn hóa dữ liệu) để mô hình học ổn định hơn. Trong dự án này, chuẩn hóa phổ biến là **MinMaxScaler** (chuyển về [0,1]).

### 3.3. Chia tập train/val/test (huấn luyện/validation/kiểm thử) cho chuỗi thời gian

- **Không shuffle** (xáo trộn) khi chia tập, phải giữ nguyên thứ tự thời gian.
- Tỉ lệ dùng: **70% train / 15% validation / 15% test**

### 3.4. *Loss* (hàm mất mát) và các *metrics* (chỉ số đánh giá)

- **Loss (MSE – Mean Squared Error: trung bình bình phương sai số)**: hàm tối ưu trong *training*.
- **MAE (Mean Absolute Error – Sai số tuyệt đối trung bình)**: đơn vị USD, dễ hiểu.
- **RMSE (Root Mean Square Error – Căn phương sai lớn nhất bình)**: nhấn mạnh các điểm sai lớn (outlier – điểm ngoại lai).
- **MAPE (Mean Absolute Percentage Error – Sai số phần trăm tuyệt đối trung bình)**: thước đo phần trăm sai số.
- **Direction Accuracy (Độ chính xác hướng dự báo)**: dự báo đúng tăng/giảm.

> Khi huấn luyện, các chỉ số có thể nằm trên thang điểm đã scale. Khi đánh giá (*evaluate*), cần chuyển ngược (*inverse-transform*) về USD.

### 3.5. *LSTM* và *BiLSTM*

- **LSTM (Long Short-Term Memory – Mạng bộ nhớ dài-ngắn hạn)**: mô hình RNN với khả năng “ghi nhớ” dài hạn (giải quyết vấn đề mất dần gradient).
- **BiLSTM (Bidirectional LSTM – LSTM hai chiều)**: chạy LSTM theo cả hai chiều (forward + backward) trong cùng cửa sổ, giúp trích xuất ngữ cảnh tốt hơn.

---

<a id="sec-04"></a>
## 4. THIẾT KẾ PIPELINE (DÒNG XỬ LÝ) VÀ LUỒNG XỬ LÝ

### 4.1. Luồng tổng quan (*end-to-end*)

```
Lấy dữ liệu (CSV) → Tiền xử lý (Chuẩn hóa + Tạo chuỗi + Chia tập) → Xây dựng BiLSTM → Huấn luyện (*training*, có callback) → Đánh giá (*evaluation*) → Trực quan hóa (*visualization*) → Lưu báo cáo
```

### 4.2. Flow chi tiết theo 5 bước

**Bước 1: Đọc dữ liệu CSV (trên máy tính)**

- Đọc file trong `/data/`
- Chuẩn hóa các cột: datetime (thời gian), open (giá mở), high (cao nhất), low (thấp nhất), close (đóng), volume (khối lượng)
- Có hệ thống lưu tạm (*cache*) dữ liệu đã chuẩn hóa (tăng tốc cho lần chạy sau)

**Bước 2: Tiền xử lý**

- Chuẩn hóa dữ liệu bằng MinMaxScaler (phổ biến cho giá crypto)
- Tạo các chuỗi (*sequence*) bằng sliding window
- Chia tập train/val/test theo thời gian

**Bước 3: Xây dựng mô hình**

- Dạng input: (window_size, số lượng thuộc tính – n_features)
- 2 tầng BiLSTM + dropout + Dense layer đầu ra (dự báo giá 1 chiều)

**Bước 4: Huấn luyện**

- Tối ưu hóa (*optimizer*): Adam
- Hàm mất mát (*loss*): MSE
- *Callback*: ModelCheckpoint (lưu mô hình tốt nhất), EarlyStopping (dừng sớm), ReduceLROnPlateau (giảm tốc độ học khi kém đi)

**Bước 5: Đánh giá và trực quan hóa**

- Dự đoán trên tập test
- Chuyển ngược giá trị về USD (*inverse-transform*)
- Tính toán các chỉ số đánh giá, vẽ biểu đồ (lịch sử loss, so sánh giá thực tế – dự báo...)

---

<a id="sec-05"></a>
## 5. DỮ LIỆU VÀ TIỀN XỬ LÝ

### 5.1. Dữ liệu sử dụng

- **Nguồn**: dữ liệu lịch sử BTC/USDT (CSV).  
- **Khoảng thời gian**: 2018–2025.  
- **Khung thời gian báo cáo**: 15 phút.  
- **Các cột dữ liệu**: `datetime`, `open`, `high`, `low`, `close`, `volume`.  
- **Feature (thuộc tính) chính**: `close`.  

### 5.2. Quy trình tiền xử lý

1. **Làm sạch/chuẩn hóa định dạng**: đảm bảo datetime đúng kiểu thời gian và các cột số hợp lệ.
2. **Scaling (chuẩn hóa)**: MinMaxScaler chuyển về [0,1].
3. **Tạo chuỗi (sequences)**: dùng sliding window (vd: window=72).
4. **Chia tập**: 70/15/15 theo thời gian (không xáo trộn).
5. **Tái lập kết quả (*reproducibility*)**: dùng seed cố định (ví dụ 42).

### 5.3. Xử lý thiếu dữ liệu

- Loại bỏ dòng thiếu giá trị (null) nếu có và đảm bảo chuỗi thời gian liên tục theo dữ liệu đầu vào.

---

<a id="sec-06"></a>
## 6. KIẾN TRÚC MÔ HÌNH VÀ CẤU HÌNH HUẤN LUYỆN

### 6.1. Mô hình tốt nhất trong thực nghiệm

**Tên mô hình**: `BiLSTM_15m_w72_l30k`

**Cấu hình chính**

- **Timeframe (khung thời gian)**: 15 phút  
- **Limit dữ liệu**: 30,000 dòng cuối  
- **Window size (kích thước cửa sổ)**: 72 (≈ 18 giờ)  
- **LSTM units (số lượng đơn vị)**: [32, 16] (2 tầng BiLSTM)  
- **Dropout**: 0.2  
- **Batch size (kích thước lô)**: 32  
- **Learning rate (tốc độ học)**: 0.001  
- **Optimizer (thuật toán tối ưu hóa)**: Adam  
- **Epochs (số lần huấn luyện)**: 20 (có EarlyStopping)  
- **Early stopping patience (kiên nhẫn dừng sớm)**: 6 epochs  

**Sơ đồ kiến trúc**

```
Input (72 mốc thời gian, 1 thuộc tính)
    ↓
BiLSTM (32 đơn vị)
    ↓
Dropout (0.2)
    ↓
BiLSTM (16 đơn vị)
    ↓
Dropout (0.2)
    ↓
Dense (1 đơn vị) → Giá dự báo
```

### 6.2. Lý do lựa chọn cấu hình

- **Window size 72**: đủ ngữ cảnh (18 giờ lịch sử) nhưng không quá dài gây lẫn nhiễu/chi phí tính toán lớn.
- **2 tầng BiLSTM**: cân bằng giữa khả năng học phức tạp và tránh overfitting (quá khớp dữ liệu).
- **Dropout + callback**: giảm overfitting, tăng ổn định quá trình huấn luyện.

---

<a id="sec-07"></a>
## 7. CHIẾN LƯỢC THỬ NGHIỆM VÀ TINH CHỈNH SIÊU THAM SỐ (HYPERPARAMETER TUNING)

### 7.1. Mục tiêu tuning (*tinh chỉnh*)

Tìm cấu hình tốt nhất dựa trên các chỉ số:

- **MAE/RMSE/MAPE** (độ chính xác giá trị)
- **Độ chính xác hướng (*Direction Accuracy*)**

### 7.2. Nguyên tắc thử nghiệm có kiểm soát

- Cố định dataset (vd: `--limit 30000`) rồi thử nhiều `--window` (48/72/96/144...)
- Mỗi lần chỉ đổi 1 biến số để xác định yếu tố ảnh hưởng.
- Lưu cấu hình và metric mỗi lần chạy, dùng để so sánh.

### 7.3. Preset (bộ cấu hình nhanh)

Dự án có sẵn các preset theo mục tiêu (giao dịch ngắn hạn - *scalping*, trong ngày - *intraday*, giao dịch sóng - *swing*, sản xuất - *production*) và các preset cho dataset 30k để thử nghiệm công bằng.

Ví dụ:

- `intraday-balanced` (khuyến nghị chạy nhanh/ổn định)
- `30k-w72` (giới hạn 30,000 dòng, window 72, thuận tiện so sánh)

---

<a id="sec-08"></a>
## 8. KẾT QUẢ THỰC NGHIỆM

### 8.1. Mô hình tốt nhất: `BiLSTM_15m_w72_l30k`

**Kết quả trên tập test**

| Chỉ số (Metric) | Giá trị | Nhận xét |
|--------|---------|----------|
| **MAE** | **$399.18** | Tốt nhất trong các cấu hình đã thử |
| **RMSE** | $563.86 | Tốt |
| **MAPE** | **0.44%** | Sai số phần trăm rất nhỏ |
| **Direction Accuracy** | 52.52% | Gần mức ngẫu nhiên, cần cải thiện |

**Thông tin huấn luyện**

- **Best Epoch (tốt nhất ở vòng huấn luyện)**: 19/20  
- **Best Validation Loss (mất mát kiểm định thấp nhất)**: 0.000197  
- **Training Time (thời gian huấn luyện)**: 256.64 giây (~4 phút)  
- **Train Loss (mất mát huấn luyện)**: 0.001321  
- **Val Loss (mất mát kiểm định)**: 0.000245  

### 8.2. Top 5 cấu hình tốt nhất (theo MAE)

| Xếp hạng | Mô hình | MAE ($) | RMSE ($) | MAPE (%) | Độ chính xác hướng (%) |
|----------|-------|---------|----------|----------|------------------------|
| #1 | **w72_l30k** | **399.18** | 563.86 | **0.44%** | 52.52% |
| #2 | w96 (gốc) | 424.71 | 601.66 | 0.47% | 52.78% |
| #3 | w144_l30k | 403.10 | 541.32 | 0.45% | 50.95% |
| #4 | w96_l30k | 407.79 | **525.41** | 0.45% | 50.83% |
| #5 | w24 (gốc) | 427.97 | 627.18 | 0.49% | **53.90%** |

### 8.3. Nhận xét từ thực nghiệm

- **Window size 72** cho kết quả tốt nhất (MAE và MAPE thấp).
- **Limit=30k** và window từ 72–144 thường cho chất lượng tốt (cân bằng thời gian train và kết quả).
- Window quá nhỏ (24) hoặc quá lớn (240) kém hiệu quả qua thử nghiệm.

---

<a id="sec-09"></a>
## 9. ĐÁNH GIÁ, SO SÁNH VÀ THẢO LUẬN

### 9.1. Diễn giải các chỉ số metric

- **MAE = $399.18**: sai số trung bình ~399 USD. So với giá BTC ~100,000 USD, sai số <0.4%.
- **RMSE > MAE**: tồn tại điểm sai lớn (*outlier*).
- **MAPE = 0.44%**: sai số phần trăm nhỏ, phù hợp bài toán tài chính.

### 9.2. *Baseline* (đường cơ sở)

- Baseline đơn giản: **dự báo giá tiếp theo = giá hiện tại** (naive/last value – dự báo ngây thơ). Baseline thường cho MAE nhỏ ở vùng giá phẳng nhưng không thực sự "học" quy luật, mô hình hướng tới khả năng dự báo chuỗi phức tạp.

### 9.3. Thảo luận về Độ chính xác hướng (*Direction Accuracy*)

Direction Accuracy ~52% cho thấy mô hình dự báo **giá trị** tốt nhưng dự báo **hướng** còn hạn chế. Nguyên nhân có thể:

- Chỉ mới dùng feature "close", thiếu các yếu tố giải thích (volume, chỉ báo kỹ thuật, v.v.).
- Thị trường crypto biến động mạnh, khó dự báo hướng tăng/giảm ngắn hạn.

---

<a id="sec-10"></a>
## 10. HƯỚNG DẪN CHẠY DỰ ÁN (CLI – giao diện dòng lệnh / NOTEBOOK) VÀ CÁCH ĐỌC KẾT QUẢ

### 10.1. Cài đặt

```bash
cd /code
uv sync
```

> Nếu dùng pip: `pip install -r ../requirements.txt`

### 10.2. Chạy bằng CLI (*Command Line Interface* - giao diện dòng lệnh)

```bash
cd /code

# Chạy cấu hình mặc định (15m)
uv run python -m cli.main

# Chạy preset (bộ thông số nhanh)
uv run python -m cli.main --preset intraday-balanced

# Chạy cấu hình báo cáo (15m, limit 30k, window 72)
uv run python -m cli.main --timeframe 15m --limit 30000 --window 72 --epochs 20 --data-path ../data/btc_15m_data_2018_to_2025.csv
```

### 10.3. Chạy bằng Notebook

```bash
cd /code
uv run jupyter notebook
```

Mở file `notebooks/run_complete.ipynb` và chạy từ trên xuống. Nếu cần sửa đường dẫn data, chỉnh: `../data/btc_15m_data_2018_to_2025.csv`.

### 10.4. Đọc kết quả và so sánh các lần chạy

Sau khi chạy sẽ có thư mục kết quả:

- CLI: `reports/cli/`
- Notebook: `reports/notebook/`

Trong mỗi thư mục có thể có:

- `results_*.md`: báo cáo
- `metrics.json`: các chỉ số đánh giá
- `config.json`: cấu hình đã chạy
- `*.png`: biểu đồ

Nếu chạy notebook nhiều lần, xem tổng hợp ở `reports/notebook/KET_QUA.md`.

---

<a id="sec-11"></a>
## 11. KẾT LUẬN

1. Dự án đã xây dựng pipeline (*dòng xử lý*) dự báo giá BTC/USDT (15m) hoàn chỉnh: từ dữ liệu, tiền xử lý, BiLSTM → huấn luyện → đánh giá → tạo báo cáo.
2. Cấu hình `w72_l30k` cho kết quả tốt nhất: **MAE $399.18**, **MAPE 0.44%**.
3. Hạn chế lớn: **Direction Accuracy ~52%**, cần cải thiện khả năng dự báo hướng.

---

<a id="sec-12"></a>
## 12. CẤU TRÚC DỰ ÁN

```
/
├── code/
│   ├── src/
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── training.py
│   │   ├── results.py
│   │   ├── core/
│   │   ├── runtime/
│   │   └── visualization/
│   ├── cli/
│   ├── scripts/
│   └── notebooks/
├── docs/
├── data/
├── results/
└── BAO_CAO_DU_AN.md
```

---

<a id="sec-13"></a>
## 13. TÀI LIỆU THAM KHẢO

1. Bidirectional LSTM in NLP - Geeksforgeeks  ([link](https://www.geeksforgeeks.org/nlp/bidirectional-lstm-in-nlp/))
2. TensorFlow: Nền tảng huấn luyện mô hình học sâu –[Tensorflow](https://www.tensorflow.org/)
3. Keras: Thư viện xây dựng model học sâu – [Keras](https://keras.io/)
4. Polars: Thư viện xử lý dữ liệu nhanh – [Polars](https://docs.pola.rs/)
5. Numpy: Thư viện tính toán số học – [Numpy](https://numpy.org/doc/stable/)

---

<a id="sec-14"></a>
## 14. PHỤ LỤC

### A. Danh sách các mô hình đã thử nghiệm

Xem `results/KET_QUA.md`.

### B. Cấu hình chi tiết và chỉ số chi tiết (metrics)

Xem trong thư mục kết quả tương ứng, ví dụ:

- `results/BiLSTM_15m_w72_l30k_20251228_161832/config.json`
- `results/BiLSTM_15m_w72_l30k_20251228_161832/metrics.json`

---

*Dự án được thực hiện với mục đích học tập và nghiên cứu. Kết quả dự báo không nên dùng để đưa ra quyết định đầu tư thực tế mà không qua phân tích kỹ lưỡng và tư vấn chuyên môn.*
