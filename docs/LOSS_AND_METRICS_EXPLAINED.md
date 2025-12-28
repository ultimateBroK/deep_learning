# 📚 Giải Thích: Loss và Metrics Trong Training

**Ngày tạo:** 2025-12-28

---

## 🎯 Tổng Quan

Khi training mô hình, bạn sẽ thấy các giá trị như:
- **Train Loss**: 0.001393
- **Val Loss**: 0.000348
- **Train MAE**: 0.027805
- **Val MAE**: 0.014329

Những con số này có ý nghĩa gì? Hãy cùng tìm hiểu!

---

## 📊 1. LOSS (Mất Mát) - "Độ Sai Lệch"

### 🔍 **Loss là gì?**

**Loss** là một con số đo lường **mức độ sai lệch** giữa dự đoán của mô hình và giá trị thực tế.

**Ví dụ đời sống:**
- Giống như khi bạn làm bài kiểm tra:
  - **Loss thấp** = Bạn làm đúng nhiều câu → Điểm cao
  - **Loss cao** = Bạn làm sai nhiều câu → Điểm thấp

### 📐 **Công Thức Loss (MSE - Mean Squared Error)**

Trong project của bạn, Loss được tính bằng **MSE (Mean Squared Error)**:

```
Loss = (1/n) × Σ(y_true - y_pred)²
```

Trong đó:
- `y_true`: Giá trị thực tế (ví dụ: giá Bitcoin thực tế)
- `y_pred`: Giá trị dự đoán (ví dụ: giá Bitcoin mô hình dự đoán)
- `n`: Số lượng mẫu

**Ví dụ cụ thể:**

Giả sử bạn có 3 mẫu:
- Mẫu 1: Thực tế = $100, Dự đoán = $102 → Sai số = $2 → Bình phương = 4
- Mẫu 2: Thực tế = $200, Dự đoán = $198 → Sai số = $2 → Bình phương = 4
- Mẫu 3: Thực tế = $150, Dự đoán = $151 → Sai số = $1 → Bình phương = 1

**Loss = (4 + 4 + 1) / 3 = 3.0**

### 🎯 **Loss Càng Thấp Càng Tốt**

- ✅ **Loss = 0.001** → Rất tốt! Mô hình dự đoán gần như chính xác
- ⚠️ **Loss = 0.1** → Tạm được, nhưng còn sai khá nhiều
- ❌ **Loss = 1.0** → Rất tệ! Mô hình dự đoán sai rất nhiều

---

## 📊 2. TRAIN LOSS vs VAL LOSS

### 🔍 **Train Loss (Loss trên tập huấn luyện)**

**Train Loss** là Loss được tính trên **tập dữ liệu mà mô hình đang học** (training set).

**Ví dụ từ kết quả của bạn:**
- **Train Loss: 0.001393**
- Được tính trên **20,904 mẫu** trong tập train
- Mô hình đã "nhìn thấy" những mẫu này và học từ chúng

**Ý nghĩa:**
- Cho biết mô hình học tốt đến mức nào trên dữ liệu đã học
- **Giảm dần** khi training → Mô hình đang học tốt hơn

### 🔍 **Val Loss (Loss trên tập validation)**

**Val Loss** là Loss được tính trên **tập dữ liệu validation** (mô hình KHÔNG được học từ tập này).

**Ví dụ từ kết quả của bạn:**
- **Val Loss: 0.000348**
- Được tính trên **4,404 mẫu** trong tập validation
- Mô hình **chưa bao giờ nhìn thấy** những mẫu này khi học

**Ý nghĩa:**
- Cho biết mô hình có **tổng quát hóa** tốt không (có hoạt động tốt trên dữ liệu mới không)
- **Quan trọng hơn Train Loss** vì nó đo khả năng áp dụng vào thực tế

---

### ⚖️ **So Sánh Train Loss vs Val Loss**

Từ kết quả của bạn:
- **Train Loss: 0.001393**
- **Val Loss: 0.000348**

**Phân tích:**

✅ **Val Loss < Train Loss** → **TỐT!**
- Mô hình hoạt động tốt hơn trên dữ liệu mới
- Không bị overfitting (học thuộc lòng dữ liệu train)
- Mô hình có khả năng tổng quát hóa tốt

⚠️ **Nếu Val Loss > Train Loss** → **CẢNH BÁO!**
- Mô hình có thể bị overfitting
- Học quá kỹ dữ liệu train → Không hoạt động tốt trên dữ liệu mới

**Ví dụ tình huống xấu:**
```
Train Loss: 0.0001  (rất thấp - học rất tốt)
Val Loss:   0.01    (cao hơn nhiều - không hoạt động tốt trên dữ liệu mới)
```
→ Mô hình đã "học thuộc lòng" dữ liệu train!

---

## 📊 3. MAE (Mean Absolute Error) - "Sai Số Trung Bình"

### 🔍 **MAE là gì?**

**MAE** đo lường **sai số trung bình tuyệt đối** giữa dự đoán và thực tế.

**Công thức:**
```
MAE = (1/n) × Σ|y_true - y_pred|
```

Khác với Loss (MSE):
- **MSE**: Bình phương sai số → Phạt nặng các sai số lớn
- **MAE**: Giá trị tuyệt đối sai số → Đo lường trực tiếp sai số

**Ví dụ cụ thể:**

Giả sử bạn có 3 mẫu:
- Mẫu 1: Thực tế = $100, Dự đoán = $102 → Sai số = $2
- Mẫu 2: Thực tế = $200, Dự đoán = $198 → Sai số = $2
- Mẫu 3: Thực tế = $150, Dự đoán = $151 → Sai số = $1

**MAE = (2 + 2 + 1) / 3 = $1.67**

→ Trung bình mô hình sai khoảng **$1.67** mỗi lần dự đoán

---

### 🔍 **Train MAE vs Val MAE**

**Train MAE:**
- MAE trên tập train
- Cho biết sai số trung bình trên dữ liệu đã học

**Val MAE:**
- MAE trên tập validation
- Cho biết sai số trung bình trên dữ liệu mới

**Ví dụ từ kết quả của bạn:**
- **Train MAE: 0.027805**
- **Val MAE: 0.014329**

**Lưu ý:** Các giá trị này đang ở dạng **scaled** (đã được chuẩn hóa về [0,1]). 
- Để chuyển về USD, cần **inverse transform** với scaler
- Kết quả cuối cùng trên test set: **MAE = $424.71** (đã được chuyển về USD)

---

## 📊 4. Tại Sao Có 2 Loại Metrics?

### 🎯 **Loss (MSE) vs MAE**

| Đặc điểm | Loss (MSE) | MAE |
|----------|------------|-----|
| **Mục đích** | Để training (tối ưu mô hình) | Để đánh giá (dễ hiểu) |
| **Tính toán** | Bình phương sai số | Giá trị tuyệt đối sai số |
| **Phạt** | Phạt nặng sai số lớn | Xử lý công bằng mọi sai số |
| **Giá trị** | Thường rất nhỏ (0.001) | Dễ hiểu hơn (USD) |

**Ví dụ:**

Giả sử có 2 sai số:
- Sai số 1: $10
- Sai số 2: $20

**MSE:**
- (10² + 20²) / 2 = (100 + 400) / 2 = 250

**MAE:**
- (10 + 20) / 2 = $15

→ MSE "phạt" sai số lớn hơn nhiều!

---

## 📊 5. Giải Thích Kết Quả Của Bạn

Từ file `results_BiLSTM_15m_w96_20251228_021622.md`:

### **Training History:**
```
Train Loss: 0.001393
Val Loss:   0.000348
Train MAE: 0.027805
Val MAE:   0.014329
```

### **Phân Tích:**

1. **Val Loss < Train Loss** ✅
   - Mô hình hoạt động tốt trên dữ liệu mới
   - Không bị overfitting
   - Có khả năng tổng quát hóa tốt

2. **Val MAE < Train MAE** ✅
   - Sai số trung bình trên dữ liệu mới thấp hơn
   - Mô hình dự đoán chính xác hơn trên dữ liệu chưa thấy

3. **Best Val Loss: 0.000198** (tại epoch 12)
   - Đây là giá trị tốt nhất trong suốt quá trình training
   - Model checkpoint được lưu tại epoch này

### **Kết Quả Cuối Cùng (trên Test Set):**
```
MAE: $424.71
RMSE: $601.66
MAPE: 0.47%
Direction Accuracy: 52.78%
```

**Giải thích:**
- **MAE = $424.71**: Trung bình mô hình sai khoảng **$424.71** mỗi lần dự đoán
- Với giá Bitcoin ~$100,000 → Sai số khoảng **0.42%** → Rất tốt!
- **Direction Accuracy = 52.78%**: Mô hình dự đoán đúng hướng giá tăng/giảm khoảng **52.78%** thời gian

---

## 📊 6. Cách Đọc Training History

Khi training, bạn sẽ thấy output như:

```
Epoch 1/15
Train Loss: 0.0113, Train MAE: 0.0633
Val Loss: 0.00048, Val MAE: 0.0205

Epoch 2/15
Train Loss: 0.0016, Train MAE: 0.0301
Val Loss: 0.00093, Val MAE: 0.0293

...

Epoch 12/15  ← BEST EPOCH!
Train Loss: 0.001393, Train MAE: 0.027805
Val Loss: 0.000198, Val MAE: 0.014329  ← Val Loss thấp nhất!
```

**Quan sát:**
- ✅ Loss giảm dần → Mô hình đang học tốt hơn
- ✅ Val Loss thấp nhất tại epoch 12 → Đây là mô hình tốt nhất
- ✅ Sau epoch 12, Val Loss có thể tăng → Overfitting bắt đầu

---

## 📊 7. Các Tình Huống Thường Gặp

### ✅ **Tình Huống Tốt (Như Kết Quả Của Bạn):**
```
Train Loss: 0.001393
Val Loss:   0.000348  ← Thấp hơn Train Loss
```
→ Mô hình học tốt và tổng quát hóa tốt!

### ⚠️ **Tình Huống Overfitting:**
```
Train Loss: 0.0001   ← Rất thấp
Val Loss:   0.01      ← Cao hơn nhiều!
```
→ Mô hình học thuộc lòng dữ liệu train!

**Giải pháp:**
- Tăng dropout
- Giảm số lượng LSTM units
- Thêm regularization
- Dừng training sớm hơn (EarlyStopping)

### ⚠️ **Tình Huống Underfitting:**
```
Train Loss: 0.1      ← Cao
Val Loss:   0.12     ← Cũng cao
```
→ Mô hình chưa học đủ!

**Giải pháp:**
- Tăng số epochs
- Tăng số lượng LSTM units
- Giảm dropout
- Tăng window size

---

## 🎯 Tóm Tắt

### **Loss (MSE):**
- ✅ Đo mức độ sai lệch (bình phương)
- ✅ Dùng để training (tối ưu mô hình)
- ✅ Càng thấp càng tốt

### **MAE:**
- ✅ Đo sai số trung bình (giá trị tuyệt đối)
- ✅ Dễ hiểu hơn (có thể chuyển về USD)
- ✅ Càng thấp càng tốt

### **Train vs Val:**
- ✅ **Train**: Đo trên dữ liệu đã học
- ✅ **Val**: Đo trên dữ liệu mới (quan trọng hơn!)
- ✅ **Val < Train** → Tốt! Không bị overfitting

### **Kết Quả Của Bạn:**
- ✅ Val Loss < Train Loss → Mô hình tốt!
- ✅ Val MAE < Train MAE → Dự đoán chính xác!
- ✅ Best Val Loss = 0.000198 → Rất tốt!

---

**Chúc bạn hiểu rõ về Loss và Metrics! 🚀**
