# E-Commerce AI Dashboard

Ứng dụng Streamlit này sử dụng các mô hình học máy để phân tích và dự báo nhu cầu sản phẩm.

## Yêu cầu

- Python 3.8+ (hoặc phiên bản tương thích)
- Các thư viện trong `requirements.txt`

## Cài đặt

1. Sao chép hoặc tải mã nguồn về máy.
2. Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

## Tải mô hình

App yêu cầu các tệp mô hình và bộ tiền xử lý sau được đặt trong thư mục gốc của dự án:

- `lightgbm_demand_model.pkl`
- `knn_model.pkl`
- `scaler_knn.pkl`
- `le_brand.pkl`
- `le_category.pkl`
- `product_lookup.pkl`
- `scaler_x_lstm.pkl`
- `lstm_demand_model.keras`
- `scaler_y_lstm.pkl`
- `ecommerce_final_processed.csv`

Tải toàn bộ tệp cần thiết từ link Google Drive sau:

https://drive.google.com/drive/folders/1Ow2lvsiTDHgo1xKFrLvQDjXGwnJtz1E5?usp=sharing

Sau khi tải về, giải nén và đặt các tệp vào cùng thư mục với `app.py`.

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Ghi chú

- App sẽ cố gắng tải mô hình LSTM nếu TensorFlow được cài đặt. Nếu không có TensorFlow, phần dự báo LSTM sẽ không khả dụng.
- Nếu thiếu bất kỳ tệp nào trong các tệp mô hình hoặc dữ liệu, app có thể gặp lỗi khi khởi động.
