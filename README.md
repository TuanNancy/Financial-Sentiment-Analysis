# Financial Sentiment Analysis with FinBERT

Dự án phân tích cảm xúc tài chính sử dụng mô hình FinBERT để phân loại sentiment của các câu về tài chính.

## Mô tả

Dự án này sử dụng mô hình FinBERT (một biến thể của BERT được fine-tune cho lĩnh vực tài chính) để phân tích và phân loại cảm xúc của các câu về tài chính thành 3 loại:

- **Positive** (Tích cực)
- **Negative** (Tiêu cực)
- **Neutral** (Trung tính)

## Cấu trúc dự án

```
financial-sentiment-analysis/
├── data/
│   └── financial_sentiment_full.txt    # Dataset training
├── src/
│   ├── train_finbert.py               # Script training model
│   ├── predict.py                     # Hàm dự đoán sentiment
│   └── app.py                         # Streamlit web app
├── models/
│   └── finbert-trained/               # Model đã train
├── .gitignore
└── README.md
```

## Cài đặt

1. Tạo môi trường ảo:

```bash
python -m venv venv
```

2. Kích hoạt môi trường ảo:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Cài đặt các gói cần thiết:

```bash
pip install pandas numpy scikit-learn nltk transformers torch streamlit matplotlib seaborn datasets accelerate
```

## Sử dụng

### Training Model

Chạy script để train model:

```bash
python src/train_finbert.py
```

Model sẽ được lưu vào `./models/finbert-trained/final`

### Dự đoán Sentiment

Sử dụng script predict:

```bash
python src/predict.py
```

Hoặc import trong code:

```python
from src.predict import predict_sentiment

label, confidence = predict_sentiment("The company expects strong revenue growth.")
print(f"{label} ({confidence:.4f})")
```

### Web Demo với Streamlit

Chạy web app:

```bash
streamlit run src/app.py
```

Web app sẽ mở tại `http://localhost:8501` với các tính năng:

- 📝 **Single Text Analysis**: Phân tích sentiment cho một câu
- 📄 **Batch Analysis**: Phân tích nhiều câu cùng lúc
- 📊 **Visualization**: Hiển thị kết quả với màu sắc và confidence score

## Dataset

Dataset chứa 2478 mẫu với phân bố:

- Positive: 1254 mẫu
- Negative: 1156 mẫu
- Neutral: 68 mẫu

## Mapping Labels

- `positive` → 1
- `negative` → 0
- `neutral` → 2


https://github.com/user-attachments/assets/65942fac-25fc-47b0-a12f-49c034fc77a8


Financial Sentiment Analysis Project
