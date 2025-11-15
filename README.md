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
│   └── train_finbert.py               # Script training model
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
pip install pandas numpy scikit-learn nltk transformers torch streamlit matplotlib seaborn
```

## Sử dụng

Chạy script để load và xử lý dataset:
```bash
python src/train_finbert.py
```

## Dataset

Dataset chứa 2478 mẫu với phân bố:
- Positive: 1254 mẫu
- Negative: 1156 mẫu
- Neutral: 68 mẫu

## Mapping Labels

- `positive` → 1
- `negative` → 0
- `neutral` → 2

## Tác giả

Financial Sentiment Analysis Project

