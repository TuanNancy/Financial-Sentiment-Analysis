import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

file_path = "data/financial_sentiment_full.txt"

# Load dataset
df = pd.read_csv(
    file_path,
    names=["text", "label"],
    quotechar='"',
    skipinitialspace=True,
    skiprows=1,  # Skip empty first line
    on_bad_lines='skip',  # Skip malformed lines
    engine='python'  # Use Python engine for better error handling
)

# Map nhãn text → số
label_mapping = {
    "positive": 1,
    "negative": 0,
    "neutral": 2
}

df["label_encoded"] = df["label"].map(label_mapping)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution (text):")
print(df["label"].value_counts())
print("\nLabel distribution (encoded):")
print(df["label_encoded"].value_counts().sort_index())

# Chia train / validation / test
# Bước 1: Chia train (80%) và temp (20% - sẽ chia thành validation và test)
X = df["text"]
y = df["label_encoded"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Giữ phân bố labels
)

# Bước 2: Chia temp thành validation (50%) và test (50%)
# Tức là validation = 10% tổng, test = 10% tổng
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp  # Giữ phân bố labels
)

# In thông tin về các tập dữ liệu
print("\n" + "="*50)
print("DIVISION OF DATASET")
print("="*50)
print(f"\nTrain set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

print("\nTrain label distribution:")
print(y_train.value_counts().sort_index())
print("\nValidation label distribution:")
print(y_val.value_counts().sort_index())
print("\nTest label distribution:")
print(y_test.value_counts().sort_index())

# Load FinBERT model đã được pretrain
print("\n" + "="*50)
print("LOADING FINBERT MODEL")
print("="*50)
print("Loading FinBERT model from 'yiyanghkust/finbert-tone'...")

model_name = 'yiyanghkust/finbert-tone'
finbert = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

print("Model loaded successfully!")

# Test model trên một số mẫu
print("\n" + "="*50)
print("TESTING MODEL ON SAMPLE SENTENCES")
print("="*50)
test_sentences = [
    "there is a shortage of capital, and we need extra financing",
    "growth is strong and we have plenty of liquidity",
    "there are doubts about our finances",
    "profits are flat"
]

results = nlp(test_sentences)
for sentence, result in zip(test_sentences, results):
    label = result['label']
    score = result['score']
    print(f"\nSentence: {sentence}")
    print(f"Prediction: {label} (confidence: {score:.4f})")

# Map FinBERT labels to our label encoding
# FinBERT pipeline returns: "Positive", "Negative", "Neutral"
# Our encoding: 0=negative, 1=positive, 2=neutral
finbert_to_our_mapping = {
    'Positive': 1,   # positive
    'Negative': 0,   # negative
    'Neutral': 2     # neutral
}

# Evaluate trên test set (sample để tránh mất thời gian)
print("\n" + "="*50)
print("EVALUATING MODEL ON TEST SET (SAMPLE)")
print("="*50)
print("Evaluating on first 50 samples from test set...")

test_samples = X_test.iloc[:50].tolist()
test_labels = y_test.iloc[:50].tolist()

predictions = nlp(test_samples)
predicted_labels = [finbert_to_our_mapping[pred['label']] for pred in predictions]

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"\nAccuracy on 50 test samples: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, 
                          target_names=['negative', 'positive', 'neutral']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, predicted_labels)
print(cm)
