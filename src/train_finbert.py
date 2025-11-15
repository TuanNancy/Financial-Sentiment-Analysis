import pandas as pd
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    pipeline,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

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

# Chuẩn bị dữ liệu cho FinBERT (tokenization)
print("\n" + "="*50)
print("PREPARING DATA FOR FINBERT")
print("="*50)

# Tạo DataFrame từ train/val/test sets
train_df = pd.DataFrame({
    "text": X_train.values,
    "label_id": y_train.values
})

val_df = pd.DataFrame({
    "text": X_val.values,
    "label_id": y_val.values
})

test_df = pd.DataFrame({
    "text": X_test.values,
    "label_id": y_test.values
})

# Tạo Dataset từ pandas
train_ds = Dataset.from_pandas(train_df[["text", "label_id"]])
val_ds = Dataset.from_pandas(val_df[["text", "label_id"]])
test_ds = Dataset.from_pandas(test_df[["text", "label_id"]])

print(f"Created datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# Hàm tokenize batch
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize các dataset
print("Tokenizing datasets...")
train_ds = train_ds.map(tokenize_batch, batched=True)
val_ds = val_ds.map(tokenize_batch, batched=True)
test_ds = test_ds.map(tokenize_batch, batched=True)

# Đổi tên label_id → labels cho Trainer
train_ds = train_ds.rename_column("label_id", "labels")
val_ds = val_ds.rename_column("label_id", "labels")
test_ds = test_ds.rename_column("label_id", "labels")

# Thiết lập các cột PyTorch cần
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("Data preparation completed!")
print(f"Train dataset columns: {train_ds.column_names}")
print(f"Sample train data keys: {train_ds[0].keys()}")

# Training model với Trainer
print("\n" + "="*50)
print("TRAINING MODEL WITH TRAINER")
print("="*50)

# Hàm compute metrics cho evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/finbert-trained",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    seed=42,
    fp16=False,  # Set to True if using GPU with CUDA
)

# Tạo Trainer
trainer = Trainer(
    model=finbert,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# Training
print("Starting training...")
trainer.train()

# Evaluate trên validation set
print("\n" + "="*50)
print("EVALUATING ON VALIDATION SET")
print("="*50)
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Evaluate trên test set
print("\n" + "="*50)
print("EVALUATING ON TEST SET")
print("="*50)
test_results = trainer.evaluate(test_ds)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

# Save model và tokenizer
print("\n" + "="*50)
print("SAVING MODEL & TOKENIZER")
print("="*50)

model_save_path = "./models/finbert-trained/final"
print(f"Saving model to {model_save_path}...")
trainer.save_model(model_save_path)
print("✓ Model saved successfully")

print(f"Saving tokenizer to {model_save_path}...")
tokenizer.save_pretrained(model_save_path)
print("✓ Tokenizer saved successfully")

# Kiểm tra các file đã được lưu
saved_files = os.listdir(model_save_path)
print(f"\nSaved files in {model_save_path}:")
for file in sorted(saved_files):
    file_path = os.path.join(model_save_path, file)
    file_size = os.path.getsize(file_path)
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    elif file_size > 1024:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size} B"
    print(f"  - {file} ({size_str})")

print(f"\n✓ Model and tokenizer saved successfully to {model_save_path}")
print("You can load the model later using:")
print(f"  model = BertForSequenceClassification.from_pretrained('{model_save_path}')")
print(f"  tokenizer = BertTokenizer.from_pretrained('{model_save_path}')")

# Test model đã fine-tune trên một số mẫu
print("\n" + "="*50)
print("TESTING FINE-TUNED MODEL ON SAMPLE SENTENCES")
print("="*50)

# Load model đã fine-tune để test
fine_tuned_model = BertForSequenceClassification.from_pretrained("./models/finbert-trained/final")
fine_tuned_nlp = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=tokenizer)

test_sentences = [
    "there is a shortage of capital, and we need extra financing",
    "growth is strong and we have plenty of liquidity",
    "there are doubts about our finances",
    "profits are flat"
]

results = fine_tuned_nlp(test_sentences)
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
