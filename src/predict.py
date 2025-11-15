"""
Script để dự đoán sentiment cho câu mới sử dụng model FinBERT đã được fine-tune
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Đường dẫn đến model đã train
model_path = "./models/finbert-trained/final"

# Load tokenizer và model
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Lấy id2label mapping từ model config
id2label = model.config.id2label
print(f"Model loaded successfully!")
print(f"Available labels: {id2label}")

def predict_sentiment(text: str):
    """
    Dự đoán sentiment cho một câu văn bản
    
    Args:
        text (str): Câu văn bản cần dự đoán
        
    Returns:
        tuple: (label, confidence) - Nhãn dự đoán và độ tin cậy
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        label = id2label[pred_id]
        confidence = probs[pred_id].item()
    
    return label, confidence

def predict_batch(texts: list):
    """
    Dự đoán sentiment cho nhiều câu cùng lúc
    
    Args:
        texts (list): Danh sách các câu văn bản
        
    Returns:
        list: Danh sách các tuple (label, confidence)
    """
    results = []
    for text in texts:
        label, confidence = predict_sentiment(text)
        results.append((label, confidence))
    return results

if __name__ == "__main__":
    # Ví dụ sử dụng
    print("\n" + "="*50)
    print("TESTING PREDICTION FUNCTION")
    print("="*50)
    
    test_sentences = [
        "The company expects strong revenue growth next year.",
        "Sales were down 10% in the third quarter compared to last year.",
        "The agreement is valid until 2008 and can be extended for another three years.",
        "The company has been struggling to meet its targets.",
        "Revenue increased by 5% compared to last year."
    ]
    
    print("\nPredictions:")
    for sentence in test_sentences:
        label, confidence = predict_sentiment(sentence)
        print(f"\nText: {sentence}")
        print(f"Prediction: {label} (confidence: {confidence:.4f})")
    
    # Test batch prediction
    print("\n" + "="*50)
    print("BATCH PREDICTION")
    print("="*50)
    results = predict_batch(test_sentences)
    for text, (label, confidence) in zip(test_sentences, results):
        print(f"\n{text}")
        print(f"  -> {label} ({confidence:.4f})")

