import pandas as pd

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
