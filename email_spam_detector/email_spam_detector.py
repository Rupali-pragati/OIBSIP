# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub
import os

# 2. Download Dataset
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", dataset_path)

# 3. Locate the CSV/TSV File
csv_path = None
for file in os.listdir(dataset_path):
    if file.endswith(".csv") or file.endswith(".tsv"):
        csv_path = os.path.join(dataset_path, file)
        break

if not csv_path:
    raise FileNotFoundError("No CSV/TSV dataset file found in the downloaded folder!")

# 4. Load Dataset
if csv_path.endswith(".tsv"):
    df = pd.read_csv(csv_path, sep="\t", names=["label", "message"])
else:
    df = pd.read_csv(csv_path, encoding="latin-1")

# Keep only useful columns
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
# 5. Clean Dataset
# Drop missing and invalid rows
df = df.dropna(subset=["label", "message"])
df = df[df["label"].isin(["ham", "spam"])]

# Map labels: ham → 0, spam → 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\nData after cleaning:")
print(df.head())
print(f"\nTotal records after cleaning: {len(df)}")
print("Null values:\n", df.isnull().sum())

# 6. Split Features & Target
X = df["message"]
y = df["label"]

# 7. Convert Text to Numerical Features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_transformed = vectorizer.fit_transform(X)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)


# 9. Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 10. Make Predictions
y_pred = model.predict(X_test)

# 11. Evaluate Model

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 12. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 13. Test with Custom Email Input
def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"


sample_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your reward."
print("\nSample Email Prediction:", predict_email(sample_email))

# 14. Save Cleaned Dataset
df.to_csv("cleaned_spam_dataset.csv", index=False)
print("\nProcessing completed! Cleaned dataset saved as 'cleaned_spam_dataset.csv'.")
