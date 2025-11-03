
#IRIS FLOWER CLASSIFICATION (Kaggle Dataset Version)

# Import Required Libraries
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os

# 2️ Load Dataset from Kaggle
print(" Downloading dataset from KaggleHub...")
path = kagglehub.dataset_download("saurabh00007/iriscsv")
print(" Dataset downloaded to:", path)

# Find CSV file inside dataset folder
csv_path = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_path = os.path.join(path, file)
        break

if not csv_path:
    raise FileNotFoundError("No CSV file found in the Kaggle dataset folder!")

df = pd.read_csv(csv_path)
print("\n Dataset Loaded Successfully!")
print(df.head())
df.columns = [col.strip().lower() for col in df.columns]
# 3️ Data Exploration
print("\n Dataset Information:")
print(df.info())

print("\n Missing Values:\n", df.isnull().sum())
print("\n Class Distribution:\n", df['species'].value_counts())
df.columns = [col.strip().lower() for col in df.columns]
if 'id' in df.columns:
    df = df.drop(columns=['id'])
target_col = 'species' if 'species' in df.columns else 'species'

# 4️ Data Visualization
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='species', palette='Set2')
plt.title("Number of Samples per Iris Species")
plt.show()

sns.pairplot(df, hue="species", palette="husl")
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 5️ Data Preprocessing
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

X = df.drop(columns=['species'])
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Data Split Complete!")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")


# 6️ Model Training & Evaluation

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    
    print(f"\n {name} Results ")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, preds))
    
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 7️ Model Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.show()

best_model_name = max(results, key=results.get)
print(f"\n Best Performing Model: {best_model_name} ({results[best_model_name]*100:.2f}% accuracy)")

# 8️ Sample Prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
best_model = models[best_model_name]
predicted_species = encoder.inverse_transform(best_model.predict(sample))
print(f"\n Prediction for {sample}: {predicted_species[0]}")

# 9️ Project Summary
print("\n Project Summary:")
print(f"The {best_model_name} model achieved {results[best_model_name]*100:.2f}% accuracy on the test data.")
print("This model can now classify new Iris flowers based on sepal and petal measurements.")
