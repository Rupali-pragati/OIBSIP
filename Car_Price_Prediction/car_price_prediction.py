# Author: Rupali Pragati
# Repository: OIBSIP
# Objective: Predict the price of cars based on specifications using regression models

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import kagglehub
import os

# 2. Download Dataset

# Download latest version
path = kagglehub.dataset_download("vijayaadithyanvg/car-price-predictionused-cars")

print("Path to dataset files:", path)

# 3. Locate CSV File
csv_path = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_path = os.path.join(path, file)
        break

if not csv_path:
    raise FileNotFoundError("No CSV file found in the Kaggle dataset folder!")

# 4. Load Dataset
df = pd.read_csv(csv_path)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

print("\nDataset Loaded Successfully!")
print(df.head())

# 5. Data Overview
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# 6. Handle Missing Data (if any)
df = df.dropna()

# 7. Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# 8. Define Features and Target
if 'price' not in df.columns:
    if 'selling_price' in df.columns:
        target_col = 'selling_price'
    elif 'selling_price_inr' in df.columns:
        target_col = 'selling_price_inr'
    else:
        raise KeyError("No 'price' or 'selling_price' column found in dataset!")
else:
    target_col = 'price'

X = df.drop(columns=[target_col], errors='ignore')
y = df[target_col]


# 9. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 11. Model Training
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=150, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 12. Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# 13. Evaluation Metrics
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n {model_name} Evaluation Metrics:")
    print("MAE:", round(mean_absolute_error(y_true, y_pred), 2))
    print("MSE:", round(mean_squared_error(y_true, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("R² Score:", round(r2_score(y_true, y_pred), 2))

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")

# 14. Visual Comparison
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='teal')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # Feature Importance (Random Forest)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(10,5))
    importance.plot(kind='bar')
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

# 16. Save Cleaned Data and Model Results
df.to_csv("cleaned_car_data.csv", index=False)

print("\n Analysis Completed Successfully! Results saved as 'cleaned_car_data.csv'.")
