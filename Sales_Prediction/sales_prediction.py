# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import kagglehub
import os

# Download dataset from Kaggle
path = kagglehub.dataset_download("bumba5341/advertisingcsv")
print("Path to dataset files:", path)

# Automatically find the CSV file
csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

if not csv_file:
    raise FileNotFoundError("No CSV file found in the downloaded dataset folder.")

# Load dataset
data = pd.read_csv(csv_file)
print("First five rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in dataset:")
print(data.isnull().sum())

# Visualize data
sns.pairplot(data)
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Define independent and dependent variables
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted Sales:")
print(comparison.head())

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
