# Author: Rupali Pragati
# Repository: OIBSIP
# Objective: Analyze unemployment trends during COVID-19 using data visualization and EDA

# 1. Import Libraries
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. Download Dataset
print("Downloading dataset from KaggleHub...")
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Dataset downloaded to:", path)

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
if 'region' not in df.columns and 'area' in df.columns:
    df.rename(columns={'area': 'region'}, inplace=True)

print("\nDataset Loaded Successfully!")
print(df.head())

# 5. Data Overview
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Regions:", df['region'].nunique())
print("Date Range:", df['date'].min(), "to", df['date'].max())

# 6. Data Cleaning
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.rename(columns={'estimated_unemployment_rate_(%)': 'unemployment_rate'}, inplace=True)
df = df.dropna(subset=['unemployment_rate'])
df.to_csv("cleaned_unemployment_data.csv", index=False)

# 7. Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# 8. Visualizations
plt.figure(figsize=(12,6))
sns.barplot(x='region', y='unemployment_rate', data=df, palette='mako', ci=None)
plt.title("Average Unemployment Rate by Region")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='unemployment_rate', data=df, hue='region', legend=False, lw=1.5)
plt.title("Unemployment Rate Trend Over Time (COVID-19 Impact)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='region', y='unemployment_rate', data=df, palette='coolwarm')
plt.title("Unemployment Rate Distribution by Region")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

num_cols = df.select_dtypes(include=['float64', 'int64'])
if not num_cols.empty:
    plt.figure(figsize=(8,5))
    sns.heatmap(num_cols.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.show()

# 9. Insights Summary
print("\nProject Summary:")
print("The analysis highlights the rise in unemployment rates during COVID-19.")
print("Regional variations indicate differing economic impacts across states.")
print("Visualizations show trends and patterns useful for economic forecasting.")

print("\nAnalysis Completed Successfully!")
