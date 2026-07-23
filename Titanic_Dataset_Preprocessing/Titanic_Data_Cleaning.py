import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"Titanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
plt.figure(figsize=(8,5))
sns.boxplot(data=df[["Age","Fare"]])
plt.title("Before Removing Outliers")
plt.show()
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]
plt.figure(figsize=(8,5))
sns.boxplot(data=df[["Fare"]])
plt.title("After Removing Outliers")
plt.show()
print(df.head())
print(df.info())
