import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("--- Task 1: Data Understanding and Preprocessing ---")
# 1. Load the dataset using Pandas
df = pd.read_csv('heart.csv')

# 2. Display the first five records
print("First five records of the dataset:")
print(df.head())

# 3. Identify Numerical features and Target variable
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
target_variable = 'target'
print("\nNumerical features:", numerical_features)
print("Target variable:", target_variable)

# 4. Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Ensure static folder exists
os.makedirs('static', exist_ok=True)

# Generate and save some visualizations
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
plt.savefig('static/target_distribution.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('static/correlation_matrix.png', bbox_inches='tight')
plt.close()

# 5. Split the dataset into 80% training and 20% testing
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nDataset split into {len(X_train)} training and {len(X_test)} testing records.")

print("\n--- Task 2: Model Development ---")
# Build a classification model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {acc * 100:.2f}%")

# Save the trained model using Joblib
joblib.dump(model, 'model.pkl')
print("\nTrained model saved as model.pkl")
