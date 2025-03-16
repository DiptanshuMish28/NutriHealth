# diabetes_model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your diabetes dataset
df = pd.read_csv('diabetes.csv')  # Replace with your dataset path

# Prepare features and target
X = df.drop('Outcome', axis=1)  # Assuming 'Outcome' is your target column
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, 'models/diabetes.pkl')