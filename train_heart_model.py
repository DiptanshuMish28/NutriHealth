import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the heart disease dataset
df = pd.read_csv('heart.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with adjusted parameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # This helps with imbalanced classes
)
model.fit(X_train_scaled, y_train)

# Save both the model and scaler
joblib.dump(model, 'models/heart.pkl')
joblib.dump(scaler, 'models/heart_scaler.pkl')

# Print model accuracy
print("Model trained and saved successfully!")
print(f"Training accuracy: {model.score(X_train_scaled, y_train):.2f}")
print(f"Testing accuracy: {model.score(X_test_scaled, y_test):.2f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)