from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load your liver disease dataset
# Adjust the path and column names according to your dataset
df = pd.read_csv('liver.csv')

# Select the features in the same order as used in your app
features = df[[
    'Age',
    'Gender',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase',
    'Total_Protiens',
    'Albumin',
    'Albumin_and_Globulin_Ratio'
]]

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(features)

# Save the scaler
joblib.dump(scaler, 'models/liver_scaler.pkl')