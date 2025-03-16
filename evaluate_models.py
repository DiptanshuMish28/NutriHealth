import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('models', exist_ok=True)

def create_evaluation_metrics(y_true, y_pred, y_prob, model_name):
    """Create and save confusion matrix and metrics for a model"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations for metrics
    plt.text(1.5, -0.5, f'Accuracy: {accuracy:.2f}', fontsize=12)
    plt.text(1.5, -0.7, f'Precision: {precision:.2f}', fontsize=12)
    plt.text(1.5, -0.9, f'Recall: {recall:.2f}', fontsize=12)
    plt.text(1.5, -1.1, f'F1-Score: {f1:.2f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'static/{model_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # Create classification report
    report = classification_report(y_true, y_pred)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }

def evaluate_all_models():
    all_results = {}
    
    # 1. Diabetes Model Evaluation
    try:
        # Load diabetes data
        diabetes_data = pd.read_csv('diabetes.csv')
        X_diabetes = diabetes_data.drop('Outcome', axis=1)
        y_diabetes = diabetes_data['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
        
        # Train and evaluate diabetes model
        diabetes_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        diabetes_model.fit(X_train, y_train)
        
        y_pred_diabetes = diabetes_model.predict(X_test)
        y_prob_diabetes = diabetes_model.predict_proba(X_test)
        
        diabetes_results = create_evaluation_metrics(y_test, y_pred_diabetes, y_prob_diabetes, 'Diabetes')
        all_results['diabetes'] = diabetes_results
        
        # Save the model
        joblib.dump(diabetes_model, 'models/diabetes.pkl')
        
        print("Diabetes Model Evaluation Complete")
        print(f"Accuracy: {diabetes_results['accuracy']:.4f}")
        print(f"Classification Report:\n{diabetes_results['classification_report']}")
        
    except Exception as e:
        print(f"Error in diabetes model evaluation: {str(e)}")
    
    # 2. Heart Disease Model Evaluation
    try:
        # Load heart disease data
        heart_data = pd.read_csv('heart.csv')
        X_heart = heart_data.drop('target', axis=1)
        y_heart = heart_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate heart model
        heart_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        heart_model.fit(X_train_scaled, y_train)
        
        y_pred_heart = heart_model.predict(X_test_scaled)
        y_prob_heart = heart_model.predict_proba(X_test_scaled)
        
        heart_results = create_evaluation_metrics(y_test, y_pred_heart, y_prob_heart, 'Heart')
        all_results['heart'] = heart_results
        
        # Save the model and scaler
        joblib.dump(heart_model, 'models/heart.pkl')
        joblib.dump(scaler, 'models/heart_scaler.pkl')
        
        print("Heart Disease Model Evaluation Complete")
        print(f"Accuracy: {heart_results['accuracy']:.4f}")
        print(f"Classification Report:\n{heart_results['classification_report']}")
        
    except Exception as e:
        print(f"Error in heart model evaluation: {str(e)}")
    
    # Create summary report
    with open('model_evaluation_report.txt', 'w') as f:
        f.write("Model Evaluation Summary\n")
        f.write("======================\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n{model_name.upper()} MODEL\n")
            f.write("=" * (len(model_name) + 7) + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n" + "="*50 + "\n")
    
    return all_results

# Run the evaluation
if __name__ == "__main__":
    results = evaluate_all_models()
    print("Evaluation completed. Check the generated files for results.")