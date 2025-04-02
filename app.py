import os
from flask import Flask, render_template, request, flash, jsonify, session, redirect, url_for
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from ocr import preprocess_image, extract_text_from_image, extract_medical_fields
import joblib
from werkzeug.utils import secure_filename
import traceback
import json
import io

# Load the heart disease model
try:
    heart_model = joblib.load('models/heart.pkl')
except Exception as e:
    print(f"Error loading heart model: {str(e)}")
    heart_model = None

# Load the liver disease model
try:
    liver_model = joblib.load('models/liver.pkl')
except Exception as e:
    print(f"Error loading liver model: {str(e)}")
    liver_model = None

# Load the diabetes model
try:
    diabetes_model = joblib.load('models/diabetes.pkl')
except Exception as e:
    print(f"Error loading diabetes model: {str(e)}")
    diabetes_model = None

# Load the pneumonia model at startup
try:
    pneumonia_model = tf.keras.models.load_model("models/trained.h5")
except Exception as e:
    print(f"Error loading pneumonia model: {str(e)}")
    pneumonia_model = None

app = Flask(__name__)
#app.secret_key = 'your_secret_key_here'  # Required for flash messages

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads directory exists
app.secret_key = 'your_secret_key_here'  # Required for flash messages

def calculate_calories(bmi, gender, age, risk_level):
    # Base calorie calculation using BMI ranges
    if bmi < 18.5:  # Underweight
        base_calories = 2500 if gender == 1 else 2000  # Higher calories for weight gain
    elif bmi < 25:  # Normal weight
        base_calories = 2200 if gender == 1 else 1800
    elif bmi < 30:  # Overweight
        base_calories = 2000 if gender == 1 else 1600
    else:  # Obese
        base_calories = 1800 if gender == 1 else 1400

    # Age adjustment
    if age > 50:
        base_calories -= 200
    elif age < 30:
        base_calories += 100

    # Risk level adjustment
    if risk_level == 'High':
        base_calories -= 300  # More restriction for high risk
    elif risk_level == 'Moderate':
        base_calories -= 200

    return base_calories

def get_diet_recommendation(disease, risk_level, bmi=None, gender=None, age=None):
    recommendations = []
    
    # Calculate calories if all required parameters are provided
    daily_calories = None
    if all(v is not None for v in [bmi, gender, age]):
        daily_calories = calculate_calories(bmi, gender, age, risk_level)
        calorie_info = (
            f"📊 Daily Calorie Recommendation: {daily_calories} calories\n"
            f"• Breakfast: {int(daily_calories * 0.3)} calories\n"
            f"• Lunch: {int(daily_calories * 0.35)} calories\n"
            f"• Dinner: {int(daily_calories * 0.25)} calories\n"
            f"• Snacks: {int(daily_calories * 0.1)} calories\n\n"
            f"💡 Recommended Meal Timing:\n"
            f"• Breakfast: 7:00-9:00 AM\n"
            f"• Mid-morning snack: 10:30-11:00 AM\n"
            f"• Lunch: 12:30-2:00 PM\n"
            f"• Evening snack: 4:00-5:00 PM\n"
            f"• Dinner: 7:00-8:00 PM"
        )
        recommendations.append(calorie_info)
    
    disease_recommendations = {
        'Liver Disease': {
            'High': [
                "🍽️ Liver-Healthy Macronutrient Distribution:\n"
                "• Protein: 20-25% of daily calories\n"
                "• Complex Carbohydrates: 45-50% of daily calories\n"
                "• Healthy Fats: 25-30% of daily calories\n"
                "• Fiber: 25-30g daily\n\n"
                "Target Daily Intake:\n"
                f"• Protein: {int(daily_calories * 0.25 / 4) if daily_calories else 'N/A'}g\n"
                f"• Carbohydrates: {int(daily_calories * 0.5 / 4) if daily_calories else 'N/A'}g\n"
                f"• Healthy Fats: {int(daily_calories * 0.25 / 9) if daily_calories else 'N/A'}g",

                "✅ Liver-Healthy Foods:\n"
                "• Lean Proteins:\n"
                "  - Fish (salmon, mackerel, sardines)\n"
                "  - Skinless poultry\n"
                "  - Plant-based proteins (tofu, tempeh)\n"
                "• Complex Carbohydrates:\n"
                "  - Whole grains (quinoa, brown rice, oats)\n"
                "  - Legumes (lentils, chickpeas)\n"
                "  - Non-starchy vegetables\n"
                "• Liver-Supporting Foods:\n"
                "  - Leafy greens (spinach, kale)\n"
                "  - Cruciferous vegetables (broccoli, cauliflower)\n"
                "  - Garlic and onions\n"
                "  - Berries and citrus fruits\n"
                "• Healthy Fats:\n"
                "  - Avocados\n"
                "  - Nuts and seeds\n"
                "  - Olive oil\n"
                "  - Fatty fish",

                "❌ Foods to Avoid:\n"
                "• Alcohol and alcoholic beverages\n"
                "• Processed and fried foods\n"
                "• High-sodium foods\n"
                "• Red meat and processed meats\n"
                "• Sugary beverages and desserts\n"
                "• Raw or undercooked shellfish\n"
                "• Foods high in saturated fats\n"
                "• Artificial sweeteners",

                "📊 Liver Health Monitoring:\n"
                "• Regular liver function tests\n"
                "• Monitor weight and BMI\n"
                "• Track fluid retention\n"
                "• Check for jaundice or abdominal swelling\n\n"
                "Target Numbers:\n"
                "• ALT: Below 40 U/L\n"
                "• AST: Below 40 U/L\n"
                "• Bilirubin: Below 1.2 mg/dL\n"
                "• Albumin: 3.5-5.0 g/dL",

                "🌿 Liver-Supporting Supplements (consult doctor):\n"
                "• Milk Thistle: 200-400mg daily\n"
                "• Vitamin D: 1,000-2,000 IU daily\n"
                "• Vitamin E: 400 IU daily\n"
                "• Zinc: 15-30mg daily\n"
                "• N-acetyl cysteine: 600-1,200mg daily",

                "💧 Hydration Guidelines:\n"
                "• Drink 8-10 glasses of water daily\n"
                "• Limit caffeine to 2-3 cups\n"
                "• Avoid sugary drinks\n"
                "• Monitor fluid retention\n"
                "• Consider electrolyte balance",

                "🏃‍♂️ Exercise Recommendations:\n"
                "• Moderate aerobic exercise: 30-45 mins daily\n"
                "• Strength training: 2-3 times weekly\n"
                "• Avoid high-impact activities\n"
                "• Listen to body signals\n"
                "• Stay hydrated during exercise",

                "⚠️ Lifestyle Modifications:\n"
                "• Complete alcohol abstinence\n"
                "• Regular sleep schedule (7-8 hours)\n"
                "• Stress management techniques\n"
                "• Regular medical check-ups\n"
                "• Avoid exposure to toxins\n"
                "• Practice good hygiene\n"
                "• Get recommended vaccinations"
            ],
            'Moderate': [
                "🍽️ Balanced Liver-Healthy Diet:\n"
                "• Protein: 20% of daily calories\n"
                "• Complex Carbohydrates: 50-55% of daily calories\n"
                "• Healthy Fats: 25-30% of daily calories\n"
                "• Fiber: 20-25g daily",

                "✅ Recommended Foods:\n"
                "• Lean proteins (fish, poultry)\n"
                "• Whole grains and legumes\n"
                "• Fresh fruits and vegetables\n"
                "• Healthy fats (avocados, nuts)\n"
                "• Liver-supporting foods (leafy greens, garlic)",

                "❌ Foods to Limit:\n"
                "• Alcohol (strictly limit or avoid)\n"
                "• Processed foods\n"
                "• High-sodium foods\n"
                "• Red meat\n"
                "• Sugary foods and drinks",

                "📊 Monitoring:\n"
                "• Regular liver function tests\n"
                "• Weight monitoring\n"
                "• Physical activity tracking\n"
                "• Sleep quality assessment",

                "🌿 General Health Support:\n"
                "• Stay hydrated\n"
                "• Regular exercise\n"
                "• Stress management\n"
                "• Adequate sleep\n"
                "• Regular medical check-ups"
            ],
            'Low': [
                "🍽️ Preventive Liver Health Guidelines:\n"
                "• Balanced macronutrient distribution\n"
                "• Focus on whole foods\n"
                "• Regular meal timing\n"
                "• Portion control",

                "✅ Healthy Choices:\n"
                "• Lean proteins\n"
                "• Whole grains\n"
                "• Fresh produce\n"
                "• Healthy fats\n"
                "• Liver-supporting foods",

                "❌ Foods to Avoid:\n"
                "• Excessive alcohol\n"
                "• Processed foods\n"
                "• High-sodium foods\n"
                "• Sugary beverages",

                "📊 General Monitoring:\n"
                "• Annual health check-ups\n"
                "• Regular exercise routine\n"
                "• Stress management\n"
                "• Sleep hygiene"
            ]
        },
        'Diabetes': {
            'High': [
                "🍽️ Macronutrient Distribution:\n"
                "• Carbohydrates: 45-50% of daily calories\n"
                "• Protein: 20-25% of daily calories\n"
                "• Healthy Fats: 25-30% of daily calories\n"
                "• Fiber: 25-30g daily\n\n"
                "Target Daily Intake:\n"
                f"• Carbohydrates: {int(daily_calories * 0.5 / 4) if daily_calories else 'N/A'}g\n"
                f"• Protein: {int(daily_calories * 0.25 / 4) if daily_calories else 'N/A'}g\n"
                f"• Healthy Fats: {int(daily_calories * 0.25 / 9) if daily_calories else 'N/A'}g",

                "✅ Recommended Foods:\n"
                "• Complex Carbohydrates:\n"
                "  - Whole grains (quinoa, brown rice, oats)\n"
                "  - Legumes (lentils, chickpeas, black beans)\n"
                "  - Non-starchy vegetables\n"
                "• Lean Proteins:\n"
                "  - Fish (salmon, tuna, mackerel)\n"
                "  - Skinless poultry\n"
                "  - Plant-based proteins (tofu, tempeh)\n"
                "• Healthy Fats:\n"
                "  - Avocados\n"
                "  - Nuts and seeds\n"
                "  - Olive oil\n"
                "• Low-Glycemic Fruits:\n"
                "  - Berries\n"
                "  - Apples\n"
                "  - Citrus fruits",

                "❌ Foods to Avoid:\n"
                "• Refined carbohydrates (white bread, pasta)\n"
                "• Sugary beverages and desserts\n"
                "• Processed foods\n"
                "• High-sodium foods\n"
                "• Trans fats and saturated fats",

                "📊 Blood Sugar Management:\n"
                "• Monitor blood sugar 4-6 times daily\n"
                "• Keep a food diary\n"
                "• Track carbohydrate intake\n"
                "• Regular exercise (30-45 mins daily)\n\n"
                "Target Numbers:\n"
                "• Fasting Blood Sugar: 80-130 mg/dL\n"
                "• Post-meal (2 hours): Below 180 mg/dL\n"
                "• HbA1c: Below 7%",

                "🌿 Supplements (consult doctor):\n"
                "• Vitamin D: 1,000-2,000 IU daily\n"
                "• Magnesium: 400mg daily\n"
                "• Chromium: 200-400mcg daily\n"
                "• Alpha-lipoic acid: 600-800mg daily",

                "💧 Hydration Guidelines:\n"
                "• Drink 8-10 glasses of water daily\n"
                "• Limit caffeine to 2-3 cups\n"
                "• Avoid sugary drinks\n"
                "• Monitor fluid intake with exercise"
            ],
            'Moderate': [
                "🍽️ Balanced Macronutrient Distribution:\n"
                "• Carbohydrates: 50-55% of daily calories\n"
                "• Protein: 20% of daily calories\n"
                "• Healthy Fats: 25-30% of daily calories\n"
                "• Fiber: 20-25g daily",

                "✅ Recommended Foods:\n"
                "• Whole grains and legumes\n"
                "• Lean proteins\n"
                "• Healthy fats\n"
                "• Fresh fruits and vegetables",

                "❌ Foods to Limit:\n"
                "• Refined sugars\n"
                "• Processed foods\n"
                "• High-sodium foods\n"
                "• Saturated fats",

                "📊 Monitoring:\n"
                "• Regular blood sugar checks\n"
                "• Weekly weight monitoring\n"
                "• Physical activity tracking"
            ],
            'Low': [
                "🍽️ General Dietary Guidelines:\n"
                "• Balanced macronutrient distribution\n"
                "• Focus on whole foods\n"
                "• Regular meal timing\n"
                "• Portion control",

                "✅ Healthy Choices:\n"
                "• Whole grains\n"
                "• Lean proteins\n"
                "• Healthy fats\n"
                "• Fresh produce",

                "📊 General Monitoring:\n"
                "• Annual health check-ups\n"
                "• Regular exercise routine\n"
                "• Stress management"
            ]
        },
        'Heart Disease': {
            'High': [
                "🫀 Advanced Cardiac Nutrition Protocol:\n"
                "• Sodium restriction: 1,500-2,000mg/day\n"
                "• Saturated fat intake: <6% of total calories\n"
                "• Dietary cholesterol: <150mg/day\n"
                "• Soluble fiber: 30-35g/day\n"
                "• Omega-3 fatty acids: 3-4g/day\n"
                "• Plant sterols: 2g/day\n"
                "• Potassium intake: 4,700mg/day\n"
                "• Magnesium intake: 400-500mg/day",

                "🏋️ Cardiorespiratory Exercise Protocol:\n"
                "• Zone 2 Training (65-75% MHR): 30-45 mins, 4-5x/week\n"
                "• HIIT Protocol: 4x4 method (4 min at 85-95% MHR, 3 min active recovery)\n"
                "• Resistance Training: 2-3 sets, 12-15 reps, RPE 6-7\n"
                "• Myocardial Adaptation Phase: 8-12 weeks\n"
                "• VO2 Max Target: Progressive increase to >35 ml/kg/min\n"
                "• Heart Rate Recovery Goal: <12 BPM drop in first minute\n"
                "• Blood Pressure Response: <10 mmHg spike during exercise",

                "🥗 Advanced Cardiac-Protective Diet:\n"
                "• Pre-Exercise Meal (2-3 hrs before):\n"
                "  - Complex CHO: 40-50g (Low GI <55)\n"
                "  - Lean protein: 15-20g\n"
                "  - Antioxidant-rich fruits: 1-2 servings\n\n"
                "• Post-Exercise Recovery:\n"
                "  - Whey/plant protein isolate: 25-30g\n"
                "  - Branch Chain Amino Acids: 5-7g\n"
                "  - Electrolyte replenishment: Na+/K+ balanced solution\n\n"
                "• Daily Micronutrient Targets:\n"
                "  - CoQ10: 200-300mg\n"
                "  - L-Carnitine: 2-3g\n"
                "  - D-Ribose: 5g\n"
                "  - Taurine: 1-2g\n"
                "  - Magnesium Citrate: 400mg",

                "📊 Biomarker Monitoring Protocol:\n"
                "• Lipid Panel Targets:\n"
                "  - LDL-C: <70 mg/dL\n"
                "  - HDL-C: >60 mg/dL\n"
                "  - Triglycerides: <100 mg/dL\n"
                "  - ApoB: <80 mg/dL\n"
                "• Inflammatory Markers:\n"
                "  - hs-CRP: <1 mg/L\n"
                "  - IL-6: <2 pg/mL\n"
                "  - Fibrinogen: <350 mg/dL\n"
                "• Metabolic Health:\n"
                "  - HbA1c: <5.7%\n"
                "  - Fasting Glucose: <90 mg/dL\n"
                "  - HOMA-IR: <1.5",

                "🌿 Phytonutrient Supplementation Strategy:\n"
                "• Primary Cardiovascular Support:\n"
                "  - Aged Garlic Extract: 600-1,200mg/day\n"
                "  - Bergamot Extract: 500mg BID\n"
                "  - Grape Seed Extract: 300-600mg/day\n"
                "  - Quercetin: 500mg BID\n"
                "• Nitric Oxide Boosters:\n"
                "  - L-Citrulline: 6-8g/day\n"
                "  - Beetroot Extract: 500mg/day\n"
                "  - Pomegranate Extract: 500mg/day\n"
                "• Antioxidant Complex:\n"
                "  - Mixed Tocotrienols: 200mg/day\n"
                "  - Astaxanthin: 12mg/day\n"
                "  - R-Lipoic Acid: 300mg BID",

                "⚡ Metabolic Optimization Protocol:\n"
                "• Meal Timing:\n"
                "  - Feeding Window: 8-10 hours\n"
                "  - Circadian Alignment: First meal within 1 hour of waking\n"
                "  - Pre-sleep Fast: 3 hours before bed\n"
                "• Macronutrient Distribution:\n"
                "  - Protein: 1.6-1.8g/kg lean mass\n"
                "  - Carbohydrates: 3-4g/kg (focus on resistant starch)\n"
                "  - Fats: 0.8-1g/kg (prioritize MUFA and omega-3)\n"
                "• Glucose Management:\n"
                "  - Post-meal glucose delta: <30 mg/dL\n"
                "  - Glycemic variability: <15%\n"
                "  - Time in range: >90%",

                "🧘‍♂️ Stress Management & Recovery:\n"
                "• HRV Optimization:\n"
                "  - Morning RMSSD Target: >45ms\n"
                "  - Daily HRV CV: <8%\n"
                "  - LF/HF Ratio: <2.0\n"
                "• Sleep Architecture:\n"
                "  - Total Sleep Time: 7.5-8.5 hours\n"
                "  - Deep Sleep: >20% of TST\n"
                "  - REM Sleep: >20% of TST\n"
                "  - Sleep Latency: <15 minutes\n"
                "• Parasympathetic Activation:\n"
                "  - Diaphragmatic Breathing: 6 breaths/min\n"
                "  - Progressive Muscle Relaxation: 15 mins/day\n"
                "  - Cold Exposure: 2-3 mins at 55°F"
            ],
            'Moderate': [
                "🫀 Intermediate Cardiac Protocol:\n"
                "• Sodium: 2,000-2,300mg/day\n"
                "• Saturated fat: <8% of calories\n"
                "• Fiber: 25-30g/day\n"
                "• Omega-3: 2-3g/day\n"
                "• Plant sterols: 1.5g/day",

                "🏋️ Exercise Protocol:\n"
                "• Zone 2 Training: 30 mins, 3-4x/week\n"
                "• HIIT: 30:30 method (30s high, 30s recovery)\n"
                "• Resistance Training: 2 sets, 12 reps\n"
                "• Target HR: 65-75% max\n"
                "• BP Response: <15 mmHg spike",

                "🥗 Nutrition Strategy:\n"
                "• Pre-Exercise:\n"
                "  - Complex CHO: 30-40g\n"
                "  - Protein: 15g\n"
                "• Post-Exercise:\n"
                "  - Protein: 20-25g\n"
                "  - BCAAs: 5g\n"
                "• Daily Supplements:\n"
                "  - CoQ10: 100-200mg\n"
                "  - Magnesium: 300mg\n"
                "  - Omega-3: 2g EPA/DHA"
            ],
            'Low': [
                "🫀 Preventive Protocol:\n"
                "• Sodium: <2,300mg/day\n"
                "• Fiber: >25g/day\n"
                "• Omega-3: 1-2g/day",

                "🏋️ Basic Exercise Plan:\n"
                "• Moderate activity: 150 mins/week\n"
                "• Strength training: 2x/week\n"
                "• Target HR: 60-70% max",

                "🥗 Basic Nutrition:\n"
                "• Balanced macros\n"
                "• Regular meal timing\n"
                "• Hydration: 2-3L/day"
            ]
        }
    }

    # Add disease-specific recommendations
    if disease in disease_recommendations and risk_level in disease_recommendations[disease]:
        recommendations.extend(disease_recommendations[disease][risk_level])
    
    return recommendations if recommendations else ["Maintain a balanced diet and consult with a healthcare provider"]

def predict(values, dic):
    result = {}
    
    # Diabetes Prediction
    if len(values) == 8:
        try:
            features = np.array([[float(dic[field]) for field in [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]])

            model = joblib.load('models/diabetes.pkl')
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            risk_percentage = round(prediction_proba[1] * 100, 2)

            result['disease'] = 'Diabetes'
            result['risk'] = risk_percentage
            result['level'] = 'High' if risk_percentage > 75 else 'Moderate' if risk_percentage > 50 else 'Low'
            result['message'] = f"Diabetes {'Detected' if prediction == 1 else 'Not Detected'} ({result['level']} Risk - {risk_percentage}%)"
        except Exception as e:
            raise e

    # Liver Disease Prediction
    elif len(values) == 10:
        try:
            features = np.array([[float(dic[field]) for field in [
                'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                'Aspartate_Aminotransferase', 'Total_Protiens', 
                'Albumin', 'Albumin_and_Globulin_Ratio']]])
            
            model = joblib.load('models/liver.pkl')
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            risk_percentage = round(prediction_proba[1] * 100, 2)

            result['disease'] = 'Liver Disease'
            result['risk'] = risk_percentage
            result['level'] = 'High' if risk_percentage > 75 else 'Moderate' if risk_percentage > 50 else 'Low'
            result['message'] = f"Liver Disease {'Detected' if prediction == 1 else 'Not Detected'} ({result['level']} Risk - {risk_percentage}%)"
        except Exception as e:
            raise e

    # Heart Disease Prediction
    elif len(values) == 13:
        try:
            features = np.array([[float(dic[field]) for field in [
                'Age', 'Gender', 'ChestPainType', 'RestingBP', 'Cholesterol',
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                'Oldpeak', 'ST_Slope']]])
            
            model = joblib.load('models/heart.pkl')
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            risk_percentage = round(prediction_proba[1] * 100, 2)

            result['disease'] = 'Heart Disease'
            result['risk'] = risk_percentage
            result['level'] = 'High' if risk_percentage > 70 else 'Moderate' if risk_percentage > 30 else 'Low'
            result['message'] = f"Heart Disease {'Detected' if prediction == 1 else 'Not Detected'} ({result['level']} Risk - {risk_percentage}%)"
        except Exception as e:
            raise e

    return result

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    data = None
    if request.method == 'POST':
        if 'image_file' in request.files:
            image_file = request.files['image_file']
            if image_file:
                # Save the uploaded image
                image_path = os.path.join('uploads', image_file.filename)
                image_file.save(image_path)

                # Use OCR functions from ocr.py to extract data
                preprocessed_image = preprocess_image(image_path)
                if preprocessed_image is not None:
                    extracted_text = extract_text_from_image(preprocessed_image)
                    if extracted_text:
                        data = extract_medical_fields(extracted_text)
                        
                        # Handle Gender field (not present in OCR output)
                        data['Gender'] = ''
                        
                        # Check if any fields are "Not found"
                        not_found_fields = [field for field, value in data.items() if value == "Not found"]
                        if not_found_fields:
                            flash(f"Some fields could not be extracted: {', '.join(not_found_fields)}. Please fill them manually.", "warning")
                        else:
                            flash("Data successfully extracted from the image. Please review and correct if necessary.", "success")
                    else:
                        flash("Could not extract text from the image. Please enter the data manually.", "error")
                else:
                    flash("Could not process the image. Please try again with a clearer image.", "error")
                
                # Clean up the uploaded file
                os.remove(image_path)
            else:
                flash("No file uploaded. Please select an image file.", "error")

    return render_template('liver.html', data=data)

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            
            # Extract BMI and other values
            bmi_display = to_predict_dict.get('bmi_display', None)
            gender = float(to_predict_dict.get('Gender', to_predict_dict.get('sex', 0)))
            age = float(to_predict_dict.get('Age', to_predict_dict.get('age', 0)))
            disease_type = to_predict_dict.get('disease_type', '')  # Store disease_type

            # Remove non-prediction fields
            prediction_dict = to_predict_dict.copy()  # Create a copy for prediction
            for field in ['bmi_display', 'height', 'weight', 'disease_type']:
                if field in prediction_dict:
                    del prediction_dict[field]

            # Convert values and predict
            for key, value in prediction_dict.items():
                try:
                    prediction_dict[key] = float(value)
                except ValueError:
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({
                            'success': False,
                            'error': f"Invalid value for {key}: {value}"
                        })
                    return render_template("home.html", message=f"Invalid value for {key}: {value}")

            to_predict_list = list(map(float, list(prediction_dict.values())))
            prediction_result = predict(to_predict_list, prediction_dict)

            # Calculate BMI category
            bmi_category = None
            bmi_value = None
            if bmi_display:
                try:
                    bmi_value = float(bmi_display)
                    if bmi_value < 18.5:
                        bmi_category = "Underweight"
                    elif bmi_value < 25:
                        bmi_category = "Normal weight"
                    elif bmi_value < 30:
                        bmi_category = "Overweight"
                    else:
                        bmi_category = "Obese"
                except ValueError:
                    bmi_category = None

            # Get diet recommendation
            diet_recommendation = get_diet_recommendation(
                disease_type,  # Use stored disease_type
                prediction_result.get('level', 'Low'),
                bmi_value,
                gender,
                age
            )

            # Select template based on disease type
            template = 'predict.html'  # default template
            if 'heart' in disease_type.lower():
                template = 'heart_predict.html'
            elif 'liver' in disease_type.lower():
                template = 'liver_predict.html'
            elif 'diabetes' in disease_type.lower():
                template = 'diabetes_predict.html'

            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'pred': prediction_result.get('message', ''),
                    'diet': diet_recommendation,
                    'bmi': bmi_display,
                    'bmi_category': bmi_category,
                    'risk_percentage': prediction_result.get('risk', 0)
                })

            return render_template(template,
                                pred=prediction_result.get('message', ''),
                                diet=diet_recommendation,
                                bmi=bmi_display,
                                bmi_category=bmi_category,
                                risk_percentage=prediction_result.get('risk', 0))

    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        traceback.print_exc()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': str(e)
            })
        return render_template("home.html", message=f"Error: {str(e)}")

    return render_template('home.html')

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image'])
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img))
        except:
            message = "Please upload an image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'GET':
        return redirect(url_for('pneumoniaPage'))
        
    try:
        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('pneumoniaPage'))
        
        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('pneumoniaPage'))

        # Check if model is loaded
        if pneumonia_model is None:
            flash('Model not available. Please try again later.', 'error')
            return redirect(url_for('pneumoniaPage'))

        # Process image and get prediction
        try:
            # Read and preprocess image
            image = Image.open(file.stream)
            image = image.convert('RGB')  # Convert to RGB
            image = image.resize((300, 300))  # Resize to match model's expected input
            
            # Convert to numpy array and normalize
            img_array = np.array(image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = pneumonia_model.predict(img_array)
            probability = float(prediction[0][0])  # Convert to Python float
            
            # Determine result
            is_pneumonia = probability > 0.5
            result_message = "Pneumonia Detected" if is_pneumonia else "Normal"
            
            # Return results template directly
            return render_template('pneumonia_predict.html',
                                pred=result_message)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            traceback.print_exc()
            flash('Error processing image. Please ensure you uploaded a valid chest X-ray image.', 'error')
            return redirect(url_for('pneumoniaPage'))

    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        traceback.print_exc()
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('pneumoniaPage'))

    return render_template('pneumonia.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            print("No filename provided")
            return jsonify({'error': 'No file selected'})
        
        if not file or not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload an image file (JPG, PNG, JPEG)'})
        
        # Get test type from header
        test_type = request.headers.get('X-Test-Type', '').lower()
        print(f"Processing {test_type} test type")
        
        # Read and process the image
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            print("Image opened successfully")
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            return jsonify({'error': 'Error processing image file'})
        
        # Extract text using OCR
        text = extract_text_from_image(img)
        if not text:
            print("No text extracted from image")
            return jsonify({'error': 'Could not extract text from the image. Please try a clearer image.'})
        
        print("Extracted text:", text)
        
        # Extract fields based on test type
        try:
            test_type = test_type.lower()  # Convert to lowercase for case-insensitive comparison
            if test_type == 'heart':
                data = extract_medical_fields(text, 'heart')
                print("Extracted heart data:", data)
            elif test_type == 'liver':
                data = extract_medical_fields(text, 'liver')
                print("Extracted liver data:", data)
            elif test_type == 'diabetes':
                data = extract_medical_fields(text, 'diabetes')
                print("Extracted diabetes data:", data)
            else:
                print(f"Invalid test type: {test_type}")
                return jsonify({'error': 'Invalid test type'})
        except Exception as e:
            print(f"Error extracting fields: {str(e)}")
            return jsonify({'error': f'Error extracting data from image: {str(e)}'})
        
        if not data:
            print("No data extracted")
            return jsonify({'error': 'No data could be extracted from the image. Please ensure the image is clear and contains the required information.'})
        
        # Always return the data, even if some fields are "Not found"
        # This allows partial population of the form
        return jsonify({'data': data})
        
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/heartpredictPage', methods=['GET', 'POST'])
def heartpredictPage():
    if request.method == 'GET':
        if request.args.get('results'):
            prediction_data = session.get('prediction')
            if prediction_data:
                # Format prediction message based on risk level
                risk_message = f"{prediction_data['risk_level']} Risk ({prediction_data['probability']}%)"
                return render_template('heart_predict.html',
                                    pred=risk_message,
                                    diet=prediction_data['diet_recommendations'],
                                    risk_percentage=prediction_data['probability'])
        return render_template('heart.html')
    
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Validate required fields
        required_fields = [
            'Age', 'Gender', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
            'ST_Slope', 'MajorVessels', 'Thalassemia'
        ]
        
        for field in required_fields:
            if field not in data:
                error_msg = f'Missing required field: {field}'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': error_msg})
                flash(error_msg, 'error')
                return redirect(url_for('heartPage'))

        # Convert gender to numeric value
        gender_value = data['Gender']
        if isinstance(gender_value, str):
            gender_value = 1 if gender_value.lower() in ['male', 'm', '1'] else 0

        # Create prediction list in correct order
        try:
            to_predict_list = [
                float(data['Age']),
                float(gender_value),  # Use converted gender value
                float(data['ChestPainType']),
                float(data['RestingBP']),
                float(data['Cholesterol']),
                float(data['FastingBS']),
                float(data['RestingECG']),
                float(data['MaxHR']),
                float(data['ExerciseAngina']),
                float(data['Oldpeak']),
                float(data['ST_Slope']),
                float(data['MajorVessels']),
                float(data['Thalassemia'])
            ]
        except ValueError as e:
            error_msg = f'Invalid value in form data: {str(e)}'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': error_msg})
            flash(error_msg, 'error')
            return redirect(url_for('heartPage'))
            
        # Make prediction
        prediction = heart_model.predict([to_predict_list])[0]
        probability = heart_model.predict_proba([to_predict_list])[0][1]
        risk_percentage = round(probability * 100, 2)

        # Determine risk level
        if risk_percentage >= 70:
            risk_level = "High"
        elif risk_percentage >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # Get diet recommendations
        diet_recommendations = get_diet_recommendation(
            'Heart Disease',
            risk_level,
            bmi=None,
            gender=gender_value,  # Use converted gender value
            age=float(data['Age'])
        )

        result = {
            'success': True,
            'prediction': int(prediction),
            'probability': risk_percentage,
            'risk_level': risk_level,
            'diet_recommendations': diet_recommendations
        }

        # Store in session for GET request
        session['prediction'] = result

        # Return based on request type
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(result)
        
        # For regular form submit, redirect to results page
        return redirect(url_for('heartpredictPage', results=True))

    except Exception as e:
        error_msg = f'Error processing prediction: {str(e)}'
        print(f"Error in heart prediction: {str(e)}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': error_msg})
        flash(error_msg, 'error')
        return redirect(url_for('heartPage'))

@app.route("/liverpredict", methods=['POST', 'GET'])
def liverpredictPage():
    try:
        # If it's a GET request with results parameter, render the results page
        if request.method == 'GET' and request.args.get('results'):
            # Get the stored prediction data from sessionStorage
            prediction_data = request.args.get('data')
            if prediction_data:
                try:
                    data = json.loads(prediction_data)
                    return render_template('liver_predict.html',
                                         pred=data.get('pred', ''),
                                         diet=data.get('diet', []),
                                         risk_percentage=data.get('risk_percentage', 0))
                except json.JSONDecodeError:
                    pass
            return render_template('liver_predict.html')
            
        if request.method == 'POST':
            # Handle both form data and JSON data
            if request.is_json:
                to_predict_dict = request.get_json()
            else:
                to_predict_dict = request.form.to_dict()
            
            # Extract gender and age
            gender = to_predict_dict.get('Gender', '0')  # Default to '0' if not provided
            age = float(to_predict_dict.get('Age', 0))

            # Remove disease_type if present
            if 'disease_type' in to_predict_dict:
                del to_predict_dict['disease_type']

            # Convert values and predict
            for key, value in to_predict_dict.items():
                try:
                    # Handle empty or invalid values
                    if not value or value == "Not found":
                        return jsonify({
                            'success': False,
                            'error': f"Missing or invalid value for {key}"
                        })
                    
                    # Skip gender conversion as it's handled separately
                    if key == 'Gender':
                        continue
                    
                    # Convert to float, handling decimal values
                    to_predict_dict[key] = float(value)
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': f"Invalid value for {key}: {value}"
                    })

            # Convert gender to numeric value
            gender = 1 if str(gender).lower() in ['male', 'm', '1'] else 0

            # Create prediction list in the correct order
            to_predict_list = [
                age,
                gender,
                to_predict_dict['Total_Bilirubin'],
                to_predict_dict['Direct_Bilirubin'],
                to_predict_dict['Alkaline_Phosphotase'],
                to_predict_dict['Alamine_Aminotransferase'],
                to_predict_dict['Aspartate_Aminotransferase'],
                to_predict_dict['Total_Protiens'],
                to_predict_dict['Albumin'],
                to_predict_dict['Albumin_and_Globulin_Ratio']
            ]
            
            # Make prediction
            prediction = liver_model.predict([to_predict_list])
            probability = liver_model.predict_proba([to_predict_list])
            
            # Calculate risk level and percentage
            risk_percentage = probability[0][1] * 100
            if risk_percentage > 70:
                risk_level = 'High'
            elif risk_percentage > 30:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'

            prediction_result = {
                'disease': 'Liver Disease',
                'prediction': bool(prediction[0]),
                'probability': risk_percentage,
                'level': risk_level,
                'message': f"Risk of Liver Disease: {risk_percentage:.1f}%"
            }

            # Get diet recommendation
            diet_recommendation = get_diet_recommendation(
                'Liver Disease',
                risk_level,
                bmi=None,  # BMI not available for liver test
                gender=gender,
                age=age
            )

            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'pred': prediction_result['message'],
                    'diet': diet_recommendation,
                    'risk_percentage': risk_percentage
                })

            return render_template('liver_predict.html', 
                                 pred=prediction_result['message'],
                                 diet=diet_recommendation,
                                 risk_percentage=risk_percentage)

    except Exception as e:
        print(f"Error in liver prediction route: {str(e)}")
        traceback.print_exc()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': str(e)
            })
        return render_template("home.html", message=f"Error: {str(e)}")

    return render_template('home.html')

@app.route("/diabetespredict", methods=['POST', 'GET'])
def diabetespredictPage():
    # ... existing code ...
    return render_template('diabetes_predict.html', 
                         pred=prediction_result['message'],
                         diet=diet_recommendation,
                         bmi=bmi_display,
                         bmi_category=bmi_category,
                         risk_percentage=risk_percentage)

if __name__ == '__main__':
    app.run(debug = True)