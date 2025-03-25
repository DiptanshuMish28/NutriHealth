import os
from flask import Flask, render_template, request, flash, jsonify
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from ocr import preprocess_image, extract_text_from_image, extract_medical_fields
import joblib
from werkzeug.utils import secure_filename
import traceback

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
        'Heart Disease': {
            'High': [
                "🫀 Cardiac Diet Guidelines:\n"
                "• Limit daily sodium to 1,500-2,000mg\n"
                "• Restrict saturated fats to less than 6% of daily calories\n"
                "• Aim for 25-30g of fiber daily\n"
                "• Keep cholesterol under 200mg daily",

                "✅ Heart-Healthy Foods:\n"
                "• Lean proteins: Fish (especially salmon, mackerel), skinless poultry\n"
                "• Whole grains: Oats, quinoa, brown rice\n"
                "• Vegetables: Leafy greens, broccoli, carrots\n"
                "• Fruits: Berries, citrus fruits, apples\n"
                "• Healthy fats: Olive oil, avocados, nuts\n\n"
                "Sample Meals:\n"
                "Breakfast: Oatmeal with berries and nuts\n"
                "Lunch: Grilled fish with quinoa and vegetables\n"
                "Dinner: Lean chicken breast with sweet potato",

                "❌ Foods to Avoid:\n"
                "• Processed meats (bacon, sausage)\n"
                "• Full-fat dairy products\n"
                "• Fried foods\n"
                "• Foods high in sodium\n"
                "• Sugary beverages and snacks",

                "📊 Daily Monitoring:\n"
                "• Blood pressure readings\n"
                "• Salt intake tracking\n"
                "• Physical activity (aim for 30 mins daily)\n"
                "• Weight monitoring\n\n"
                "Target Numbers:\n"
                "• Blood Pressure: Below 120/80 mmHg\n"
                "• Resting Heart Rate: 60-100 bpm\n"
                "• Cholesterol: LDL < 100 mg/dL",

                "🌿 Supplements (consult doctor):\n"
                "• Omega-3: 1,000-2,000mg daily\n"
                "• CoQ10: 100-200mg daily\n"
                "• Magnesium: 400mg daily\n"
                "• Vitamin D: 1,000-2,000 IU daily"
            ],
            'Moderate': [
                "🫀 Modified Heart-Healthy Guidelines:\n"
                "• Limit sodium to 2,000-2,300mg daily\n"
                "• Keep saturated fats under 10% of daily calories\n"
                "• Aim for 20-25g of fiber daily",

                "✅ Recommended Foods:\n"
                "• Fish twice weekly\n"
                "• Daily servings of fruits and vegetables\n"
                "• Whole grains\n"
                "• Low-fat dairy products",

                "❌ Foods to Limit:\n"
                "• Red meat (limit to 1-2 times per week)\n"
                "• Processed foods\n"
                "• Added sugars\n"
                "• High-sodium foods",

                "📊 Monitoring:\n"
                "• Regular blood pressure checks\n"
                "• Weekly weight monitoring\n"
                "• Physical activity tracking"
            ],
            'Low': [
                "🫀 Preventive Diet Guidelines:\n"
                "• Maintain a balanced diet\n"
                "• Focus on portion control\n"
                "• Include variety of foods",

                "✅ Healthy Choices:\n"
                "• Regular fish consumption\n"
                "• Plenty of fruits and vegetables\n"
                "• Whole grain options\n"
                "• Healthy cooking methods",

                "📊 General Monitoring:\n"
                "• Annual health check-ups\n"
                "• Regular exercise routine\n"
                "• Stress management"
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

            return render_template(template,
                                pred=prediction_result.get('message', ''),
                                diet=diet_recommendation,
                                bmi=bmi_display,
                                bmi_category=bmi_category,
                                risk_percentage=prediction_result.get('risk', 0))

    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        traceback.print_exc()
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
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                return render_template('pneumonia.html', message='No file selected')
            
            file = request.files['image']
            if file.filename == '':
                return render_template('pneumonia.html', message='No file selected')

            # Process image and get prediction
            try:
                image = Image.open(file.stream)
                image = image.convert('RGB')  # Convert to RGB
                image = image.resize((300, 300))  # Resize to match model's expected input
                
                # Convert to numpy array and normalize
                img_array = np.array(image)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Load model and predict
                model = tf.keras.models.load_model("models/trained.h5")
                pred = model.predict(img_array)
                pred = "Pneumonia Detected" if pred[0][0] > 0.5 else "Normal"
                
                return render_template('pneumonia_predict.html', pred=pred)
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return render_template('pneumonia.html', message=f'Error processing image: {str(e)}')

    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        return render_template('pneumonia.html', message=f'Error: {str(e)}')

    return render_template('pneumonia.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        test_type = request.headers.get('X-Test-Type', 'liver')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Save the uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            try:
                # Process the image
                preprocessed_image = preprocess_image(image_path)
                if preprocessed_image is not None:
                    extracted_text = extract_text_from_image(preprocessed_image)
                    if extracted_text:
                        data = extract_medical_fields(extracted_text, test_type)
                        return jsonify({
                            'test_type': test_type.capitalize(),
                            'data': data
                        })
                    else:
                        return jsonify({'error': 'Could not extract text from image'}), 400
                else:
                    return jsonify({'error': 'Could not process image'}), 400

            finally:
                # Clean up the uploaded file
                if os.path.exists(image_path):
                    os.remove(image_path)

    except Exception as e:
        print(f"Error in upload: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route("/heartpredict", methods=['POST', 'GET'])
def heartpredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            
            # Extract BMI and other values
            bmi_display = to_predict_dict.get('bmi_display', None)
            gender = float(to_predict_dict.get('Gender', to_predict_dict.get('sex', 0)))
            age = float(to_predict_dict.get('Age', to_predict_dict.get('age', 0)))

            # Remove non-prediction fields
            if 'bmi_display' in to_predict_dict:
                del to_predict_dict['bmi_display']
            if 'height' in to_predict_dict:
                del to_predict_dict['height']
            if 'weight' in to_predict_dict:
                del to_predict_dict['weight']

            # Convert values and predict
            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = float(value)
                except ValueError:
                    return render_template("home.html", message=f"Invalid value for {key}: {value}")

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            
            # Make prediction
            prediction = heart_model.predict([to_predict_list])
            probability = heart_model.predict_proba([to_predict_list])
            
            # Calculate risk level and percentage
            risk_percentage = probability[0][1] * 100
            if risk_percentage > 70:
                risk_level = 'High'
            elif risk_percentage > 30:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'

            prediction_result = {
                'disease': 'Heart Disease',
                'prediction': bool(prediction[0]),
                'probability': risk_percentage,
                'level': risk_level,
                'message': f"Risk of Heart Disease: {risk_percentage:.1f}%"
            }

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
                'Heart Disease',
                risk_level,
                bmi_value,
                gender,
                age
            )

            return render_template('heart_predict.html', 
                                 pred=prediction_result['message'],
                                 diet=diet_recommendation,
                                 bmi=bmi_display,
                                 bmi_category=bmi_category,
                                 risk_percentage=risk_percentage)

    except Exception as e:
        print(f"Error in heart prediction route: {str(e)}")
        traceback.print_exc()
        return render_template("home.html", message=f"Error: {str(e)}")

    return render_template('home.html')  # Changed to return to home.html

@app.route("/liverpredict", methods=['POST', 'GET'])
def liverpredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            
            # Extract BMI and other values
            bmi_display = to_predict_dict.get('bmi_display', None)
            gender = float(to_predict_dict.get('Gender', to_predict_dict.get('sex', 0)))
            age = float(to_predict_dict.get('Age', to_predict_dict.get('age', 0)))

            # Remove non-prediction fields
            if 'bmi_display' in to_predict_dict:
                del to_predict_dict['bmi_display']
            if 'height' in to_predict_dict:
                del to_predict_dict['height']
            if 'weight' in to_predict_dict:
                del to_predict_dict['weight']

            # Convert values and predict
            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = float(value)
                except ValueError:
                    return render_template("home.html", message=f"Invalid value for {key}: {value}")

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            
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
                'Liver Disease',
                risk_level,
                bmi_value,
                gender,
                age
            )

            return render_template('liver_predict.html', 
                                 pred=prediction_result['message'],
                                 diet=diet_recommendation,
                                 bmi=bmi_display,
                                 bmi_category=bmi_category,
                                 risk_percentage=risk_percentage)

    except Exception as e:
        print(f"Error in liver prediction route: {str(e)}")
        traceback.print_exc()
        return render_template("home.html", message=f"Error: {str(e)}")

    return render_template('home.html')  # Changed to return to home.html

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