import os
from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from ocr import preprocess_image, extract_text_from_image, extract_medical_fields
import joblib

app = Flask(__name__)
#app.secret_key = 'your_secret_key_here'  # Required for flash messages

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads directory exists
app.secret_key = 'your_secret_key_here'  # Required for flash messages
def predict(values, dic):
    # diabetes
    if len(values) == 8:
        try:
            features = np.array([[
                float(dic['Pregnancies']),
                float(dic['Glucose']),
                float(dic['BloodPressure']),
                float(dic['SkinThickness']),
                float(dic['Insulin']),
                float(dic['BMI']),
                float(dic['DiabetesPedigreeFunction']),
                float(dic['Age'])
            ]])

            model = joblib.load('models/diabetes.pkl')
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            risk_percentage = round(prediction_proba[1] * 100, 2)

            if prediction == 1:
                if risk_percentage > 75:
                    return f"Diabetes Detected (High Risk - {risk_percentage}%)"
                elif risk_percentage > 50:
                    return f"Diabetes Detected (Moderate Risk - {risk_percentage}%)"
                else:
                    return f"Diabetes Detected (Low Risk - {risk_percentage}%)"
            else:
                return f"No Diabetes Detected (Risk: {risk_percentage}%)"

        except Exception as e:
            print(f"Error in diabetes prediction: {str(e)}")
            raise e

    # breast_cancer
    elif len(values) == 22:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # heart disease
    elif len(values) == 13:
        try:
            features = np.array([[
                float(dic['age']),
                float(dic['sex']),
                float(dic['cp']),
                float(dic['trestbps']),
                float(dic['chol']),
                float(dic['fbs']),
                float(dic['restecg']),
                float(dic['thalach']),
                float(dic['exang']),
                float(dic['oldpeak']),
                float(dic['slope']),
                float(dic['ca']),
                float(dic['thal'])
            ]])

            model = joblib.load('models/heart.pkl')
            scaler = joblib.load('models/heart_scaler.pkl')
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            risk_percentage = round(prediction_proba[1] * 100, 2)

            if prediction == 1:
                if risk_percentage > 75:
                    return f"Heart Disease Detected (High Risk - {risk_percentage}%)"
                elif risk_percentage > 50:
                    return f"Heart Disease Detected (Moderate Risk - {risk_percentage}%)"
                else:
                    return f"Heart Disease Detected (Low Risk - {risk_percentage}%)"
            else:
                return f"No Heart Disease Detected (Risk: {risk_percentage}%)"

        except Exception as e:
            print(f"Error in heart disease prediction: {str(e)}")
            raise e

    # kidney disease
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # liver disease
    elif len(values) == 10:
        try:
            features = np.array([[
                float(dic['Age']),
                float(dic['Gender']),
                float(dic['Total_Bilirubin']),
                float(dic['Direct_Bilirubin']),
                float(dic['Alkaline_Phosphotase']),
                float(dic['Alamine_Aminotransferase']),
                float(dic['Aspartate_Aminotransferase']),
                float(dic['Total_Protiens']),
                float(dic['Albumin']),
                float(dic['Albumin_and_Globulin_Ratio'])
            ]])

            # Load model only
            model = joblib.load('models/liver.pkl')
            
            # Make prediction directly without scaling
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            risk_percentage = round(prediction_proba[1] * 100, 2)

            if prediction == 1:
                if risk_percentage > 75:
                    return f"Liver Disease Detected (High Risk - {risk_percentage}%)"
                elif risk_percentage > 50:
                    return f"Liver Disease Detected (Moderate Risk - {risk_percentage}%)"
                else:
                    return f"Liver Disease Detected (Low Risk - {risk_percentage}%)"
            else:
                return f"No Liver Disease Detected (Risk: {risk_percentage}%)"

        except Exception as e:
            print(f"Error in liver disease prediction: {str(e)}")
            raise e

    return "Error: Invalid input dimensions"

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
            
            # Extract BMI display value
            bmi_display = to_predict_dict.get('bmi_display', None)
            
            # Remove any non-prediction fields
            if 'bmi_display' in to_predict_dict:
                del to_predict_dict['bmi_display']
            if 'height' in to_predict_dict:
                del to_predict_dict['height']
            if 'weight' in to_predict_dict:
                del to_predict_dict['weight']

            print("Form data:", to_predict_dict)  # Debug print

            # Convert values to float
            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = float(value)
                except ValueError as e:
                    print(f"Error converting {key}: {value}")
                    raise ValueError(f"Invalid value for {key}: {value}")

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            
            print("Prediction input list:", to_predict_list)  # Debug print
            
            pred = predict(to_predict_list, to_predict_dict)
            
            # Calculate BMI category if BMI is available
            bmi_category = None
            if bmi_display:
                try:
                    bmi_value = float(bmi_display)
                    if bmi_value < 18.5:
                        bmi_category = "Underweight"
                    elif 18.5 <= bmi_value < 25:
                        bmi_category = "Normal weight"
                    elif 25 <= bmi_value < 30:
                        bmi_category = "Overweight"
                    else:
                        bmi_category = "Obese"
                except ValueError:
                    bmi_category = None

            return render_template('predict.html', 
                                pred=pred, 
                                bmi=bmi_display, 
                                bmi_category=bmi_category)
                                
    except Exception as e:
        print(f"Error in prediction route: {str(e)}")  # Debug print
        message = f"Error: {str(e)}"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=None)

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

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image']).convert('L')
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img))
        except:
            message = "Please upload an image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia_predict.html', pred=pred)

if __name__ == '__main__':
    app.run(debug = True)