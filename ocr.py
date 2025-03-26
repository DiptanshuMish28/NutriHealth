import os
import pytesseract
import cv2
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Image preprocessing function
def preprocess_image(image_path):
    try:
        # Open the image file
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert('L')

        # Apply a filter to improve sharpness
        img = img.filter(ImageFilter.SHARPEN)

        # Apply thresholding to binarize the image
        img = img.point(lambda x: 0 if x < 140 else 255)

        # Optionally save preprocessed image for inspection
        img.save('preprocessed_image.png')

        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Extract text using OCR
def extract_text_from_image(preprocessed_image):
    try:
        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(preprocessed_image)
        print("Extracted text:", extracted_text)  # Debug print
        return extracted_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

# Regex-based extraction for Diabetes Test
def extract_diabetes_fields(text):
    patterns = {
        "Pregnancies": r"Pregnancies\s*[:\-\s]\s*(\d+)",
        "Glucose": r"Glucose\s*[:\-\s]\s*([\d.]+)",
        "BloodPressure": r"Blood Pressure\s*[:\-\s]\s*([\d.]+)",
        "SkinThickness": r"Skin Thickness\s*[:\-\s]\s*([\d.]+)",
        "Insulin": r"Insulin\s*[:\-\s]\s*([\d.]+)",
        "BMI": r"BMI\s*[:\-\s]\s*([\d.]+)",
        "DiabetesPedigreeFunction": r"Diabetes Pedigree Function\s*[:\-\s]\s*([\d.]+)",
        "Age": r"Age\s*[:\-\s]\s*(\d+)"
    }

    extracted_values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted_values[field] = match.group(1) if match else ""

    return extracted_values

# Regex-based extraction for Liver Function Test (LFT)
def extract_liver_fields(text):
    # Convert text to uppercase for consistent matching
    text = text.upper()
    
    patterns = {
        "Age": r"AGE/GENDER\s*:\s*(\d+)",
        "Total_Bilirubin": r"TOTAL BILIRUBIN\s*[:\-\s]*([\d.]+)",
        "Direct_Bilirubin": r"DIRECT BILIRUBIN\s*[:\-\s]*([\d.]+)",
        "Alkaline_Phosphotase": r"ALKALINE PHOSPHATASE\s*[:\-\s]*([\d.]+)",
        "Alamine_Aminotransferase": r"SGPT\s*[:\-\s]*([\d.]+)",
        "Aspartate_Aminotransferase": r"SGOT\s*[:\-\s]*([\d.]+)",
        "Total_Protiens": r"TOTAL PROTEINS\s*[:\-\s]*([\d.]+)",
        "Albumin": r"ALBUMIN\s*[:\-\s]*([\d.]+)",
        "Albumin_and_Globulin_Ratio": r"A/G RATIO\s*[:\-\s]*([\d.]+)"
    }

    extracted_values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            
            # Clean and validate each field
            try:
                if field == 'Age':
                    value = str(int(value))
                elif field in ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                             'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
                             'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']:
                    # Handle decimal values
                    value = str(float(value))
                
                extracted_values[field] = value
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Error processing {field}: {e}")
                extracted_values[field] = "Not found"
        else:
            print(f"Pattern not found for {field}")
            extracted_values[field] = "Not found"

    # Add empty Gender field for liver test
    extracted_values['Gender'] = ''

    # Debug print
    print("Extracted text:", text)
    print("Extracted values:", extracted_values)
    
    return extracted_values

# Identify test type based on extracted text
def identify_test_type(text):
    if "Glucose" in text or "Diabetes" in text or "BMI" in text:
        return "Diabetes"
    elif "Bilirubin" in text or "SGOT" in text or "Liver" in text:
        return "Liver"
    return "Unknown"

def extract_medical_fields(extracted_text, test_type='liver'):
    if test_type == 'heart':
        # Convert text to uppercase for consistent matching
        extracted_text = extracted_text.upper()
        
        patterns = {
            "Age": r"AGE:\s*(\d+)",  # Matches "Age: 45"
            "Gender": r"SEX:\s*(MALE|FEMALE|M|F)",  # Matches "Sex: Male"
            "ChestPainType": r"CHEST PAIN TYPE \(CP\):\s*(\d+)",  # Matches "Chest Pain Type (CP): 2"
            "RestingBP": r"RESTING BLOOD PRESSURE:\s*(\d+)\s*(?:MM HG)?",  # Matches "Resting Blood Pressure: 130 mm Hg"
            "Cholesterol": r"SERUM CHOLESTEROL:\s*(\d+)\s*(?:MG/DL)?",  # Matches "Serum Cholesterol: 236 mg/dl"
            "FastingBS": r"FASTING BLOOD SUGAR:\s*[>|<]?\s*(\d+)\s*(?:MG/DL)?",  # Matches "Fasting Blood Sugar: >120 mg/dl"
            "RestingECG": r"RESTING ECG RESULTS:\s*(\d+)",  # Matches "Resting ECG Results: 1"
            "MaxHR": r"MAXIMUM HEART RATE \(THALACH\):\s*(\d+)",  # Matches "Maximum Heart Rate (Thalach): 150"
            "ExerciseAngina": r"EXERCISE INDUCED ANGINA:\s*(\d+)",  # Matches "Exercise Induced Angina: 0"
            "Oldpeak": r"ST DEPRESSION \(OLDPEAK\):\s*([\d.]+)",  # Matches "ST Depression (Oldpeak): 2.3"
            "ST_Slope": r"SLOPE OF PEAK EXERCISE ST SEGMENT:\s*(\d+)"  # Matches "Slope of Peak Exercise ST Segment: 1"
        }

        extracted_values = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, extracted_text)
            if match:
                value = match.group(1).strip()
                
                # Clean and validate each field
                try:
                    if field == 'Gender':
                        value = '1' if value in ['MALE', 'M'] else '0'
                    elif field == 'FastingBS':
                        # Extract number and check if >120
                        num = int(re.search(r'\d+', value).group())
                        value = str(num)  # Store actual value instead of binary
                    elif field in ['ChestPainType', 'RestingECG', 'ST_Slope', 'ExerciseAngina']:
                        value = str(int(value))
                    elif field in ['RestingBP', 'Cholesterol', 'MaxHR']:
                        value = str(int(re.sub(r'[^\d]', '', value)))
                    elif field == 'Oldpeak':
                        value = str(float(value))
                    elif field == 'Age':
                        value = str(int(value))
                    
                    extracted_values[field] = value
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Error processing {field}: {e}")
                    extracted_values[field] = "Not found"
            else:
                print(f"Pattern not found for {field}")
                extracted_values[field] = "Not found"

        # Debug print
        print("Extracted text:", extracted_text)
        print("Extracted values:", extracted_values)
        
        return extracted_values

    elif test_type == 'liver':
        # Convert text to uppercase for consistent matching
        extracted_text = extracted_text.upper()
        
        patterns = {
            "Age": r"AGE/GENDER\s*:\s*(\d+)",
            "Total_Bilirubin": r"TOTAL BILIRUBIN\s*[:\-\s]*([\d.]+)",
            "Direct_Bilirubin": r"DIRECT BILIRUBIN\s*[:\-\s]*([\d.]+)",
            "Alkaline_Phosphotase": r"ALKALINE PHOSPHATASE\s*[:\-\s]*([\d.]+)",
            "Alamine_Aminotransferase": r"SGPT\s*[:\-\s]*([\d.]+)",
            "Aspartate_Aminotransferase": r"SGOT\s*[:\-\s]*([\d.]+)",
            "Total_Protiens": r"TOTAL PROTEINS\s*[:\-\s]*([\d.]+)",
            "Albumin": r"ALBUMIN\s*[:\-\s]*([\d.]+)",
            "Albumin_and_Globulin_Ratio": r"A/G RATIO\s*[:\-\s]*([\d.]+)"
        }
    elif test_type == 'diabetes':
        # Convert text to uppercase for consistent matching
        extracted_text = extracted_text.upper()
        
        patterns = {
            "Pregnancies": r"PREGNANCIES:\s*(\d+)",
            "Glucose": r"GLUCOSE:\s*(\d+)(?:\s*MG/DL)?",
            "BloodPressure": r"BLOOD PRESSURE:\s*(\d+)(?:\s*MM HG)?",
            "SkinThickness": r"SKIN THICKNESS:\s*(\d+)(?:\s*MM)?",
            "Insulin": r"INSULIN:\s*(\d+)(?:\s*[µU]/ML)?",
            "BMI": r"BMI:\s*([\d.]+)",
            "DiabetesPedigreeFunction": r"DIABETES PEDIGREE FUNCTION:\s*([\d.]+)",
            "Age": r"AGE:\s*(\d+)(?:\s*YEARS)?"
        }

        extracted_values = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, extracted_text)
            if match:
                value = match.group(1).strip()
                
                # Clean and validate each field
                try:
                    if field in ['Pregnancies', 'Age']:
                        value = str(int(value))
                    elif field in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']:
                        value = str(int(re.sub(r'[^\d]', '', value)))
                    elif field in ['BMI', 'DiabetesPedigreeFunction']:
                        value = str(float(value))
                    
                    extracted_values[field] = value
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Error processing {field}: {e}")
                    extracted_values[field] = "Not found"
            else:
                print(f"Pattern not found for {field}")
                extracted_values[field] = "Not found"

        # Debug print
        print("Extracted text:", extracted_text)
        print("Extracted values:", extracted_values)
        
        return extracted_values

    extracted_values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Post-process specific fields
            if test_type == 'heart':
                if field == 'Gender':
                    value = '1' if value.lower() in ['male', 'm'] else '0'
                elif field == 'FastingBS':
                    value = '1' if '>120' in value else '0'
            elif test_type == 'diabetes':
                if field == 'Pregnancies':
                    value = int(value)
                elif field == 'Glucose':
                    value = float(value)
                elif field == 'BloodPressure':
                    value = int(value)
                elif field == 'SkinThickness':
                    value = int(value)
                elif field == 'Insulin':
                    value = int(value)
                elif field == 'BMI':
                    value = float(value)
                elif field == 'DiabetesPedigreeFunction':
                    value = float(value)
                elif field == 'Age':
                    value = int(value)
            extracted_values[field] = value
        else:
            extracted_values[field] = "Not found"

    # Add empty Gender field for liver test
    if test_type == 'liver':
        extracted_values['Gender'] = ''

    return extracted_values

# Route to handle file upload and extraction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    preprocessed_image = preprocess_image(file_path)
    if preprocessed_image:
        extracted_text = extract_text_from_image(preprocessed_image)
        if extracted_text:
            test_type = identify_test_type(extracted_text)
            if test_type == "Diabetes":
                extracted_fields = extract_diabetes_fields(extracted_text)
            elif test_type == "Liver":
                extracted_fields = extract_liver_fields(extracted_text)
            else:
                return jsonify({"error": "Test type not recognized"})

            return jsonify({"test_type": test_type, "data": extracted_fields})

    return jsonify({"error": "Failed to extract data"})

# Route to render the diabetes prediction page
@app.route('/')
def home():
    return render_template('diabetes.html')  # Change as needed

if __name__ == '__main__':
    app.run(debug=True)
