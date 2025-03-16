from PIL import Image, ImageDraw, ImageFont
import os

def create_dummy_report():
    # Create a white background image
    width = 1000
    height = 1400
    background_color = (255, 255, 255)
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load Arial font
        font = ImageFont.truetype("arial.ttf", 32)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Add hospital header
    draw.text((50, 50), "MEDICAL HEART TEST REPORT", fill=(0, 0, 0), font=font)
    draw.text((50, 100), "Hospital: City Medical Center", fill=(0, 0, 0), font=small_font)
    draw.text((50, 130), "Date: 2024-02-20", fill=(0, 0, 0), font=small_font)
    
    # Add patient information
    draw.text((50, 200), "PATIENT INFORMATION:", fill=(0, 0, 0), font=font)
    draw.text((50, 250), "Age: 45", fill=(0, 0, 0), font=small_font)
    draw.text((50, 280), "Sex: Male", fill=(0, 0, 0), font=small_font)
    
    # Add test results
    draw.text((50, 350), "CARDIAC ASSESSMENT:", fill=(0, 0, 0), font=font)
    
    test_results = [
        "Chest Pain Type (CP): 2",
        "Resting Blood Pressure: 130 mm Hg",
        "Serum Cholesterol: 236 mg/dl",
        "Fasting Blood Sugar: >120 mg/dl",
        "Resting ECG Results: 1",
        "Maximum Heart Rate (Thalach): 150",
        "Exercise Induced Angina: 0",
        "ST Depression (Oldpeak): 2.3",
        "Slope of Peak Exercise ST Segment: 1",
        "Number of Major Vessels (CA): 2",
        "Thalassemia (Thal): 3"
    ]
    
    y_position = 400
    for result in test_results:
        draw.text((50, y_position), result, fill=(0, 0, 0), font=small_font)
        y_position += 40
    
    # Add footer
    draw.text((50, height-100), "This is a computer generated report.", fill=(0, 0, 0), font=small_font)
    
    # Save the image
    if not os.path.exists('test_reports'):
        os.makedirs('test_reports')
    
    image_path = 'test_reports/dummy_heart_report.png'
    image.save(image_path)
    print(f"Report generated and saved as {image_path}")
    return image_path

if __name__ == "__main__":
    create_dummy_report()