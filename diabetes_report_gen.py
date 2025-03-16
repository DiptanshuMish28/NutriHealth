from PIL import Image, ImageDraw, ImageFont
import os

def create_diabetes_report():
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
    draw.text((50, 50), "DIABETES TEST REPORT", fill=(0, 0, 0), font=font)
    draw.text((50, 100), "Hospital: City Medical Center", fill=(0, 0, 0), font=small_font)
    draw.text((50, 130), "Date: 2024-02-20", fill=(0, 0, 0), font=small_font)
    
    # Add patient information
    draw.text((50, 200), "PATIENT INFORMATION:", fill=(0, 0, 0), font=font)
    
    # Add test results
    draw.text((50, 350), "DIABETES ASSESSMENT:", fill=(0, 0, 0), font=font)
    
    test_results = [
        "Pregnancies: 2",
        "Glucose: 140 mg/dL",
        "Blood Pressure: 80 mm Hg",
        "Skin Thickness: 35 mm",
        "Insulin: 150 µU/mL",
        "BMI: 32.5",
        "Diabetes Pedigree Function: 0.627",
        "Age: 45 years"
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
    
    image_path = 'test_reports/dummy_diabetes_report.png'
    image.save(image_path)
    print(f"Report generated and saved as {image_path}")
    return image_path

if __name__ == "__main__":
    create_diabetes_report()