import pytesseract
from PIL import Image
import os
import re

folder_path = 'raw_extract_pricing/'

# Test just the FIRST image
test_file = [f for f in os.listdir(folder_path) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))][0]

img_path = os.path.join(folder_path, test_file)
img = Image.open(img_path)

print(f"Testing file: {test_file}")
print(f"Image size: {img.size}\n")

# Get OCR text WITHOUT cropping
text = pytesseract.image_to_string(img, config='--psm 6')

print("="*80)
print("FULL OCR OUTPUT:")
print("="*80)
print(text)
print("="*80)

# Find all numbers that look like prices
price_pattern = r'\$\s*(\d{1,5}(?:,\d{3})?)\s*(?:/month|/mo)'
matches = re.findall(price_pattern, text, re.IGNORECASE)

print(f"\nPrice matches with current regex: {matches}")

# Find where "23" appears
if "23" in text:
    print(f"\n⚠️  Found '23' in the text - here's the context:")
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if '23' in line:
            print(f"  Line {i}: {line.strip()}")