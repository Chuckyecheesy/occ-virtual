import pytesseract
from PIL import Image
import os
import re
import csv

# Folder containing apartment screenshots
folder_path = 'raw_extract_pricing/'
output_file = 'sublets.csv'

data = []

# ---------------------------
# Known company mapping
# ---------------------------
company_lookup = {
    "295 Lester St, Waterloo": "Rent My Space",
    "256 Phillip St, Waterloo": "RezOne",
    "250 Albert St, Waterloo": "Private Apartment Company/Townhouse",
    "433 Keats Way, Waterloo": "Private Apartment Company/Townhouse",
    "365 Albert St, Waterloo": "Accommod8u",
    "388 Hazel St, Waterloo": "Private Apartment Company/Townhouse",
    "277 Lester St, Waterloo": "Private Apartment Company/Townhouse",
    "300 Keats Way, Waterloo": "Private Apartment Company/Townhouse",
    "16 Cardill Cres, Waterloo": "Private Apartment Company/Townhouse",
    "20 Cardill Cres, Waterloo": "Private Apartment Company/Townhouse",
    "251 Hemlock St, Waterloo": "Sage 6 Platinum",
    "280 Phillip St, Waterloo": "WCRI A-Dorm",
    "132 Brighton St, Waterloo": "Private Apartment Company/Townhouse",
    "52 Cardill Cres, Waterloo": "Private Apartment Company/Townhouse",
    "254 Toll Gate Blvd, Waterloo": "Private Apartment Company/Townhouse",
    "271 Westcourt Pl, Waterloo": "Private Apartment Company/Townhouse",
    "320 Westcourt Pl, Waterloo": "Private Apartment Company/Townhouse",
    "261 Lester St, Waterloo": "Private Apartment Company/Townhouse"
}

# ---------------------------
# Price regex patterns
# ---------------------------
price_pattern_1 = r'\$\s*(\d{1,5}(?:,\d{3})?)\s*(?:/\s*month|/\s*mo)'
price_pattern_2 = r'(\d{3,5})\s*/\s*month'

# ---------------------------
# Address regex pattern
# ---------------------------
address_pattern = r'\b(\d{1,4}[A-Z]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:St|Ave|Rd|Dr|Blvd|Cres|Pl|Way|Ct|Ln))),\s*Waterloo\b'

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)

        # OCR extraction
        text = pytesseract.image_to_string(img, config='--psm 6')
        cleaned_text = re.sub(r'\s+', ' ', text)

        # ---------------------------
        # Extract Address
        # ---------------------------
        address_match = re.search(address_pattern, cleaned_text)

        if address_match:
            address = f"{address_match.group(1)} {address_match.group(2)}, Waterloo"
        else:
            address = "Unknown"

        # ---------------------------
        # Extract Price
        # ---------------------------
        price = 0

        price_match = re.search(price_pattern_1, cleaned_text, re.IGNORECASE)
        if price_match:
            price = float(price_match.group(1).replace(',', ''))
        else:
            price_match = re.search(price_pattern_2, cleaned_text, re.IGNORECASE)
            if price_match:
                price = float(price_match.group(1).replace(',', ''))

        # ---------------------------
        # Determine Company
        # ---------------------------
        if address in company_lookup:
            company = company_lookup[address]
        else:
            company = "Private Property"

        # ---------------------------
        # Save Data
        # ---------------------------
        if address != "Unknown":
            data.append({
                "address": address,
                "monthly_rent": price,
                "company": company
            })

            print(f"✓ {address} -> ${price}/month | {company}")

# --- Add synthetic edge-case test data ---
synthetic_entries = [
    {"address": "100 Test St, Waterloo", "monthly_rent": 0, "company": "Foster Residence"},       # zero budget
    {"address": "101 Test St, Waterloo", "monthly_rent": 200, "company": "Private Property"},     # extremely low budget
    {"address": "102 Test St, Waterloo", "monthly_rent": 300, "company": "Private Property"},     # very low budget
    {"address": "103 Test St, Waterloo", "monthly_rent": 400, "company": "Private Property"},     # low budget
    {"address": "104 Test St, Waterloo", "monthly_rent": 500, "company": "Private Property"},     # lower-bound realistic
]

data.extend(synthetic_entries)

# ---------------------------
# Write to CSV
# ---------------------------
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["address", "monthly_rent", "company"]
    )
    writer.writeheader()
    writer.writerows(data)

print(f"\n✓ Extracted {len(data)} listings to {output_file}")

