from flask import Flask, jsonify
import csv
import os

# -------------------- Flask App --------------------
app = Flask(__name__)

# -------------------- Load Apartments --------------------
def load_apartments(csv_file='sublets.csv'):
    """
    Load apartment data from CSV.
    Expects columns: address, monthly_rent, company
    Returns list of dictionaries with float monthly_rent
    """
    apartments = []

    # Ensure CSV exists
    if not os.path.exists(csv_file):
        print(f"[WARNING] CSV file '{csv_file}' not found. Returning empty list.")
        return apartments

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['monthly_rent'] = float(row['monthly_rent'])
                apartments.append(row)
            except (ValueError, KeyError):
                print(f"[WARNING] Skipping row with invalid rent: {row}")
                continue

    print(f"[INFO] Loaded {len(apartments)} apartments from '{csv_file}'")
    return apartments

# -------------------- API Endpoint --------------------
@app.route('/api/affordable/<int:budget>', methods=['GET'])
def get_affordable_apartments(budget):
    """
    Returns all apartments with monthly_rent <= budget as JSON.
    """
    apartments = load_apartments()
    affordable = [apt for apt in apartments if apt.get('monthly_rent', float('inf')) <= budget]

    print(f"[DEBUG] Budget requested: {budget}")
    print(f"[DEBUG] Affordable apartments found: {len(affordable)}")

    return jsonify({
        'budget': budget,
        'count': len(affordable),
        'apartments': affordable
    })

# -------------------- Run Server --------------------
if __name__ == '__main__':
    print("🚀 Starting Flask server for Off-Campus Housing API...")
    print("Endpoint example: http://127.0.0.1:5000/api/affordable/800\n")
    app.run(host='127.0.0.1', port=5000, debug=True)

