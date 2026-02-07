from flask import Flask, jsonify
import csv
import os

app = Flask(__name__)

# Load apartments data from CSV
def load_apartments():
    apartments = []
    csv_path = os.path.join(os.path.dirname(__file__), 'sublets.csv')
    
    if not os.path.exists(csv_path):
        return apartments
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row['monthly_rent'] = float(row['monthly_rent'])
                apartments.append(row)
            except (ValueError, KeyError):
                continue
    
    return apartments

@app.route('/api/affordable/<int:budget>', methods=['GET'])
def get_affordable_apartments(budget):
    """
    Get all apartments with monthly_rent <= budget
    """
    apartments = load_apartments()
    affordable = [apt for apt in apartments if apt.get('monthly_rent', 0) <= budget]
    
    return jsonify({
        'budget': budget,
        'count': len(affordable),
        'apartments': affordable
    })

if __name__ == '__main__':
    app.run(debug=True)
