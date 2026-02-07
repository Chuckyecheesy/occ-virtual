"""
affordability_model.py
Adaptable AI-powered student rent advisor.
Uses historical data (CSV) and predicts safe rent.
Filters sublets.csv for apartments under predicted rent.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# ---------- Step 1: Load historical student data ----------
def load_historical_data(csv_file='historical_students.csv'):
    """
    Loads historical student financial data for training ML model.
    If CSV not found, uses a small default dataset.
    """
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Small default dataset for hackathon
        data = [
            [8000, 1500, 1000, 0, 2000, 1500, 8, 800],
            [9000, 2000, 0, 0, 1000, 3000, 8, 750],
            [7000, 500, 800, 500, 1000, 1000, 8, 600],
        ]
        columns = ['tuition','bank_balance','part_time_income','internship_income','scholarships','loans','months','safe_rent']
        df = pd.DataFrame(data, columns=columns)
    df['safe_rent'] = pd.to_numeric(df['safe_rent'], errors='coerce')
    df = df.dropna(subset=['safe_rent'])
    return df

# ---------- Step 2: Train regression model ----------
def train_model(df_train):
    X = df_train.drop('safe_rent', axis=1)
    y = df_train['safe_rent']
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------- Step 3: Load sublets ----------
def load_sublets(csv_file='sublets.csv'):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df['monthly_rent'] = pd.to_numeric(df['monthly_rent'], errors='coerce')
        df = df.dropna(subset=['monthly_rent'])
        return df
    else:
        print("Warning: sublets.csv not found. No apartments loaded.")
        return pd.DataFrame(columns=['address','monthly_rent'])

# ---------- Step 4: Predict safe rent ----------
def predict_safe_rent(model, user_input: dict):
    features = [[
        user_input.get('tuition',0),
        user_input.get('bank_balance',0),
        user_input.get('part_time_income',0),
        user_input.get('internship_income',0),
        user_input.get('scholarships',0),
        user_input.get('loans',0),
        user_input.get('months',8)
    ]]
    rent = model.predict(features)[0]
    return round(max(rent,0.0),2)

# ---------- Step 5: Recommend apartments ----------
def recommend_apartments(safe_rent, sublets_df):
    return sublets_df[sublets_df['monthly_rent'] <= safe_rent]

# ---------- Step 6: Example usage ----------
if __name__ == "__main__":
    # Load data and train model
    df_train = load_historical_data()
    model = train_model(df_train)
    
    # Example user input
    user_input = {
        'tuition': 8000,
        'bank_balance': 1500,
        'part_time_income': 1000,
        'internship_income': 0,
        'scholarships': 2000,
        'loans': 1500,
        'months': 8
    }
    
    # Predict safe rent
    safe_rent = predict_safe_rent(model, user_input)
    print(f"Predicted safe monthly rent (AI): ${safe_rent}")
    
    # Load apartments and recommend
    sublets_df = load_sublets()
    recommendations = recommend_apartments(safe_rent, sublets_df)
    
    print(f"Found {len(recommendations)} apartments under ${safe_rent}:")
    print(recommendations)
