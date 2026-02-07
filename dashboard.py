# dashboard.py - Streamlit Dashboard with Chatbot Q&A + Snowflake Integration

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from affordability_model import (
    load_historical_data, 
    train_model, 
    predict_safe_rent, 
    recommend_apartments
)
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_repeat import listen_and_transcribe  # your audio recording module
import snowflake.connector

# ================= Page Configuration =================
st.set_page_config(
    page_title="Student Housing Recommendation",
    page_icon="🏠",
    layout="wide"
)

# ================= Snowflake Integration =================
def load_sublets_from_snowflake():
    """
    Connect to Snowflake and load apartment/sublet data dynamically.
    Assumes credentials stored in Streamlit secrets.
    """
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake_user"],
        password=st.secrets["snowflake_password"],
        account=st.secrets["snowflake_account"],
        warehouse=st.secrets["snowflake_warehouse"],
        database=st.secrets["snowflake_database"],
        schema=st.secrets["snowflake_schema"]
    )
    query = "SELECT * FROM sublets_table"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ================= Model Initialization =================
@st.cache_resource
def initialize_model():
    df_train = load_historical_data()
    model = train_model(df_train)
    return model

@st.cache_data
def load_apartments_data():
    return load_sublets_from_snowflake()

# Load model and apartments
model = initialize_model()
apartments_df = load_apartments_data()

# ================= Chatbot Interaction =================
st.title("🏠 Off-Campus Housing Recommendation Tool")
st.markdown("""
Welcome! I am your student housing assistant. Let's find the best affordable apartments for you.  
Please answer the following questions.
""")

def get_user_input():
    """
    Prompt user via chatbot (text + optional audio) to gather financial info.
    """
    # Use audio_record module if user wants audio input
    use_audio = st.checkbox("Answer via voice", value=False)
    
    input_data = {}
    
    questions = [
        ("tuition", "Enter your annual tuition fee ($)"),
        ("bank_balance", "Enter your current bank balance ($)"),
        ("part_time_income", "Enter your monthly part-time income ($)"),
        ("internship_income", "Enter your monthly internship income ($)"),
        ("scholarships", "Enter total scholarships received ($)"),
        ("loans", "Enter total available loans ($)"),
        ("months", "Enter the number of months for which you need housing")
    ]
    
    for key, question in questions:
        st.markdown(f"**{question}**")
        if use_audio:
            # Record audio and transcribe
            value = listen_and_transcribe()
            try:
                # Convert numeric responses to float or int
                if key == "months":
                    value = int(value)
                else:
                    value = float(value)
            except:
                st.warning(f"Could not interpret your response: {value}. Please type it manually.")
                value = st.number_input(question, min_value=0, step=100 if key != "months" else 1)
        else:
            value = st.number_input(question, min_value=0, step=100 if key != "months" else 1)
        
        input_data[key] = value
    
    return input_data

user_input = get_user_input()

# ================= Predict Safe Rent =================
safe_rent = predict_safe_rent(model, user_input)

# Filter apartments under safe rent
recommended_apts = recommend_apartments(safe_rent, apartments_df)

# ================= Display Metrics =================
st.markdown("---")
col_metric1, col_metric2, col_metric3 = st.columns(3)

with col_metric1:
    st.metric("AI-Predicted Safe Rent", f"${safe_rent:.2f}", delta="per month", delta_color="off")
with col_metric2:
    st.metric("Total Monthly Income", f"${user_input['part_time_income'] + user_input['internship_income']:.2f}")
with col_metric3:
    st.metric("Available Apartments", len(recommended_apts), delta="under budget", delta_color="off")

# ================= Display Recommendations =================
st.markdown("---")
st.header("📋 Recommended Apartments")

if len(recommended_apts) > 0:
    display_df = recommended_apts.copy()
    display_df['monthly_rent'] = display_df['monthly_rent'].apply(lambda x: f"${x:.2f}")
    st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)

    # Bar chart visualization
    st.subheader("🎯 Apartment Prices Comparison")
    chart_data = recommended_apts.sort_values('monthly_rent')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(chart_data['address'], chart_data['monthly_rent'], color='steelblue')
    ax.axvline(x=safe_rent, color='red', linestyle='--', linewidth=2, label='Safe Rent Limit')
    ax.set_xlabel('Monthly Rent ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Address', fontsize=12, fontweight='bold')
    ax.set_title('Recommended Apartments - Monthly Rent', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning(f"No apartments found under ${safe_rent:.2f}/month")
    if len(apartments_df) > 0:
        st.info(f"Available apartments range from ${apartments_df['monthly_rent'].min():.2f} to ${apartments_df['monthly_rent'].max():.2f}/month")

# ================= Financial Summary =================
st.markdown("---")
st.subheader("📊 Your Financial Summary")

col_summary1, col_summary2, col_summary3 = st.columns(3)

with col_summary1:
    st.write("**Income Sources:**")
    st.write(f"- Part-Time: ${user_input['part_time_income']:.2f}")
    st.write(f"- Internship: ${user_input['internship_income']:.2f}")
    st.write(f"- **Total: ${user_input['part_time_income'] + user_input['internship_income']:.2f}**")

with col_summary2:
    st.write("**Financial Support:**")
    st.write(f"- Scholarships: ${user_input['scholarships']:.2f}")
    st.write(f"- Loans: ${user_input['loans']:.2f}")
    st.write(f"- Bank Balance: ${user_input['bank_balance']:.2f}")

with col_summary3:
    st.write("**Rent Analysis:**")
    st.write(f"- Safe Rent: ${safe_rent:.2f}")
    if len(recommended_apts) > 0:
        min_rent = recommended_apts['monthly_rent'].min()
        max_rent = recommended_apts['monthly_rent'].max()
        st.write(f"- Available Range: ${min_rent:.2f} - ${max_rent:.2f}")
    else:
        st.write("- No options available")

# ================= Footer =================
st.markdown("---")
st.markdown("""
**How It Works:**
1. Answer the chatbot prompts (text or voice) to provide financial information
2. The AI model predicts your safe monthly rent
3. We show all apartments within your budget
4. Compare prices and find the best option

*Last Updated: February 2026*
""")

