import streamlit as st
from audio_repeat import speak_text, clean_float, listen_and_transcribe
from affordability_model import load_historical_data, train_model, load_sublets, recommend_apartments, predict_safe_rent

# -------------------------
# Session state initialization
# -------------------------
if 'q_index' not in st.session_state:
    st.session_state.q_index = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'tone' not in st.session_state:
    st.session_state.tone = 'neutral'
if 'finished' not in st.session_state:
    st.session_state.finished = False
if 'tone_confirmed' not in st.session_state:
    st.session_state.tone_confirmed = False
if 'results_spoken' not in st.session_state:
    st.session_state.results_spoken = False
if 'fast_demo' not in st.session_state:
    st.session_state.fast_demo = True
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True
if 'spoken_raw' not in st.session_state:
    st.session_state.spoken_raw = {}
if 'spoken_value' not in st.session_state:
    st.session_state.spoken_value = {}

questions = [
    "Enter your annual tuition fee:",
    "Enter your current bank balance:",
    "Enter your monthly part-time income:",
    "Enter your monthly internship income:",
    "Enter total received scholarships:",
    "Enter total available loans:",
    "Enter the number of months for which you need housing:"
]

st.title("🏠 Off-Campus Community Virtual Budget Assistant")
st.checkbox("Fast demo mode (reduce latency)", key="fast_demo")
st.checkbox("1-minute demo mode (auto-fill answers)", key="demo_mode")

@st.cache_data
def _load_historical_data():
    return load_historical_data()

@st.cache_resource
def _load_model():
    df_train = _load_historical_data()
    return train_model(df_train)

@st.cache_data
def _load_sublets():
    return load_sublets()

# -------------------------
# Step 1: Voice tone selection (3 buttons → go to 1st question)
# -------------------------
if not st.session_state.tone_confirmed:
    st.subheader("Choose your preferred voice tone")
    st.caption("Click a button to set the assistant's tone and start the questions.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 Friendly", use_container_width=True):
            st.session_state.tone = "friendly"
            st.session_state.tone_confirmed = True
            st.session_state.q_index = 1
            speak_text(
                "You selected friendly tone. Let's start with the first question.",
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )
    with col2:
        if st.button("💼 Professional", use_container_width=True):
            st.session_state.tone = "professional"
            st.session_state.tone_confirmed = True
            st.session_state.q_index = 1
            speak_text(
                "You selected professional tone. Let's start with the first question.",
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )
    with col3:
        if st.button("😐 Neutral", use_container_width=True):
            st.session_state.tone = "neutral"
            st.session_state.tone_confirmed = True
            st.session_state.q_index = 1
            speak_text(
                "You selected neutral tone. Let's start with the first question.",
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )

# Quick demo auto-fill

if st.session_state.demo_mode and not st.session_state.finished:
    if st.button("Auto-fill demo answers (finish now)", use_container_width=True):
        st.session_state.answers = {
            0: 12000.0,
            1: 1500.0,
            2: 800.0,
            3: 1200.0,
            4: 2000.0,
            5: 3000.0,
            6: 8.0
        }
        if not st.session_state.tone_confirmed:
            st.session_state.tone = "neutral"
            st.session_state.tone_confirmed = True
        st.session_state.q_index = len(questions)
        st.session_state.finished = True
        st.session_state.results_spoken = False
        st.rerun()
# -------------------------
# Step 2: Question display (voice reads question → Type or Speak → proceed)
# -------------------------
if st.session_state.tone_confirmed and not st.session_state.finished:
    q_idx = st.session_state.q_index - 1
    question = questions[q_idx]

    st.subheader(f"Question {st.session_state.q_index} of {len(questions)}")
    st.write(question)

    # Speak the question once when entering this question (skip in demo mode)
    if not st.session_state.demo_mode:
        if 'spoken_q' not in st.session_state or st.session_state.spoken_q != q_idx:
            speak_text(
                question,
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )
            st.session_state.spoken_q = q_idx

    # Input method: Type or Speak
    answer_method = st.radio("Answer by:", options=["Type", "Speak"], key=f"method_{q_idx}", horizontal=True)

    if answer_method == "Type":
        typed_answer = st.text_input("Type your answer (e.g. 100 or one hundred) and press Enter:", key=f"type_{q_idx}", placeholder="Enter a number...")
        if st.button("Submit Answer", key=f"submit_{q_idx}"):
            if typed_answer is None or (isinstance(typed_answer, str) and not typed_answer.strip()):
                st.warning("Please type your answer before submitting.")
            else:
                st.session_state.answers[q_idx] = clean_float(typed_answer)
                if not st.session_state.fast_demo:
                    speak_text(f"You entered {typed_answer}.", tone=st.session_state.tone)
                if st.session_state.q_index < len(questions):
                    st.session_state.q_index += 1
                else:
                    st.session_state.finished = True
                st.rerun()
    else:
        # Speak: record voice, convert words to number (e.g. "one hundred" → 100)
        if st.session_state.get("speak_error"):
            st.warning(st.session_state.speak_error)
            st.session_state.speak_error = None
        if st.button("🎤 Start Speaking", key=f"speak_btn_{q_idx}"):
            with st.spinner("Listening... say your answer now."):
                if not st.session_state.fast_demo:
                    speak_text("Listening. Say your answer now.", tone=st.session_state.tone)
                raw = listen_and_transcribe(duration=3 if st.session_state.fast_demo else 5)
                if raw:
                    numeric = clean_float(raw)
                    st.session_state.spoken_raw[q_idx] = raw
                    st.session_state.spoken_value[q_idx] = numeric
                else:
                    st.session_state.speak_error = "We didn't catch that. Please try again."

        if q_idx in st.session_state.get('spoken_raw', {}):
            raw = st.session_state.spoken_raw[q_idx]
            num = st.session_state.spoken_value[q_idx]
            st.success(f"You said: **{raw}** → **{num}**")
            if st.button("Submit Answer", key=f"submit_{q_idx}"):
                st.session_state.answers[q_idx] = num
                if not st.session_state.fast_demo:
                    speak_text(f"You said {raw}.", tone=st.session_state.tone)
                if st.session_state.q_index < len(questions):
                    st.session_state.q_index += 1
                else:
                    st.session_state.finished = True
                st.rerun()

# -------------------------
# Step 3: Visualization + budget + recommendations (read aloud)
# -------------------------
if st.session_state.finished:
    user_input = {
        'tuition': st.session_state.answers[0],
        'bank_balance': st.session_state.answers[1],
        'part_time_income': st.session_state.answers[2],
        'internship_income': st.session_state.answers[3],
        'scholarships': st.session_state.answers[4],
        'loans': st.session_state.answers[5],
        'months': int(st.session_state.answers[6])
    }

    model = _load_model()
    safe_rent = predict_safe_rent(model, user_input)
    sublets_df = _load_sublets()
    recommendations = recommend_apartments(safe_rent, sublets_df)

    st.subheader("Your budget & recommended housing")
    st.metric("Suggested monthly rent budget", f"${safe_rent:,.2f}")

    # Visualization: budget vs recommended apartments
    if not recommendations.empty:
        cols = list(recommendations.columns)
        if 'address' in cols and 'monthly_rent' in cols:
            chart_df = recommendations[['address', 'monthly_rent']].head(10)
            chart_df = chart_df.rename(columns={'address': 'Address', 'monthly_rent': 'Rent ($)'})
            st.bar_chart(chart_df.set_index('Address'))
        st.write("**Recommended off-campus options within your budget:**")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
        addresses = ", ".join(recommendations["address"].tolist())
        summary_text = f"Your suggested monthly rent budget is ${safe_rent:,.2f}. Recommended housing options within your budget are: {addresses}."
    else:
        st.info("No listings found within your budget. Try adjusting your inputs or check back later.")
        summary_text = f"Your suggested monthly rent budget is ${safe_rent:,.2f}. No suitable housing options were found within your budget."

    # Read results aloud once
    if not st.session_state.results_spoken:
        speak_text(summary_text, tone=st.session_state.tone, async_playback=st.session_state.fast_demo)
        st.session_state.results_spoken = True
