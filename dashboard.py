import os
import base64
import datetime as dt
import random
import streamlit as st
from dotenv import load_dotenv
from audio_repeat import speak_text, clean_float, listen_and_transcribe, openrouter_clarify_number
from affordability_model import load_historical_data, train_model, load_sublets, recommend_apartments, predict_safe_rent

try:
    from authlib.integrations.requests_client import OAuth2Session
except ImportError:
    OAuth2Session = None

try:
    import snowflake.connector as snowflake_connector
except ImportError:
    snowflake_connector = None

load_dotenv()

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
if 'read_aloud' not in st.session_state:
    st.session_state.read_aloud = True
if 'tone_prompt_spoken' not in st.session_state:
    st.session_state.tone_prompt_spoken = False
if 'method_prompt_spoken' not in st.session_state:
    st.session_state.method_prompt_spoken = {}

def _parse_voice_list(env_key, fallback_key=None):
    raw = os.getenv(env_key, "")
    ids = [v.strip() for v in raw.split(",") if v.strip()] if raw else []
    if not ids and fallback_key:
        fallback = os.getenv(fallback_key)
        if fallback:
            ids = [fallback.strip()]
    return ids

def _init_voice_map():
    if "voice_map" in st.session_state:
        return
    if "voice_seed" not in st.session_state:
        st.session_state.voice_seed = os.urandom(8).hex()
    rng = random.Random(st.session_state.voice_seed)
    voice_lists = {
        "friendly": _parse_voice_list("VOICE_FRIENDLY_IDS", "VOICE_FRIENDLY"),
        "professional": _parse_voice_list("VOICE_PROFESSIONAL_IDS", "VOICE_PROFESSIONAL"),
        "neutral": _parse_voice_list("VOICE_NEUTRAL_IDS"),
    }
    st.session_state.voice_map = {
        tone: rng.choice(ids) for tone, ids in voice_lists.items() if ids
    }

def _speak(text, tone=None, async_playback=False):
    t = tone or st.session_state.tone
    voice_id = st.session_state.voice_map.get(t)
    # Always enqueue audio to avoid overlap; order is preserved by the queue.
    speak_text(text, tone=t, async_playback=True, voice_id=voice_id)

_init_voice_map()

def _set_background(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .block-container {{
                max-width: 1200px;
                margin: 0 auto;
                padding-top: 3rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }}
            .frame {{
                background: #cfcfcf;
                border: 4px solid #000;
                border-radius: 0;
                padding: 24px 28px;
                margin: 20px auto;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
            }}
            .frame-title {{
                font-weight: 700;
                font-size: 1.1rem;
                margin-bottom: 12px;
                padding-bottom: 10px;
                border-bottom: 4px solid #000;
            }}
            .frame .stButton > button {{
                border: 3px solid #000 !important;
            }}
            .frame .stCaption, .frame .stSubheader, .frame .stMarkdown {{
                padding-bottom: 8px;
                border-bottom: 3px solid #000;
                margin-bottom: 12px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

def _frame_start(title):
    st.markdown(f"<div class='frame'><div class='frame-title'>{title}</div>", unsafe_allow_html=True)

def _frame_end():
    st.markdown("</div>", unsafe_allow_html=True)

_set_background(os.path.join(os.path.dirname(__file__), "assets", "home_poster.png"))
if 'spoken_raw' not in st.session_state:
    st.session_state.spoken_raw = {}
if 'spoken_value' not in st.session_state:
    st.session_state.spoken_value = {}
if 'spoken_candidate' not in st.session_state:
    st.session_state.spoken_candidate = {}
if 'login_logged' not in st.session_state:
    st.session_state.login_logged = False
if 'confirm_spoken' not in st.session_state:
    st.session_state.confirm_spoken = {}
if 'recommendation_logged' not in st.session_state:
    st.session_state.recommendation_logged = False
if 'budget_logged' not in st.session_state:
    st.session_state.budget_logged = False

questions = [
    "Enter your annual tuition fee:",
    "Enter your current bank balance:",
    "Enter your monthly part-time income:",
    "Enter your monthly internship income:",
    "Enter total received scholarships:",
    "Enter total available loans:",
    "Enter the number of months for which you need housing:"
]

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

def _auth0_config():
    return {
        "domain": os.getenv("AUTH0_DOMAIN"),
        "client_id": os.getenv("AUTH0_CLIENT_ID"),
        "client_secret": os.getenv("AUTH0_CLIENT_SECRET"),
        "redirect_uri": os.getenv("AUTH0_REDIRECT_URI"),
        "audience": os.getenv("AUTH0_AUDIENCE")
    }

def _auth0_ready(cfg):
    return all([cfg["domain"], cfg["client_id"], cfg["client_secret"], cfg["redirect_uri"]])

def _auth0_oauth(cfg):
    return OAuth2Session(
        cfg["client_id"],
        cfg["client_secret"],
        scope="openid profile email",
        redirect_uri=cfg["redirect_uri"]
    )

def _auth0_login_url(cfg):
    if st.session_state.get("auth0_login_url") and st.session_state.get("auth0_state"):
        return st.session_state.auth0_login_url
    oauth = _auth0_oauth(cfg)
    authorize_url = f"https://{cfg['domain']}/authorize"
    url, state = oauth.create_authorization_url(
        authorize_url,
        audience=cfg["audience"]
    )
    st.session_state.auth0_state = state
    st.session_state.auth0_login_url = url
    return url

def _auth0_handle_callback(cfg):
    params = st.query_params
    if "code" not in params or "state" not in params:
        return
    expected_state = st.session_state.get("auth0_state")
    if expected_state and params.get("state") != expected_state:
        st.error("Auth0 login state mismatch. Please try again.")
        st.query_params.clear()
        return
    oauth = _auth0_oauth(cfg)
    token_url = f"https://{cfg['domain']}/oauth/token"
    token = oauth.fetch_token(
        token_url,
        code=params.get("code"),
        client_secret=cfg["client_secret"]
    )
    userinfo_url = f"https://{cfg['domain']}/userinfo"
    access_token = token.get("access_token")
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    userinfo = oauth.get(userinfo_url, headers=headers).json()
    st.session_state.auth0_token = token
    st.session_state.auth0_user = userinfo
    st.query_params.clear()
    st.rerun()

def _auth0_logout_url(cfg):
    return f"https://{cfg['domain']}/v2/logout?client_id={cfg['client_id']}&returnTo={cfg['redirect_uri']}"

def _snowflake_config():
    return {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "table": os.getenv("SNOWFLAKE_IAM_TABLE", "AUTH0_LOGIN_EVENTS"),
        "rec_table": os.getenv("SNOWFLAKE_RECOMMENDATIONS_TABLE", "RECOMMENDATION_EVENTS"),
        "budget_table": os.getenv("SNOWFLAKE_BUDGET_TABLE", "BUDGET_HISTORY")
    }

def _snowflake_ready(cfg):
    required = [cfg["account"], cfg["user"], cfg["password"], cfg["warehouse"], cfg["database"], cfg["schema"]]
    return all(required)

@st.cache_resource
def _snowflake_connect(cfg):
    return snowflake_connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        password=cfg["password"],
        role=cfg["role"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"]
    )

def _snowflake_log_login(userinfo):
    if snowflake_connector is None:
        return "Snowflake connector not installed. Run: pip install snowflake-connector-python"
    cfg = _snowflake_config()
    if not _snowflake_ready(cfg):
        return "Snowflake not configured. Set SNOWFLAKE_* env vars."
    try:
        conn = _snowflake_connect(cfg)
        cur = conn.cursor()
        full_table = f"{cfg['database']}.{cfg['schema']}.{cfg['table']}"
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {full_table} (
                user_id STRING,
                email STRING,
                name STRING,
                provider STRING,
                login_ts TIMESTAMP
            )
            """
        )
        user_id = userinfo.get("sub")
        email = userinfo.get("email")
        name = userinfo.get("name")
        provider = userinfo.get("sub", "").split("|")[0] if userinfo.get("sub") else None
        login_ts = dt.datetime.utcnow()
        cur.execute(
            f"INSERT INTO {full_table} (user_id, email, name, provider, login_ts) VALUES (%s, %s, %s, %s, %s)",
            (user_id, email, name, provider, login_ts)
        )
        cur.close()
        return None
    except Exception as e:
        return f"Snowflake log failed: {e}"

def _snowflake_log_recommendations(userinfo, safe_rent, recommendations):
    if snowflake_connector is None:
        return "Snowflake connector not installed. Run: pip install snowflake-connector-python"
    cfg = _snowflake_config()
    if not _snowflake_ready(cfg):
        return "Snowflake not configured. Set SNOWFLAKE_* env vars."
    try:
        conn = _snowflake_connect(cfg)
        cur = conn.cursor()
        full_table = f"{cfg['database']}.{cfg['schema']}.{cfg['rec_table']}"
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {full_table} (
                user_id STRING,
                email STRING,
                safe_rent NUMBER(10,2),
                address STRING,
                monthly_rent NUMBER(10,2),
                company STRING,
                event_ts TIMESTAMP
            )
            """
        )
        user_id = userinfo.get("sub")
        email = userinfo.get("email")
        event_ts = dt.datetime.utcnow()
        if recommendations.empty:
            cur.execute(
                f"INSERT INTO {full_table} (user_id, email, safe_rent, address, monthly_rent, company, event_ts) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, email, safe_rent, None, None, None, event_ts)
            )
        else:
            for _, row in recommendations.iterrows():
                cur.execute(
                    f"INSERT INTO {full_table} (user_id, email, safe_rent, address, monthly_rent, company, event_ts) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        user_id,
                        email,
                        safe_rent,
                        row.get("address"),
                        row.get("monthly_rent"),
                        row.get("company"),
                        event_ts
                    )
                )
        cur.close()
        return None
    except Exception as e:
        return f"Snowflake recommendation log failed: {e}"

def _snowflake_log_budget(userinfo, user_input, safe_rent):
    if snowflake_connector is None:
        return "Snowflake connector not installed. Run: pip install snowflake-connector-python"
    cfg = _snowflake_config()
    if not _snowflake_ready(cfg):
        return "Snowflake not configured. Set SNOWFLAKE_* env vars."
    try:
        conn = _snowflake_connect(cfg)
        cur = conn.cursor()
        full_table = f"{cfg['database']}.{cfg['schema']}.{cfg['budget_table']}"
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {full_table} (
                user_id STRING,
                email STRING,
                tuition NUMBER(12,2),
                bank_balance NUMBER(12,2),
                part_time_income NUMBER(12,2),
                internship_income NUMBER(12,2),
                scholarships NUMBER(12,2),
                loans NUMBER(12,2),
                months NUMBER(10,0),
                safe_rent NUMBER(10,2),
                event_ts TIMESTAMP
            )
            """
        )
        user_id = userinfo.get("sub")
        email = userinfo.get("email")
        event_ts = dt.datetime.utcnow()
        cur.execute(
            f"""
            INSERT INTO {full_table}
            (user_id, email, tuition, bank_balance, part_time_income, internship_income, scholarships, loans, months, safe_rent, event_ts)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                email,
                user_input.get("tuition"),
                user_input.get("bank_balance"),
                user_input.get("part_time_income"),
                user_input.get("internship_income"),
                user_input.get("scholarships"),
                user_input.get("loans"),
                user_input.get("months"),
                safe_rent,
                event_ts
            )
        )
        cur.close()
        return None
    except Exception as e:
        return f"Snowflake budget log failed: {e}"

def _snowflake_fetch_budget(userinfo, limit=20):
    if snowflake_connector is None:
        return None, "Snowflake connector not installed. Run: pip install snowflake-connector-python"
    cfg = _snowflake_config()
    if not _snowflake_ready(cfg):
        return None, "Snowflake not configured. Set SNOWFLAKE_* env vars."
    try:
        conn = _snowflake_connect(cfg)
        cur = conn.cursor()
        full_table = f"{cfg['database']}.{cfg['schema']}.{cfg['budget_table']}"
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {full_table} (
                user_id STRING,
                email STRING,
                tuition NUMBER(12,2),
                bank_balance NUMBER(12,2),
                part_time_income NUMBER(12,2),
                internship_income NUMBER(12,2),
                scholarships NUMBER(12,2),
                loans NUMBER(12,2),
                months NUMBER(10,0),
                safe_rent NUMBER(10,2),
                event_ts TIMESTAMP
            )
            """
        )
        user_id = userinfo.get("sub")
        cur.execute(
            f"""
            SELECT user_id, email, tuition, bank_balance, part_time_income, internship_income, scholarships, loans, months, safe_rent, event_ts
            FROM {full_table}
            WHERE user_id = %s
            ORDER BY event_ts DESC
            LIMIT %s
            """,
            (user_id, limit)
        )
        rows = cur.fetchall()
        cur.close()
        data = [
            {
                "user_id": r[0],
                "email": r[1],
                "tuition": r[2],
                "bank_balance": r[3],
                "part_time_income": r[4],
                "internship_income": r[5],
                "scholarships": r[6],
                "loans": r[7],
                "months": r[8],
                "safe_rent": r[9],
                "event_ts": r[10],
            }
            for r in rows
        ]
        return data, None
    except Exception as e:
        return None, f"Snowflake fetch failed: {e}"

def _snowflake_fetch_recommendations(userinfo, limit=20):
    if snowflake_connector is None:
        return None, "Snowflake connector not installed. Run: pip install snowflake-connector-python"
    cfg = _snowflake_config()
    if not _snowflake_ready(cfg):
        return None, "Snowflake not configured. Set SNOWFLAKE_* env vars."
    try:
        conn = _snowflake_connect(cfg)
        cur = conn.cursor()
        full_table = f"{cfg['database']}.{cfg['schema']}.{cfg['rec_table']}"
        user_id = userinfo.get("sub")
        cur.execute(
            f"""
            SELECT user_id, email, safe_rent, address, monthly_rent, company, event_ts
            FROM {full_table}
            WHERE user_id = %s
            ORDER BY event_ts DESC
            LIMIT %s
            """,
            (user_id, limit)
        )
        rows = cur.fetchall()
        cur.close()
        data = [
            {
                "user_id": r[0],
                "email": r[1],
                "safe_rent": r[2],
                "address": r[3],
                "monthly_rent": r[4],
                "company": r[5],
                "event_ts": r[6],
            }
            for r in rows
        ]
        return data, None
    except Exception as e:
        return None, f"Snowflake fetch failed: {e}"

def _auth_gate():
    if OAuth2Session is None:
        st.error("Auth0 requires the 'authlib' package. Run: pip install authlib")
        st.stop()
    cfg = _auth0_config()
    if not _auth0_ready(cfg):
        st.info(
            "Auth0 is not configured yet. Add these to your .env:\n"
            "AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_REDIRECT_URI, "
            "(optional) AUTH0_AUDIENCE"
        )
        st.stop()
    _auth0_handle_callback(cfg)
    if "auth0_user" not in st.session_state:
        st.subheader("Sign in to continue")
        login_url = _auth0_login_url(cfg)
        st.link_button("Log in with Auth0", login_url, use_container_width=True)
        with st.expander("Auth0 login URL (debug)"):
            st.code(login_url)
        st.stop()
    user = st.session_state.auth0_user
    name = user.get("name") or user.get("email") or "Authenticated user"
    st.caption(f"Signed in as {name}")
    if not st.session_state.login_logged:
        err = _snowflake_log_login(user)
        if err:
            st.warning(err)
        else:
            st.session_state.login_logged = True
    if st.button("Log out"):
        st.session_state.pop("auth0_user", None)
        st.session_state.pop("auth0_token", None)
        st.session_state.pop("auth0_state", None)
        st.session_state.pop("auth0_login_url", None)
        st.session_state.login_logged = False
        st.session_state.recommendation_logged = False
        st.session_state.budget_logged = False
        st.markdown(f"[Complete logout]({_auth0_logout_url(cfg)})")
        st.stop()

st.title("🏠 Off-Campus Community Virtual Budget Assistant")
_auth_gate()
_frame_start("Settings")
st.checkbox("Fast demo mode (reduce latency)", key="fast_demo")
st.checkbox("1-minute demo mode (auto-fill answers)", key="demo_mode")
st.checkbox("Read instructions aloud", key="read_aloud")
_frame_end()

_frame_start("Audio Status & Test")
with st.expander("Audio status & test", expanded=False):
    st.write(f"Read aloud enabled: {st.session_state.read_aloud}")
    st.write(f"Current tone: {st.session_state.tone}")
    st.write(f"Voice ID in use: {st.session_state.voice_map.get(st.session_state.tone)}")
    st.write(f"ElevenLabs API key set: {bool(os.getenv('ELEVENLABS_API_KEY'))}")
    if st.button("Test audio now"):
        _speak("Audio test. If you can hear this, text to speech is working.", tone=st.session_state.tone, async_playback=False)
_frame_end()

_frame_start("History (Snowflake)")
with st.expander("Recommendation history (from Snowflake)", expanded=False):
    if "auth0_user" in st.session_state:
        history, err = _snowflake_fetch_recommendations(st.session_state.auth0_user, limit=20)
        if err:
            st.warning(err)
        elif not history:
            st.info("No recommendation history yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)

with st.expander("Budget input history (from Snowflake)", expanded=False):
    if "auth0_user" in st.session_state:
        history, err = _snowflake_fetch_budget(st.session_state.auth0_user, limit=20)
        if err:
            st.warning(err)
        elif not history:
            st.info("No budget history yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
_frame_end()

# -------------------------
# Step 1: Voice tone selection (3 buttons → go to 1st question)
# -------------------------
if not st.session_state.tone_confirmed:
    _frame_start("Tone Selection")
    st.subheader("Choose your preferred voice tone")
    st.caption("Click a button to set the assistant's tone and start the questions.")
    if st.session_state.read_aloud and not st.session_state.tone_prompt_spoken:
        _speak(
            "Choose your preferred voice tone. Click a button to start.",
            tone=st.session_state.tone,
            async_playback=True
        )
        st.session_state.tone_prompt_spoken = True
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 Friendly", use_container_width=True):
            st.session_state.tone = "friendly"
            st.session_state.tone_confirmed = True
            st.session_state.q_index = 1
            st.session_state.tone_prompt_spoken = False
            _speak(
                "You selected friendly tone. Let's start with the first question.",
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )
    with col2:
        if st.button("💼 Professional", use_container_width=True):
            st.session_state.tone = "professional"
            st.session_state.tone_confirmed = True
            st.session_state.q_index = 1
            st.session_state.tone_prompt_spoken = False
            _speak(
                "You selected professional tone. Let's start with the first question.",
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )
    with col3:
        if st.button("😐 Neutral", use_container_width=True):
            st.session_state.tone = "neutral"
            st.session_state.tone_confirmed = True
            st.session_state.q_index = 1
            st.session_state.tone_prompt_spoken = False
            _speak(
                "You selected neutral tone. Let's start with the first question.",
                tone=st.session_state.tone,
                async_playback=st.session_state.fast_demo
            )
    _frame_end()

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
    _frame_start("Question")
    q_idx = st.session_state.q_index - 1
    question = questions[q_idx]

    st.subheader(f"Question {st.session_state.q_index} of {len(questions)}")
    st.write(question)
    if st.button("Replay question audio", key=f"replay_q_{q_idx}"):
        _speak(
            f"Question {st.session_state.q_index} of {len(questions)}.",
            tone=st.session_state.tone,
            async_playback=False
        )
        _speak(
            question,
            tone=st.session_state.tone,
            async_playback=False
        )

    # Speak the question once when entering this question
    if st.session_state.read_aloud:
        if 'spoken_q' not in st.session_state or st.session_state.spoken_q != q_idx:
            _speak(
                f"Question {st.session_state.q_index} of {len(questions)}.",
                tone=st.session_state.tone,
                async_playback=False
            )
            _speak(
                question,
                tone=st.session_state.tone,
                async_playback=False
            )
            st.session_state.spoken_q = q_idx

    # Input method: Type or Speak
    answer_method = st.radio("Answer by:", options=["Type", "Speak"], key=f"method_{q_idx}", horizontal=True)

    if answer_method == "Type":
        if st.session_state.read_aloud and not st.session_state.method_prompt_spoken.get(q_idx):
            _speak(
                "You can type your answer and press submit.",
                tone=st.session_state.tone,
                async_playback=True
            )
            st.session_state.method_prompt_spoken[q_idx] = True
        typed_answer = st.text_input("Type your answer (e.g. 100 or one hundred) and press Enter:", key=f"type_{q_idx}", placeholder="Enter a number...")
        if st.button("Submit Answer", key=f"submit_{q_idx}"):
            if typed_answer is None or (isinstance(typed_answer, str) and not typed_answer.strip()):
                st.warning("Please type your answer before submitting.")
            else:
                st.session_state.answers[q_idx] = clean_float(typed_answer)
                if not st.session_state.fast_demo:
                    _speak(f"You entered {typed_answer}.", tone=st.session_state.tone)
                if st.session_state.q_index < len(questions):
                    st.session_state.q_index += 1
                else:
                    st.session_state.finished = True
                st.rerun()
    else:
        if st.session_state.read_aloud and not st.session_state.method_prompt_spoken.get(q_idx):
            _speak(
                "You can speak your answer. Click start speaking.",
                tone=st.session_state.tone,
                async_playback=True
            )
            st.session_state.method_prompt_spoken[q_idx] = True
        # Speak: record voice, convert words to number (e.g. "one hundred" → 100)
        if st.session_state.get("speak_error"):
            st.warning(st.session_state.speak_error)
            st.session_state.speak_error = None
        if st.button("🎤 Start Speaking", key=f"speak_btn_{q_idx}"):
            with st.spinner("Listening... say your answer now."):
                if not st.session_state.fast_demo:
                    _speak("Listening. Say your answer now.", tone=st.session_state.tone)
                raw = listen_and_transcribe(duration=3 if st.session_state.fast_demo else 5)
                if raw:
                    suggestion = openrouter_clarify_number(raw)
                    numeric = None
                    normalized = None
                    if suggestion and suggestion.get("number") is not None:
                        try:
                            numeric = float(suggestion.get("number"))
                            normalized = suggestion.get("normalized") or raw
                        except Exception:
                            numeric = None
                    if numeric is None:
                        numeric = clean_float(raw)
                        normalized = raw
                    st.session_state.spoken_raw[q_idx] = raw
                    st.session_state.spoken_value[q_idx] = numeric
                    st.session_state.spoken_candidate[q_idx] = normalized
                else:
                    st.session_state.speak_error = "We didn't catch that. Please try again."

        if q_idx in st.session_state.get('spoken_raw', {}):
            raw = st.session_state.spoken_raw[q_idx]
            num = st.session_state.spoken_value[q_idx]
            normalized = st.session_state.spoken_candidate.get(q_idx, raw)
            st.success(f"We heard: **{raw}** → **{normalized}** (**{num}**)")
            if st.session_state.read_aloud and not st.session_state.confirm_spoken.get(q_idx):
                _speak(
                    f"We heard {normalized}. Is that correct? Say yes to submit or no to try again.",
                    tone=st.session_state.tone,
                    async_playback=True
                )
                st.session_state.confirm_spoken[q_idx] = True
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, submit", key=f"confirm_yes_{q_idx}"):
                    st.session_state.answers[q_idx] = num
                    if not st.session_state.fast_demo:
                        _speak(f"You said {normalized}.", tone=st.session_state.tone)
                    if st.session_state.q_index < len(questions):
                        st.session_state.q_index += 1
                    else:
                        st.session_state.finished = True
                    st.rerun()
            with col_no:
                if st.button("No, try again", key=f"confirm_no_{q_idx}"):
                    st.session_state.spoken_raw.pop(q_idx, None)
                    st.session_state.spoken_value.pop(q_idx, None)
                    st.session_state.spoken_candidate.pop(q_idx, None)
                    st.session_state.confirm_spoken.pop(q_idx, None)
                    st.session_state.speak_error = "Okay, please try again."
                    st.rerun()
    _frame_end()

# -------------------------
# Step 3: Visualization + budget + recommendations (read aloud)
# -------------------------
if st.session_state.finished:
    _frame_start("Results")
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

    if "auth0_user" in st.session_state and not st.session_state.recommendation_logged:
        rec_err = _snowflake_log_recommendations(st.session_state.auth0_user, safe_rent, recommendations)
        if rec_err:
            st.warning(rec_err)
        else:
            st.session_state.recommendation_logged = True

    if "auth0_user" in st.session_state and not st.session_state.budget_logged:
        budget_err = _snowflake_log_budget(st.session_state.auth0_user, user_input, safe_rent)
        if budget_err:
            st.warning(budget_err)
        else:
            st.session_state.budget_logged = True

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
        _speak(summary_text, tone=st.session_state.tone, async_playback=st.session_state.fast_demo)
        st.session_state.results_spoken = True
    _frame_end()
