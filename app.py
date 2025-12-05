import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Medical No-Show Predictor", layout="centered")

# Name of the CSV file in your GitHub repo root
CSV_FILE = "KaggleV2-May-2016.csv"

# ----------------------------
# CUSTOM AESTHETIC THEME
# ----------------------------
st.markdown(
    """
    <style>

    /* ============================================================
       GLOBAL BACKGROUND + TEXT STYLES
       ============================================================ */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f2f7ff 100%) !important;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        color: #1e1e2f !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    p, label, span, div {
        color: #2b2b2b !important;
        font-size: 16px !important;
    }

    /* ============================================================
       SIDEBAR (LIGHT MODE)
       ============================================================ */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e3e6ea !important;
        color: #1e1e2f !important;
    }

    /* ============================================================
       INPUTS + DROPDOWNS (LIGHT MODE)
       ============================================================ */
    div[data-baseweb="select"] > div,
    .stSelectbox div[role="button"],
    .stNumberInput input,
    input[type="text"],
    input[type="number"] {
        background-color: #f3f4f6 !important;
        color: #1e1e2f !important;
        border-radius: 10px !important;
        border: 1px solid #d5d8df !important;
    }

    /* Dropdown selected text */
    .css-1uccc91-singleValue {
        color: #1e1e2f !important;
    }

    /* Dropdown placeholder text */
    .css-1wa3eu0-placeholder {
        color: #6b7280 !important;
    }

    /* Dropdown arrow icon */
    div[data-baseweb="select"] svg {
        color: #1e1e2f !important;
    }

    /* Style prediction result alert */
    .stAlert {
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: white !important;
        padding: 20px;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        color: #1e1e2f !important;
    }

    /* Buttons (purple gradient) */
    .stButton>button {
        background: linear-gradient(135deg, #6a8dff, #a875ff);
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(122, 87, 255, 0.4);
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(122, 87, 255, 0.5);
    }

    /* Footer caption */
    .stCaption {
        color: #555 !important;
        font-size: 14px !important;
        opacity: 0.9;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# SIDEBAR CONTENT
# ----------------------------
with st.sidebar:
    st.title("🩺 About this app")
    st.write(
        "This demo predicts the risk that a patient will **no-show** a medical "
        "appointment using a logistic regression model trained on the "
        "Kaggle Medical Appointment No-Show dataset."
    )

    st.markdown("### 📊 Model details")
    st.write("- Algorithm: Logistic Regression")
    st.write("- Focus: Higher recall on no-shows")
    st.write("- Inputs: Age, days between scheduling and appointment, comorbidities, SMS, etc.")

    st.markdown("### 💡 How to use")
    st.write(
        "1. Enter patient and appointment details.\n"
        "2. Click **Predict No-Show Risk**.\n"
        "3. Use the risk level to decide on reminders or rescheduling."
    )

    st.caption("Prototype for educational use only – not real clinical advice.")


# ----------------------------------------------------
# 1. TRAIN MODEL ON STARTUP (CACHED)
# ----------------------------------------------------
@st.cache_resource
def train_model():
    # Load data from repo
    df = pd.read_csv(CSV_FILE)

    # ---------- BASIC CLEANING ----------
    # Age filter
    if "Age" not in df.columns:
        raise ValueError("The 'Age' column is missing from the CSV.")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= 0) & (df["Age"] <= 100)].copy()

    # Parse dates
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce")
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")

    df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])

    # Normalize to date only, compute days between scheduling and appointment
    sched_date = df["ScheduledDay"].dt.normalize()
    appt_date = df["AppointmentDay"].dt.normalize()
    df["DaysBetween"] = (appt_date - sched_date).dt.days

    # Only keep non-negative delays
    df = df[df["DaysBetween"] >= 0].copy()

    # Weekday and weekend flag
    df["ApptWeekday"] = df["AppointmentDay"].dt.day_name()
    df["IsWeekend"] = (df["AppointmentDay"].dt.weekday >= 5).astype(int)

    # Target: 1 = No-show, 0 = Show
    if "No-show" not in df.columns:
        raise ValueError("The 'No-show' column is missing from the CSV.")

    df["NoShow"] = df["No-show"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["NoShow"])
    df["NoShow"] = df["NoShow"].astype(int)

    # Handicap 0–4 -> binary: 0 = no disability, 1+ = has disability
    df["Handcap"] = (df["Handcap"] > 0).astype(int)

    # ---------- FEATURE SETUP ----------
    numeric_features = ["Age", "DaysBetween"]
    binary_features = [
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "IsWeekend",
    ]
    categorical_features = ["Gender", "Neighbourhood", "ApptWeekday"]

    feature_cols = numeric_features + binary_features + categorical_features

    # Ensure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column in data: {col}")

    X = df[feature_cols].copy()
    y = df["NoShow"]

    # ---------- PREPROCESSING + MODEL ----------
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    model.fit(X, y)

    return model


pipeline = train_model()


# ----------------------------------------------------
# 2. STREAMLIT UI
# ----------------------------------------------------
# Hero header
st.markdown(
    """
    <div style="
        background: white;
        padding: 24px 28px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        margin-bottom: 24px;
    ">
        <h1 style="margin-bottom: 0.4rem;">🩺 Medical No-Show Risk Dashboard</h1>
        <p style="font-size: 16px; margin-bottom: 0;">
            Estimate how likely a patient is to <b>miss</b> a scheduled appointment so staff can
            send reminders or proactively reschedule.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("👤 Patient & Appointment Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    scholarship_yn = st.selectbox("On Scholarship (welfare program)?", ["No", "Yes"])
    hipertension_yn = st.selectbox("Hypertension?", ["No", "Yes"])
    diabetes_yn = st.selectbox("Diabetes?", ["No", "Yes"])

with col2:
    alcoholism_yn = st.selectbox("Alcoholism?", ["No", "Yes"])
    handcap_yn = st.selectbox("Disability (Handicap)?", ["No", "Yes"])
    sms_yn = st.selectbox("SMS Reminder Sent?", ["No", "Yes"])
    appt_weekday = st.selectbox(
        "Appointment Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )
    days_between = st.number_input(
        "Days Between Scheduling and Appointment",
        min_value=0,
        max_value=200,
        value=10,
    )

neighbourhood = st.text_input("Neighbourhood", value="JARDIM DA PENHA")


def yn_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


# derive IsWeekend from weekday
is_weekend = 1 if appt_weekday in ["Saturday", "Sunday"] else 0


# ----------------------------------------------------
# 3. PREDICTION
# ----------------------------------------------------
if st.button("Predict No-Show Risk"):

    # Build a single-row DataFrame matching training feature structure
    input_row = {
        "Age": age,
        "DaysBetween": days_between,
        "Gender": gender,
        "Neighbourhood": neighbourhood,
        "ApptWeekday": appt_weekday,
        "Scholarship": yn_to_int(scholarship_yn),
        "Hipertension": yn_to_int(hipertension_yn),
        "Diabetes": yn_to_int(diabetes_yn),
        "Alcoholism": yn_to_int(alcoholism_yn),
        "Handcap": yn_to_int(handcap_yn),
        "SMS_received": yn_to_int(sms_yn),
        "IsWeekend": is_weekend,
    }

    X_input = pd.DataFrame([input_row])

    # Get prediction + probability
    proba_no_show = pipeline.predict_proba(X_input)[0, 1]
    pred = pipeline.predict(X_input)[0]

    st.subheader("📌 Prediction")

    # Display prediction result
    if pred == 1:
        st.error("Prediction: Patient is **LIKELY TO NOT SHOW UP** ❌")
    else:
        st.success("Prediction: Patient is **LIKELY TO SHOW UP** ✅")

    # Convert decimal to percentage
    percent = proba_no_show * 100

    # Determine risk label
    if percent < 30:
        risk_label = "Low"
    elif percent < 60:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    # Show metric card
    st.metric(
        label="Predicted No-Show Probability",
        value=f"{percent:.1f}%",
        delta=f"Risk level: {risk_label}",
    )

    # Visual progress bar for risk level
    st.write("No-show risk level")
    st.progress(int(min(max(percent, 0), 100)))

    st.caption(
        "This is an educational prototype and should not be used as a sole basis for real clinical decisions."
    )
