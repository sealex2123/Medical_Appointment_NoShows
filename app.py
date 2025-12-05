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

# Adjust this if you rename the file in GitHub
CSV_FILE = "KaggleV2-May-2016.csv"


# ----------------------------------------------------
# 1. TRAIN MODEL ON STARTUP (CACHED)
# ----------------------------------------------------
@st.cache_resource
def train_model():
    # Load data from repo
    df = pd.read_csv(CSV_FILE)

    # ---------- BASIC CLEANING ----------
    # Age filter
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

    # binary_features are already 0/1, so we passthrough
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

    return model, numeric_features, binary_features, categorical_features


pipeline, numeric_features, binary_features, categorical_features = train_model()


# ----------------------------------------------------
# 2. STREAMLIT UI
# ----------------------------------------------------
st.title("Medical Appointment No-Show Predictor")

st.write(
    """
This app predicts the risk that a patient will **not** show up to their medical appointment.
The goal is to help clinics identify high-risk patients so they can send reminders or offer rescheduling.
"""
)

st.subheader("Patient & Appointment Information")

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


# helper to convert Yes/No to 0/1
def yn_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


# derive IsWeekend from weekday
is_weekend = 1 if appt_weekday in ["Saturday", "Sunday"] else 0

if st.button("Predict No-Show Risk"):
    # Build a single-row DataFrame that matches training features
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

    proba_no_show = pipeline.predict_proba(X_input)[0, 1]
    pred = pipeline.predict(X_input)[0]

    st.subheader("Prediction")

    if pred == 1:
        st.error("Prediction: Patient is **LIKELY TO NO-SHOW** ❌")
    else:
        st.success("Prediction: Patient is **LIKELY TO SHOW UP** ✅")

   # Convert probability to percentage
    percent = proba_no_show * 100

    st.metric(
        label="Predicted No-Show Probability",
        value=f"{percent:.1f}%",   # e.g., 51.3%
    )

    st.caption(
        "This is an educational prototype and should not be used as a sole basis for real clinical decisions."
    )
