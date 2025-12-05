import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------
# 1. TRAIN MODEL ON STARTUP (CACHED)
# ----------------------------------------------------


@st.cache_resource
def train_model():
    # Load raw data from repo
    df = pd.read_csv("KaggleV2-May-2016.csv")

    # ----- Basic cleaning -----
    # Keep reasonable ages
    df = df[(df["Age"] >= 0) & (df["Age"] <= 100)].copy()

    # Parse dates
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce")
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")

    df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])

    # Make everything date-only and compute waiting days
    sched_date = df["ScheduledDay"].dt.normalize()
    appt_date = df["AppointmentDay"].dt.normalize()
    df["DaysBetween"] = (appt_date - sched_date).dt.days

    # Only keep non-negative gaps
    df = df[df["DaysBetween"] >= 0].copy()

    # Weekday and weekend indicator
    df["ApptWeekday"] = df["AppointmentDay"].dt.day_name()
    df["IsWeekend"] = (df["AppointmentDay"].dt.weekday >= 5).astype(int)

    # Target: 1 = No-show, 0 = Show
    df["NoShow"] = df["No-show"].map({"No": 0, "Yes": 1})

    # Drop rows with missing target just in case
    df = df.dropna(subset=["NoShow"])

    # Features we will use in the app form
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

    X = df[feature_cols].copy()
    y = df["NoShow"].astype(int)

    # ----- Preprocessing + model pipeline -----
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # binary_features are already 0/1, just pass them through
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
        class_weight="balanced",  # handle class imbalance a bit
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    model.fit(X, y)

    return model, feature_cols


# Train once and cache
pipeline, feature_cols = train_model()

# ----------------------------------------------------
# 2. STREAMLIT UI
# ----------------------------------------------------

st.set_page_config(page_title="Medical No-Show Predictor", layout="centered")

st.title("Medical Appointment No-Show Predictor")

st.write(
    """
This app predicts the risk that a patient will **not** show up to their medical appointment.
Clinics could use this to send reminders or proactively offer rescheduling to high-risk patients.
"""
)

st.subheader("Patient & Appointment Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["F", "M"])
    scholarship = st.selectbox("On Scholarship (welfare program)?", [0, 1])
    hipertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])

with col2:
    alcoholism = st.selectbox("Alcoholism", [0, 1])
    handcap = st.selectbox("Handcap (0–4)", [0, 1, 2, 3, 4])
    sms_received = st.selectbox("SMS Reminder Received", [0, 1])
    is_weekend = st.selectbox("Appointment on Weekend?", [0, 1])
    days_between = st.number_input(
        "Days Between Scheduling and Appointment",
        min_value=0,
        value=5,
    )

neighbourhood = st.text_input("Neighbourhood", value="JARDIM DA PENHA")
appt_weekday = st.selectbox(
    "Appointment Weekday",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)

if st.button("Predict No-Show Risk"):
    # Build a single-row DataFrame that matches the training columns
    input_dict = {
        "Age": age,
        "DaysBetween": days_between,
        "Gender": gender,
        "Neighbourhood": neighbourhood,
        "ApptWeekday": appt_weekday,
        "Scholarship": scholarship,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "IsWeekend": is_weekend,
    }

    row = {c: input_dict[c] for c in feature_cols}
    X_input = pd.DataFrame([row])

    # Predict
    proba_no_show = pipeline.predict_proba(X_input)[0][1]
    pred = pipeline.predict(X_input)[0]

    st.subheader("Prediction")

    if pred == 1:
        st.error("Prediction: Patient is **LIKELY TO NO-SHOW** ❌")
    else:
        st.success("Prediction: Patient is **LIKELY TO SHOW UP** ✅")

    st.metric("Predicted No-Show Probability", f"{proba_no_show:.3f}")

    st.caption(
        "Educational prototype only – not for real clinical decision-making."
    )
