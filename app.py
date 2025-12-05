import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Medical No-Show Predictor", layout="centered")

@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

artifact = load_artifact("no_show_model.pkl")
pipeline = artifact["pipeline"]
feature_cols = artifact["feature_cols"]

st.title("Medical Appointment No-Show Predictor")

st.write(
    """
This app predicts the risk that a patient will **not** show up to their medical appointment.
Clinics can use this to send reminders or offer proactive rescheduling.
"""
)

st.subheader("Patient & Appointment Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["F", "M"])
    scholarship = st.selectbox("Scholarship (welfare program)", [0, 1])
    hipertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])

with col2:
    alcoholism = st.selectbox("Alcoholism", [0, 1])
    handcap = st.selectbox("Handcap (0–4)", [0, 1, 2, 3, 4])
    sms_received = st.selectbox("SMS Received", [0, 1])
    is_weekend = st.selectbox("Is Weekend?", [0, 1])
    days_between = st.number_input("Days Between", min_value=0, value=5)

neighbourhood = st.text_input("Neighbourhood", value="JARDIM DA PENHA")
appt_weekday = st.selectbox(
    "Appointment Weekday",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

if st.button("Predict No-Show Risk"):
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

    proba = pipeline.predict_proba(X_input)[0][1]
    pred = pipeline.predict(X_input)[0]

    st.subheader("Prediction")

    if pred == 1:
        st.error("Prediction: Patient likely to **NO-SHOW**")
    else:
        st.success("Prediction: Patient likely to **SHOW UP**")

    st.metric("No-Show Probability", f"{proba:.3f}")

    st.caption("Prototype model for educational use — not for real clinical decisions.")
