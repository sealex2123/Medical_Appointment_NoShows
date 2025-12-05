import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# LOAD & TRAIN MODEL
# -------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("KaggleV2-May-2016.csv")

    # Convert binary columns to Yes/No for UI, but model uses numeric
    df["Scholarship"] = df["Scholarship"].astype(int)
    df["Hipertension"] = df["Hipertension"].astype(int)
    df["Diabetes"] = df["Diabetes"].astype(int)
    df["Alcoholism"] = df["Alcoholism"].astype(int)
    df["SMS_received"] = df["SMS_received"].astype(int)

    df["Handcap"] = df["Handcap"].clip(upper=1)  # Make 0/1

    df["No-show"] = df["No-show"].map({"No": 0, "Yes": 1})

    # Extract weekday
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["Weekday"] = df["AppointmentDay"].dt.day_name()

    feature_cols = [
        "Gender", "Age", "Scholarship", "Hipertension", "Diabetes",
        "Alcoholism", "Handcap", "SMS_received", "Weekday",
        "DaysBetween"
    ]

    # feature: days between scheduling & appointment
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["DaysBetween"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

    # Encode categorical
    df_encoded = pd.get_dummies(df[feature_cols], drop_first=True)

    X = df_encoded
    y = df["No-show"]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    return model, list(df_encoded.columns)


# Load trained model
model, feature_cols = train_model()

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Medical No-Show Predictor",
    layout="wide"
)

# -------------------------
# CUSTOM CSS (UPDATED COLORS)
# -------------------------
st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */
    body, .stApp {
        background-color: #ffffff !important;
    }

    /* HEADER */
    .big-header {
        font-size: 42px;
        font-weight: 800;
        color: #222222;
        text-align: center;
        margin-bottom: 10px;
    }

    .sub-header {
        font-size: 20px;
        color: #444444;
        text-align: center;
        margin-bottom: 40px;
    }

    /* ===== SIDEBAR LIGHT GRAY ===== */
    section[data-testid="stSidebar"] {
        background-color: #f2f2f5 !important;
        border-right: 1px solid #e0e0e5 !important;
        color: #1e1e2f !important;
    }

    /* ===== INPUT FIELDS LIGHT GRAY ===== */
    div[data-baseweb="select"] > div,
    .stSelectbox div[role="button"],
    .stNumberInput input,
    input[type="text"],
    input[type="number"],
    textarea {
        background-color: #f2f2f5 !important;
        color: #1e1e2f !important;
        border-radius: 10px !important;
        border: 1px solid #d3d3da !important;
    }

    /* Dropdown text */
    .css-1uccc91-singleValue { 
        color: #1e1e2f !important;
    }

    /* Dropdown placeholder */
    .css-1wa3eu0-placeholder { 
        color: #6c6f78 !important;
    }

    /* Icon color */
    div[data-baseweb="select"] svg {
        color: #1e1e2f !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# MAIN HEADER
# -------------------------
st.markdown("<div class='big-header'>🩺 Medical No-Show Risk Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Estimate the chance a patient will miss their appointment so clinics can take action.</div>", unsafe_allow_html=True)

# -------------------------
# USER INPUTS
# -------------------------
st.subheader("📄 Patient & Appointment Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=0, max_value=115, value=30)
    scholarship = st.selectbox("On Scholarship?", ["Yes", "No"])
    hypertension = st.selectbox("Hypertension?", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes?", ["Yes", "No"])

with col2:
    alcoholism = st.selectbox("Alcoholism?", ["Yes", "No"])
    handcap = st.selectbox("Disability (Handicap)?", ["Yes", "No"])
    sms = st.selectbox("SMS Reminder Sent?", ["Yes", "No"])
    weekday = st.selectbox(
        "Appointment Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    )
    days_between = st.number_input(
        "Days Between Scheduling and Appointment",
        min_value=0, max_value=90, value=5
    )

# -------------------------
# FORMAT INPUT FOR MODEL
# -------------------------
input_dict = {
    "Gender": 1 if gender == "M" else 0,
    "Age": age,
    "Scholarship": 1 if scholarship == "Yes" else 0,
    "Hipertension": 1 if hypertension == "Yes" else 0,
    "Diabetes": 1 if diabetes == "Yes" else 0,
    "Alcoholism": 1 if alcoholism == "Yes" else 0,
    "Handcap": 1 if handcap == "Yes" else 0,
    "SMS_received": 1 if sms == "Yes" else 0,
    "DaysBetween": days_between,
}

# One-hot weekday
for wd in ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
    input_dict[f"Weekday_{wd}"] = 1 if weekday == wd else 0

# Ensure correct column order
row = []
for col in feature_cols:
    row.append(input_dict.get(col, 0))

row = np.array(row).reshape(1, -1)

# -------------------------
# PREDICTION BUTTON
# -------------------------
st.markdown("### Prediction")

if st.button("Predict No-Show Risk"):
    prob = model.predict_proba(row)[0][1]
    percent = int(prob * 100)

    if percent >= 50:
        st.error(f"❌ Patient is **LIKELY to No-Show** ({percent}%)")
    else:
        st.success(f"✅ Patient is **LIKELY to Attend** ({percent}%)")

    st.metric("Predicted No-Show Probability", f"{percent}%")

    st.caption("This is an educational prototype — not for clinical decision-making.")
