import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Sleep Disorder Prediction",
    page_icon="😴",
    layout="wide"
)

# -------------------------------
# Load Model & Encoders
# -------------------------------
@st.cache_resource
def load_model():
    with open("sleep_disorder_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model_data = load_model()
    model = model_data["model"]
    encoders = model_data["encoders"]
except Exception as e:
    st.exception(e)
    st.stop()

# -------------------------------
# Title
# -------------------------------
st.title("😴 Sleep Disorder Prediction System")
st.markdown("Enter the patient's health and lifestyle information below:")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 1, 100, 30)
    occupation = st.selectbox(
        "Occupation",
        [
            "Accountant",
            "Doctor",
            "Engineer",
            "Lawyer",
            "Manager",
            "Nurse",
            "Sales Representative",
            "Salesperson",
            "Scientist",
            "Software Engineer",
            "Teacher",
        ],
    )
    sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 15.0, 7.0, step=0.1)

with col2:
    quality_of_sleep = st.number_input("Quality of Sleep (1-10)", 1, 10, 7)
    physical_activity = st.number_input("Physical Activity Level (1-100)", 0, 100, 50)
    stress_level = st.number_input("Stress Level (1-10)", 1, 10, 5)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"])

with col3:
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 70)
    daily_steps = st.number_input("Daily Steps", 0, 50000, 8000, step=100)
    systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
    diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 130, 80)

st.divider()

if st.button("Predict Sleep Disorder", use_container_width=True):

    # Build DataFrame with raw values
    sample = pd.DataFrame(
        {
            "Gender": [gender],
            "Age": [age],
            "Occupation": [occupation],
            "Sleep Duration": [sleep_duration],
            "Quality of Sleep": [quality_of_sleep],
            "Physical Activity Level": [physical_activity],
            "Stress Level": [stress_level],
            "BMI Category": [bmi_category],
            "Heart Rate": [heart_rate],
            "Daily Steps": [daily_steps],
            "Systolic": [systolic],
            "Diastolic": [diastolic],
        }
    )

    # Encode categorical columns using the saved LabelEncoders
    for col, encoder in encoders.items():
        sample[col] = encoder.transform(sample[col])

    # Predict
    prediction = model.predict(sample)
    result = prediction[0]

    if result == "None":
        st.success("The model predicts that the patient is **unlikely** to have any sleep disorder.")
    elif result == "Insomnia":
        st.warning("The model predicts that the patient is likely to have **Insomnia**.")
    else:
        st.error("The model predicts that the patient is likely to have **Sleep Apnea**.")
