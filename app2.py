import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler from disk
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler (1).pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def predict_cancer(age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history):
    user_data = pd.DataFrame([[age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history]],
                             columns=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    return prediction[0]

# Streamlit app
st.title("Cancer Prediction App")

st.write("Please enter the following details:")

age = st.number_input("Age", min_value=20, max_value=80, step=1)
gender = st.radio("Gender", options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
bmi = st.slider("BMI", min_value=15.0, max_value=40.0, step=0.1)
smoking = st.radio("Smoking", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
genetic_risk = st.selectbox("Genetic Risk", options=[0, 1, 2], format_func=lambda x: ['Low', 'Medium', 'High'][x])
physical_activity = st.slider("Physical Activity (hours/week)", min_value=0.0, max_value=10.0, step=0.1)
alcohol_intake = st.slider("Alcohol Intake (units/week)", min_value=0.0, max_value=5.0, step=0.1)
cancer_history = st.radio("Cancer History", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

if st.button("Predict"):
    result = predict_cancer(age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history)
    if result == 0:
        st.success("Prediction: No Cancer")
    else:
        st.error("Prediction: Cancer")

# Run the app with: streamlit run app.py
