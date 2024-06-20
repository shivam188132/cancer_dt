import pandas as pd
import streamlit as st
import pickle

# Load the model and scaler from disk
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def predict_cancer(age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history):
    user_data = pd.DataFrame([[age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history]],
                             columns=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    return prediction[0]

# Streamlit app
st.title(":red[Cancer] Prediction :blue[App]")
# video_file = open('muy.mp4', 'rb')
# video_bytes = video_file.read()

# st.video(video_bytes,  muted=False)


st.write(":green[Please enter the following details:]")
st.markdown("**👴 Age**")
age = st.number_input("", min_value=20, max_value=80, step=1)
st.markdown("**👦🏻/👩🏻 :green[Gender]**")
gender = st.radio("", options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
st.markdown("**📇 :green[Body Mass Index](BMI)**")
bmi = st.slider("", min_value=15.0, max_value=40.0, step=0.05)
st.markdown("**😗🚬 :red[Smoking]**")
smoking = st.radio("", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
st.markdown("**🧬 :blue[Genetic Risk]**")
genetic_risk = st.selectbox("", options=[0, 1, 2], format_func=lambda x: ['Low', 'Medium', 'High'][x])
st.markdown("**🏋🏽🔥💪🏼🎧 :green[Physical Activity ](hours/week)**")
physical_activity = st.slider("", min_value=0.0, max_value=10.0, step=0.05)
st.markdown("**🍻 :red[Alcohol Intake ](units/week)**")
alcohol_intake = st.slider("", min_value=0.0, max_value=5.0, step=0.05)
st.markdown("**🦀 :blue[Cancer History]**")
cancer_history = st.radio(" ", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

if st.button("Predict"):
    result = predict_cancer(age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history)
    if result == 0:
        st.success("Prediction : 🍀✅ You are Safe (No Cancer)")
    else:
        st.error("Prediction  : ⚠️☠️🚨 High Chances of Cancer")

# Run the app with: streamlit run app.py
