import pandas as pd
import streamlit as st
import pickle
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

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
st.title(":red[AI powered] Cancer detection :blue[App]")
st.subheader(":rainbow[Harnessing Machine learning for early diagnosis]")

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
        st.success("Prediction : 🍀 You are Safe ")
    else:
        st.error("Prediction  : ⚠️☠️🚨 High Chances of Cancer")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      .stApp {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
      }
      .main {
        flex: 1;
      }
    </style>
    """

    style_div = styles(
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="green",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        
        # link("", image('https://i.imgur.com/QmFEKSu.png',width=px(100), height=px(100))),
        "Made by  ",
        link("https://www.linkedin.com/in/shivam-kumar-5a6b42237/", "@SHIVAM"),
        " and ",
        link("https://www.bitmesra.ac.in/Visit_My_Page?cid=1&deptid=69&i=ZxRnKJwUT130fLMhEW4psmTtOj8vKBO70fL4Q6xnqy0%3d", "@ANIRUDDHA DEB"),
        br(),
        " Department of Chemical Engineering, BIT Mesra "
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()
