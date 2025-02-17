import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import datetime
from PIL import Image

# -------------------------- PAGE CONFIG --------------------------
st.set_page_config(
    page_title="3D Employee Attrition Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- ASSETS --------------------------
LOADING_GIF = "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDA4dXl6anZ6enZ2eGF4aG5jYmUwN2JxbHh5dDltdGQ2MHczcWt4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26n6WywJyh39n1pBu/giphy.gif"
SUCCESS_GIF = "https://media.giphy.com/media/t3sZxY5zS5B0z5zMIz/giphy.gif?cid=790b761168e2nl5t8ey7oz035c4cd611vza2cd7tlmj87zci&ep=v1_gifs_search&rid=giphy.gif&ct=g"
ALERT_GIF = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGhtM2UzbjAxbnVlOWI0dmpyY2t4aHhha2l0c3p3eHo1ZmtlMXJ2eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1BQdjXovIqSLS/giphy.gif"
INFO_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnI1OXhndjFtOG5scWNjY2tvMXJiYTltbGV0NjZ0bWhoaTA4NTFiciZlcD12MV9naWZzX3NlYXJjaCZjdD1n/13rQ7rrTrvZXlm/giphy.gif"

# -------------------------- BACKGROUND HELPER --------------------------
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            animation: fadeIn 2s;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set initial background
set_background(INFO_GIF)

# -------------------------- MODEL LOADING --------------------------
@st.cache_resource
def load_models():
    with open("scaler_fixed.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_models()
all_features = list(model.feature_names_in_)

# -------------------------- SIDEBAR --------------------------
def build_sidebar():
    # Image path - ensure the file exists or use an online URL
    image_path = r"C:\Users\admin\OneDrive\Pictures\54c87715239a0ecae5c76df51b22b6d1.jpg"
    image_url = f"file:///{image_path.replace('\\', '/')}"
    
    st.sidebar.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            .sidebar-container {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                padding: 30px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 20px;
                font-family: 'Roboto', sans-serif;
                font-size: 18px;
            }}
            .sidebar-container h2 {{
                color: #f1c40f;
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 10px;
            }}
            .sidebar-container p {{
                color: #ffffff;
                font-size: 20px;
                line-height: 1.5;
            }}
            .sidebar-divider {{
                border: 2px solid #f1c40f;
                margin: 20px 0;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        """
        <div class='sidebar-container'>
            <h2>Welcome! 👋</h2>
            <p>Enter employee details and click <b>Predict</b> to see the attrition risk.</p>
            <div class='sidebar-divider'></div>
            <p>This interactive app uses modern UI elements and AI/ML to help you make informed decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        """
        <div style='text-align: center; color: #ffffff; margin-top: 20px; font-family: "Roboto", sans-serif; font-size: 20px;'>
            Developed by <b>Harshit Rai</b> 🚀
        </div>
        """,
        unsafe_allow_html=True
    )

build_sidebar()

# -------------------------- CUSTOM CSS --------------------------
def load_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            body { 
                color: #ffffff; 
                font-family: 'Roboto', sans-serif; 
                font-size: 18px;
            }
            .main-header { 
                text-align: center; 
                padding: 40px 0; 
            }
            .main-header h1 { 
                color: #f1c40f; 
                font-size: 56px; 
                font-weight: 700;
                margin-bottom: 20px; 
            }
            .input-section { 
                background-color: rgba(54, 47, 47, 0.9); 
                border-radius: 12px; 
                padding: 40px; 
                margin-top: 30px; 
                box-shadow: 0px 0px 20px rgba(0,0,0,0.6);
                animation: slideUp 1s;
            }
            @keyframes slideUp {
                from { transform: translateY(30px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .card {
                background: rgba(0, 0, 0, 0.5);  /* Semi-transparent black */
                border: 1px solid #ffffff;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            .footer {
                text-align: center;
                padding: 30px;
                margin-top: 50px;
                background: linear-gradient(90deg, rgba(52,73,94,0.8), rgba(44,62,80,0.8));
                color: #f1c40f;
                font-size: 20px;
                box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.4);
                font-family: 'Roboto', sans-serif;
            }
            .footer a {
                color: #f1c40f;
                text-decoration: none;
                margin: 0 12px;
                font-weight: 700;
                transition: color 0.3s;
            }
            .footer a:hover {
                color: #e67e22;
            }
            .footer .social-icons {
                margin-top: 12px;
            }
            .footer .social-icons img {
                width: 28px;
                margin: 0 10px;
                transition: transform 0.3s;
            }
            .footer .social-icons img:hover {
                transform: scale(1.2);
            }
            
            /* Button and Input Enhancements */
            .stButton>button {
                background: linear-gradient(45deg, #f1c40f, #e67e22);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                font-weight: bold;
                transition: all 0.3s ease;
                animation: pulse 2s infinite;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(241, 196, 15, 0.4);
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            .stTextInput>div>div>input, .stSelectbox>div>div>select {
                border-radius: 10px;
                border: 2px solid #f1c40f;
                padding: 10px 15px;
                transition: all 0.3s ease;
            }
            .stTextInput>div>div>input:focus {
                border-color: #e67e22;
                box-shadow: 0 0 10px rgba(241, 196, 15, 0.3);
            }
            /* Header Typography */
            h1, h2, h3 {
                background: linear-gradient(45deg, #f1c40f, #e67e22);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: fadeIn 1.5s ease-out;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

load_custom_css()

# -------------------------- HEADER --------------------------
def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🏆 3D Employee Attrition Prediction</h1>
            <p>Discover if your employee is at risk of leaving!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

render_header()

# -------------------------- COLLECT USER INPUTS (Each Input in its Own Transparent Card) --------------------------
def get_user_inputs():
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("### ✨ Enter Employee Details", unsafe_allow_html=True)
    
    # Helper function: Create a container with a card-style div and place content inside.
    def card_input(widget_func):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            result = widget_func()  # This renders the input widget inside the card.
            st.markdown("</div>", unsafe_allow_html=True)
        return result

    # Split the inputs into two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = card_input(lambda: st.number_input("Age", min_value=18, max_value=70, value=30))
        gender_option = card_input(lambda: st.selectbox("Gender", options=["Male", "Female"]))
        monthly_income = card_input(lambda: st.number_input("Monthly Income", min_value=1000, step=500, value=5000))
        overtime_option = card_input(lambda: st.selectbox("OverTime", options=["No", "Yes"]))
        job_sat = card_input(lambda: st.slider("Job Satisfaction (1: Low, 4: High)", 1, 4, 3))
        years_at_company = card_input(lambda: st.number_input("Years At Company", min_value=0, max_value=40, value=5))
    
    with col2:
        env_sat = card_input(lambda: st.slider("Environment Satisfaction (1: Low, 4: High)", 1, 4, 3))
        wlb = card_input(lambda: st.slider("Work Life Balance (1: Low, 4: High)", 1, 4, 3))
        job_role = card_input(lambda: st.selectbox("Job Role", 
            options=["Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", 
                     "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"]))
        education_field = card_input(lambda: st.selectbox("Education Field", 
            options=["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]))
        business_travel = card_input(lambda: st.selectbox("Business Travel", 
            options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"]))
        marital_status = card_input(lambda: st.selectbox("Marital Status", options=["Divorced", "Married", "Single"]))
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close input-section

    # Build input dictionary for prediction:
    input_dict = {feat: 0 for feat in all_features}
    input_dict["Age"] = age
    input_dict["MonthlyIncome"] = monthly_income
    input_dict["JobSatisfaction"] = job_sat
    input_dict["EnvironmentSatisfaction"] = env_sat
    input_dict["YearsAtCompany"] = years_at_company
    input_dict["WorkLifeBalance"] = wlb
    input_dict["Gender"] = 0 if gender_option == "Male" else 1
    input_dict["OverTime"] = 1 if overtime_option == "Yes" else 0

    # One-hot encoding for categorical features:
    job_role_feature = f"JobRole_{job_role}"
    if job_role_feature in input_dict:
        input_dict[job_role_feature] = 1

    edu_field_feature = f"EducationField_{education_field}"
    if edu_field_feature in input_dict:
        input_dict[edu_field_feature] = 1

    if business_travel == "Travel_Rarely":
        input_dict["BusinessTravel_Travel_Rarely"] = 1
    elif business_travel == "Travel_Frequently":
        input_dict["BusinessTravel_Travel_Frequently"] = 1

    if marital_status == "Married":
        input_dict["MaritalStatus_Married"] = 1
    elif marital_status == "Single":
        input_dict["MaritalStatus_Single"] = 1

    input_df = pd.DataFrame([input_dict], columns=all_features)
    try:
        input_scaled = scaler.transform(input_df)
        error_message = None
    except Exception as e:
        input_scaled = None
        error_message = str(e)
        
    return input_scaled, error_message

input_scaled, error_message = get_user_inputs()

# -------------------------- PREDICTION --------------------------
def predict_attrition(input_scaled):
    progress_bar = st.progress(0)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(percent_complete)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability

if st.button("🔍 Predict Employee Attrition"):
    if error_message:
        st.error(f"⚠️ Error in preprocessing: {error_message}")
    else:
        with st.spinner("🔎 Analyzing employee data..."):
            prediction, probability = predict_attrition(input_scaled)
            
        if prediction == 1:
            set_background(ALERT_GIF)
            st.markdown(f"""
                <div class="result-card" style="
                    background-color: #ff4d4d;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 4px 4px 12px rgba(255,26,26,0.7), -4px -4px 12px rgba(255,102,102,0.7);
                    color: white;
                    text-align: center;
                    font-size: 22px;">
                    🚨 <b>High Risk:</b> The employee is <b>likely to leave</b> with a probability of <b>{probability:.2%}</b>.
                </div>
                <br>
                <div class="result-card" style="
                    background-color: #ffcc00;
                    padding: 18px;
                    border-radius: 12px;
                    color: black;
                    font-size: 20px;
                    text-align: center;">
                    💡 <b>Suggestion:</b> Review compensation, job satisfaction, and work–life balance.
                </div>
            """, unsafe_allow_html=True)
        else:
            set_background(SUCCESS_GIF)
            st.markdown(f"""
                <div class="result-card" style="
                    background-color: #32CD32;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 4px 4px 12px rgba(46,139,87,0.7), -4px -4px 12px rgba(60,179,113,0.7);
                    color: white;
                    text-align: center;
                    font-size: 22px;">
                    ✅ <b>Safe:</b> The employee is <b>likely to stay</b> with a probability of <b>{(1 - probability):.2%}</b>.
                </div>
                <br>
                <div class="result-card" style="
                    background-color: #66CDAA;
                    padding: 18px;
                    border-radius: 12px;
                    color: black;
                    font-size: 20px;
                    text-align: center;">
                    🎯 <b>Great Job!</b> Maintain the positive work environment. 🌟
                </div>
            """, unsafe_allow_html=True)

# -------------------------- FOOTER --------------------------
def render_footer():
    current_year = datetime.datetime.now().year
    st.markdown(
        f"""
        <div class="footer">
            <div>
                🚀 Developed by <strong>Harshit Rai</strong> | Powered by AI & ML
            </div>
            <div class="social-icons">
                <a href="https://github.com/Harshitraiii2005" target="_blank">
                    <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" alt="GitHub">
                </a>
                <a href="https://www.linkedin.com/in/harshit-rai-5b91142a8/" target="_blank">
                    <img src="https://img.icons8.com/ios-filled/30/ffffff/linkedin.png" alt="LinkedIn">
                </a>
            </div>
            <div style="margin-top: 10px;">
                &copy; {current_year} All rights reserved.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()
