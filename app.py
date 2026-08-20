import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Salary Predictor Pro | By Ameer Hamza",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3.2em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .developer-badge {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: #f8fafc;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0f172a;
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

# Load Trained Model and Feature Names
@st.cache_resource
def load_model():
    loaded_data = joblib.load('salary_gb_model.pkl')
    return loaded_data['model'], loaded_data['features']

try:
    model, feature_names = load_model()
except Exception as e:
    st.error("Error loading model file. Please ensure 'salary_gb_model.pkl' is in the current directory.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("💼 Salary Predictor Pro")
    st.caption("🚀 Machine Learning Web Application")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Profile")
    st.markdown("**Lead Developer:** Ameer Hamza")
    st.markdown("**Project:** Salary Prediction Model")
    st.markdown("**Stack:** Scikit-Learn, Streamlit, Python")
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Performance")
    st.info("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Model Accuracy (R²):** ~89.5%")
    st.write("**Error Margin (RMSE):** ~$19.4k")

# --- MAIN DASHBOARD HEADER ---
st.markdown("""
    <div class="developer-badge">
        <span style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8;">Designed & Developed By</span>
        <h2 style="margin: 0; color: #ffffff;">Ameer Hamza</h2>
    </div>
""", unsafe_allow_html=True)

st.title("💼 Enterprise Salary Intelligence Platform")
st.markdown("Accurately predict market-competitive employee compensation packages using Machine Learning.")
st.divider()

# --- INPUT FORM & CONTROLS ---
col_form, col_space, col_stats = st.columns([1.2, 0.1, 1])

with col_form:
    st.subheader("👤 Candidate Profile Inputs")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=70, value=28, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    with col2:
        years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=4.0, step=0.5)
        job_title = st.selectbox("Job Title / Role", [
            "Software Engineer", "Data Scientist", "Product Manager", 
            "Marketing Manager", "Full Stack Engineer", "Senior Project Engineer",
            "Software Engineer Manager", "Senior Software Engineer", 
            "Front end Developer", "Other"
        ])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Calculate Estimated Salary")

# Education Encoding Mapping
edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}

# --- PREDICTION & OUTPUT AREA ---
with col_stats:
    st.subheader("📊 Compensation Overview")
    
    if predict_btn:
        # Prepare input dictionary with model's expected features
        input_dict = {col: 0 for col in feature_names}
        input_dict['Age'] = age
        input_dict['Years of Experience'] = years_exp
        input_dict['Education_Encoded'] = edu_map[education]
        
        if f'Gender_{gender}' in input_dict:
            input_dict[f'Gender_{gender}'] = 1
            
        if f'Job Title Grouped_{job_title}' in input_dict:
            input_dict[f'Job Title Grouped_{job_title}'] = 1

        input_df = pd.DataFrame([input_dict])[feature_names]
        
        # Predict
        predicted_annual = model.predict(input_df)[0]
        predicted_monthly = predicted_annual / 12

        # Display Result Metrics
        st.markdown(f"""
            <div style="background-color: #ffffff; padding: 25px; border-radius: 12px; border-left: 6px solid #007bff; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <h4 style="color: #007bff; margin:0;">Estimated Annual Package</h4>
                <h1 style="color: #1e293b; margin:0; font-size: 2.5em;">${predicted_annual:,.2f}</h1>
                <p style="color: #64748b; margin-top: 5px;">Estimated Monthly: <b>${predicted_monthly:,.2f}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Additional Metrics Cards
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Primary Impact Factor", value="Experience", delta=f"{years_exp} Yrs")
        with m_col2:
            st.metric(label="Education Weight", value=education, delta=f"Level {edu_map[education]}")
            
    else:
        st.info("👈 Fill candidate parameters and click **Calculate Estimated Salary** to generate compensation analytics.")

# --- ANALYTICS CHART SECTION ---
if predict_btn:
    st.divider()
    st.subheader("📈 Experience vs. Salary Growth Trajectory")
    
    # Generate interactive curve for experience progression
    exp_range = np.linspace(0, 30, 31)
    curve_data = []
    
    for exp in exp_range:
        temp_dict = input_dict.copy()
        temp_dict['Years of Experience'] = exp
        temp_df = pd.DataFrame([temp_dict])[feature_names]
        curve_data.append(model.predict(temp_df)[0])
        
    chart_df = pd.DataFrame({
        'Years of Experience': exp_range,
        'Predicted Salary ($)': curve_data
    }).set_index('Years of Experience')
    
    st.line_chart(chart_df)

# --- FOOTER CREDITS ---
st.markdown("""
    <div class="footer">
        Developed with ❤️ by <b>Ameer Hamza</b> | Machine Learning Project
    </div>
""", unsafe_allow_html=True)