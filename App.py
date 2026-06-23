import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# 1. LOAD MODEL ARTIFACTS (WITH AUTOMATIC NOTFITTEDERROR REPAIR)
# ----------------------------
@st.cache_resource
def load_artifacts():
    # Load core model structures
    model = joblib.load("model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    numerical_features = joblib.load("numerical_features.pkl")
    categorical_features = joblib.load("categorical_features.pkl")
    
    # Load the scaler object saved by joblib
    scaler = joblib.load("scaler.pkl")
    
    # Safety Check: Validate if the loaded scaler object is properly fitted
    from sklearn.utils.validation import check_is_fitted
    from sklearn.preprocessing import StandardScaler
    
    try:
        check_is_fitted(scaler)
    except:
        # If a NotFittedError is captured, rebuild and re-fit a fresh instance from the training source
        st.warning("⚠️ Scaler state was missing fitted parameters upon import. Re-fitting automatically from AD.csv...")
        try:
            raw_df = pd.read_csv(r"C:\Users\rosha\Desktop\ds\project roshaan\AD.csv")
            scaler = StandardScaler()
            scaler.fit(raw_df[numerical_features])
        except Exception as repair_error:
            st.error(f"Failed to automatically re-fit scaler: {repair_error}")
        
    return model, scaler, feature_columns, numerical_features, categorical_features

model, scaler, feature_columns, numerical_features, categorical_features = load_artifacts()

# ----------------------------
# 2. UI THEME & STYLING LAYER
# ----------------------------
st.set_page_config(page_title="Neuro Scan Pro",page_icon="🧠",
                   layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: -30px !important;
    }
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    
    .main-header {
        color: #58A6FF;
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        padding: 25px 0px 15px 0px;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #30363D;
        width: 100%;
    }
    
    div[data-testid="stForm"] { 
        background-color: #161B22; 
        border: 1px solid #30363D; 
        border-radius: 12px; 
        padding: 35px; 
        margin-top: 10px;
    }
    
    label { color: #8B949E !important; font-size: 13px !important; font-weight: 600 !important; }
    
    .section-header {
        color: #58A6FF !important;
        font-size: 26px !important; 
        font-weight: 700 !important;
        margin-top: 10px !important;
        margin-bottom: 15px !important;
        padding-left: 0px;
    }
    
    .stRadio div[role="radiogroup"] { gap: 15px; }
    
    .binary-label {
        font-size: 14px;
        font-weight: 600;
        color: #E6EDF3;
        padding-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">NEUROSCAN: CLINICAL DIAGNOSTIC PORTAL</div>', unsafe_allow_html=True)

# ----------------------------
# 3. PATIENT DATA FORM (ALL-IN-ONE PAGE)
# ----------------------------
with st.form("clinical_form"):
    user_inputs = {}
    
    # --- TOP PORTION: 3 MAIN CORE METRIC COLUMNS ---
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown('<div class="section-header">📋 Demographics</div>', unsafe_allow_html=True)
        user_inputs['Age'] = st.number_input("Age (60-90)", 60, 90, 75)
        user_inputs['Gender'] = st.radio("Gender (0: Male, 1: Female)", [0, 1], horizontal=True)
        user_inputs['Ethnicity'] = st.radio("Ethnicity (0:Cauc, 1:Afr, 2:Asn, 3:Oth)", [0, 1, 2, 3], horizontal=True)
        user_inputs['EducationLevel'] = st.radio("Education (0:None, 1:HS, 2:Bach, 3:High)", [0, 1, 2, 3], horizontal=True)
        user_inputs['BMI'] = st.number_input("BMI (15-40)", 15.0, 40.0, 18.00, format="%.2f")

    with col2:
        st.markdown('<div class="section-header">🩸 Vitals & Lipid Profile</div>', unsafe_allow_html=True)
        user_inputs['SystolicBP'] = st.number_input("Systolic BP (90-180)", 90, 180, 117)
        user_inputs['DiastolicBP'] = st.number_input("Diastolic BP (60-120)", 60, 120, 63)
        user_inputs['CholesterolTotal'] = st.number_input("Total Cholesterol", 150.0, 300.0, 151.38, format="%.2f")
        user_inputs['CholesterolLDL'] = st.number_input("LDL (50-200)", 50.0, 200.0, 69.62, format="%.2f")
        user_inputs['CholesterolHDL'] = st.number_input("HDL (20-100)", 20.0, 100.0, 77.00, format="%.2f")
        user_inputs['CholesterolTriglycerides'] = st.number_input("Triglycerides (50-400)", 50.0, 400.0, 210.57, format="%.2f")

    with col3:
        st.markdown('<div class="section-header">🚶🏻‍♀️ Lifestyle & Cognitive</div>', unsafe_allow_html=True)
        user_inputs['MMSE'] = st.number_input("MMSE Score (0-30)", 0.0, 30.0, 10.1, format="%.1f")
        user_inputs['ADL'] = st.number_input("ADL Score (0-10)", 0.0, 10.0, 4.5, format="%.1f")
        user_inputs['FunctionalAssessment'] = st.number_input("Functional Assessment (0-10)", 0.0, 10.0, 3.4, format="%.1f")
        user_inputs['AlcoholConsumption'] = st.number_input("Alcohol (0-20 Units)", 0.0, 20.0, 13.7, format="%.1f")
        user_inputs['PhysicalActivity'] = st.number_input("Activity (0-10 Hrs)", 0.0, 10.0, 4.6, format="%.1f")
        user_inputs['DietQuality'] = st.number_input("Diet Score (0-10)", 0.0, 10.0, 8.3, format="%.1f")
        user_inputs['SleepQuality'] = st.number_input("Sleep Score (4-10)", 4.0, 10.0, 7.0, format="%.1f")

    st.markdown("<br><hr style='border:1px solid #30363D'>", unsafe_allow_html=True)
    
    # --- BOTTOM PORTION: ALL CLINICAL BINARY INDICATORS DISPLAYED IN COLUMNS ---
    st.markdown('<div class="section-header">⚠️ Clinical Binary Indicators</div>', unsafe_allow_html=True)
    
    handled_keys = list(user_inputs.keys())
    remaining_cats = [c for c in categorical_features if c not in handled_keys]
    
    for i in range(0, len(remaining_cats), 2):
        row_cols = st.columns([2, 1, 2, 1])
        
        # Left side indicator column entry
        with row_cols[0]: 
            st.markdown(f'<div class="binary-label">{remaining_cats[i]}</div>', unsafe_allow_html=True)
        with row_cols[1]: 
            user_inputs[remaining_cats[i]] = st.radio(remaining_cats[i], [0, 1], horizontal=True, label_visibility="collapsed", key=f"bin_{i}")
        
        # Right side indicator column entry
        if i + 1 < len(remaining_cats):
            with row_cols[2]: 
                st.markdown(f'<div class="binary-label">{remaining_cats[i+1]}</div>', unsafe_allow_html=True)
            with row_cols[3]: 
                user_inputs[remaining_cats[i+1]] = st.radio(remaining_cats[i+1], [0, 1], horizontal=True, label_visibility="collapsed", key=f"bin_{i+1}")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("GENERATE CLINICAL PREDICTION")

# ----------------------------
# 4. PREDICTION INFERENCE PIPELINE
# ----------------------------
if submit:
    # Build array structured vector mapping model features sequence
    input_df = pd.DataFrame([user_inputs])[feature_columns]
    
    # Safely transform metrics utilizing the verified scaled parameters
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Perform standard forward inferencing steps
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Render standardized interface dashboard matching your templates
    st.markdown("---")
    res1, res2 = st.columns(2)
    
    with res1:
        if prediction == 1:
            st.error("### 🚩 RESULT: POSITIVE DIAGNOSIS INDICATED")
        else:
            st.success("### ✅ RESULT: NEGATIVE DIAGNOSIS INDICATED")
    
    with res2:
        st.write(f"**Diagnostic Confidence:** `{prob*100:.2f}%`")
        st.progress(prob)
