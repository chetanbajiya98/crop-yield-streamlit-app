import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os

# =========================================================
# 1. LOAD ML MODEL FROM HUGGING FACE
# =========================================================
MODEL_URL = "https://huggingface.co/chetanbajiya/crop-yield-model/blob/be9413013cc986294f36e30729c53d5a7feaac00/yield_model.pkl"
MODEL_PATH = "yield_model.pkl"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading ML model from Hugging Face...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error("❌ Model loading failed. Likely due to version mismatch.")
        st.code(str(e))
        st.stop()

model = load_model()

# =========================================================
# 2. LOAD METADATA FILES FROM GITHUB
# =========================================================
@st.cache_resource
def load_metadata():
    crop_list = joblib.load("crop_list.pkl")
    state_list = joblib.load("state_list.pkl")
    state_district_map = joblib.load("state_district_map.pkl")
    return crop_list, state_list, state_district_map

crop_list, state_list, state_district_map = load_metadata()

# =========================================================
# 3. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction System")
st.markdown(
    "Predict **crop yield (tonne per hectare)** using a machine learning model."
)

# =========================================================
# 4. SIDEBAR INPUTS
# =========================================================
st.sidebar.header("📍 Location")

state = st.sidebar.selectbox("State", state_list)

district = st.sidebar.selectbox(
    "District",
    sorted(state_district_map[state])
)

st.sidebar.header("🌱 Crop & Year")

crop = st.sidebar.selectbox("Crop", crop_list)

year = st.sidebar.number_input("Crop Year", 2000, 2050, 2022)

st.sidebar.header("🌦 Weather")

temperature = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 28.0)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 60.0)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 5.0, 60.0, 25.0)

predict_btn = st.sidebar.button("🚀 Predict Yield")

# =========================================================
# 5. PREDICTION
# =========================================================
if predict_btn:
    input_df = pd.DataFrame([{
        "State_Name": state,
        "District_Name": district,
        "Crop": crop,
        "Crop_Year": year,
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil_Moisture": soil_moisture
    }])

    with st.spinner("Predicting yield..."):
        log_yield = model.predict(input_df)[0]
        yield_t_ha = np.expm1(log_yield)

    st.success("✅ Prediction completed")

    st.metric("🌾 Predicted Yield (t/ha)", f"{yield_t_ha:.2f}")

    st.markdown("### 📋 Input Summary")
    st.table(input_df)

# =========================================================
# 6. FOOTER
# =========================================================
st.markdown(
    """
    ---
    **Model Information**
    - Algorithm: Extra Trees Regressor  
    - Target: Yield (tonne/ha)  
    - Coverage: All crops, states & districts in training data  
    """
)
