import streamlit as st
import requests
import pickle
import os

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# -----------------------
# DOWNLOAD MODEL
# -----------------------

MODEL_URL = "https://huggingface.co/spaces/chetanbajiya/crop-yield-api/resolve/main/yield_model5.pkl"
MODEL_PATH = "yield_model5.pkl"

@st.cache_resource
def load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading ML model from Hugging Face...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ Failed to download model.")
            st.stop()

    # Load model using pickle (more stable than joblib)
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("❌ Model loading failed. Likely version mismatch.")
        st.write(e)
        st.stop()

model = load_model()

# -----------------------
# Load state, district, crop PKL files
# -----------------------

import joblib

state_list = joblib.load("state_list.pkl")
crop_list = joblib.load("crop_list.pkl")
state_district_map = joblib.load("state_district_map.pkl")

# -----------------------
# Streamlit UI
# -----------------------

st.title("🌾 Crop Yield Prediction System")
st.markdown("Predict crop yield using weather, soil, and crop information.")

state = st.selectbox("Select State", sorted(state_list))
districts = state_district_map.get(state, [])
district = st.selectbox("Select District", sorted(districts))
crop = st.selectbox("Select Crop", sorted(crop_list))

year = st.number_input("Crop Year", min_value=1990, max_value=2035, value=2023)
temperature = st.number_input("Temperature (°C)", value=30.0)
humidity = st.number_input("Humidity (%)", value=60.0)
soil_moisture = st.number_input("Soil Moisture", value=0.25)

# -----------------------
# Predict
# -----------------------

if st.button("🚀 Predict Yield"):
    try:
        features = [[
            temperature,
            humidity,
            soil_moisture
        ]]

        pred = model.predict(features)[0]

        st.success(f"🌾 **Predicted Yield: {round(pred,2)} t/ha**")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write(e)

