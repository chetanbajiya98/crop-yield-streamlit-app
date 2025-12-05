import streamlit as st
import requests
import os
import joblib

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# ===============================================================
# 1️⃣ DOWNLOAD & LOAD ML MODEL FROM HUGGINGFACE
# ===============================================================

MODEL_URL = "https://huggingface.co/spaces/chetanbajiya/crop-yield-api/resolve/main/yield_model5.pkl"
MODEL_PATH = "yield_model5.pkl"

@st.cache_resource
def load_model():
    # Download model if not already downloaded
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading ML model from Hugging Face...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ Failed to download model.")
            st.stop()

    # Load model using joblib
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Model loading failed.")
        st.write(e)
        st.stop()

model = load_model()

# ===============================================================
# 2️⃣ LOAD SUPPORTING FILES (State, District, Crop lists)
# ===============================================================

state_list = joblib.load("state_list.pkl")
crop_list = joblib.load("crop_list.pkl")
state_district_map = joblib.load("state_district_map.pkl")

# ===============================================================
# 3️⃣ STREAMLIT UI
# ===============================================================

st.title("🌾 Crop Yield Prediction System")

st.markdown("""
Use this tool to predict crop yield based on:
- 🌡 Temperature  
- 💧 Humidity  
- 🌱 Soil Moisture  
- 📍 State, District, Crop  
""")

# Dropdown inputs
state = st.selectbox("Select State", sorted(state_list))
districts = sorted(state_district_map.get(state, []))
district = st.selectbox("Select District", districts)
crop = st.selectbox("Select Crop", sorted(crop_list))

# Numeric inputs
year = st.number_input("Crop Year", min_value=1990, max_value=2035, value=2023)
temperature = st.number_input("Temperature (°C)", value=30.0)
humidity = st.number_input("Humidity (%)", value=60.0)
soil_moisture = st.number_input("Soil Moisture", value=0.25)

# ===============================================================
# 4️⃣ PREDICTION
# ===============================================================

if st.button("🚀 Predict Yield"):
    try:
        # Prepare features for model
        X = [[temperature, humidity, soil_moisture]]

        # Prediction
        prediction = model.predict(X)[0]

        st.success(f"🌾 **Predicted Yield: {round(prediction, 2)} t/ha**")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write(e)
