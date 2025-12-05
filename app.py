import streamlit as st
import requests
import os
import joblib
import pandas as pd
import plotly.express as px

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Prediction",
    layout="wide",
    page_icon="🌱"
)

# -----------------------------------------------------
# CUSTOM CSS FOR UI + MOBILE RESPONSIVENESS
# -----------------------------------------------------
st.markdown("""
    <style>
    body, html { margin: 0; padding: 0; }

    .title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
        margin-top: 8px;
    }

    .subtitle {
        text-align: center;
        opacity: 0.7;
        margin-bottom: 20px;
    }

    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        font-size: 1.8rem;
        text-align: center;
        font-weight: bold;
        animation: fadeIn 1s ease-in-out;
    }

    .light .prediction-card {
        background: #E8FFE8;
        border: 2px solid #2E8B57;
        color: #2E8B57;
    }

    .dark .prediction-card {
        background: #213a2c;
        border: 2px solid #7cff9f;
        color: #7cff9f;
    }

    @media(max-width: 600px){
        .title {font-size: 2rem;}
        .subtitle {font-size: 1rem;}
        .stButton>button {width:100%; font-size:1.1rem;}
    }

    .indicator-box {
        padding: 12px;
        border-radius: 12px;
        font-weight: bold;
        text-align:center;
        color:white;
    }

    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# MODEL DOWNLOAD + LOAD
# -----------------------------------------------------
MODEL_URL = "https://huggingface.co/spaces/chetanbajiya/crop-yield-api/resolve/main/yield_model5.pkl"
MODEL_PATH = "yield_model5.pkl"

@st.cache_resource
def load_model():
    status = st.empty()
    status.info("⬇️ Downloading & loading ML model...")

    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            status.error("❌ Model download failed.")
            st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        status.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        status.error("❌ Model load error.")
        st.write(e)
        st.stop()

model = load_model()

# -----------------------------------------------------
# LOAD SUPPORT FILES
# -----------------------------------------------------
try:
    state_list = joblib.load("state_list.pkl")
    crop_list = joblib.load("crop_list.pkl")
    state_district_map = joblib.load("state_district_map.pkl")
except:
    st.error("❌ Missing PKL files in GitHub repo.")
    st.stop()

# -----------------------------------------------------
# TITLE + SUBTITLE
# -----------------------------------------------------
st.markdown("<div class='title'>🌾 Crop Yield Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered Agriculture • Mobile Friendly • Light/Dark Optimized</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------
st.sidebar.header("⚙️ Input Parameters")

state = st.sidebar.selectbox("State", sorted(state_list))
district = st.sidebar.selectbox("District", sorted(state_district_map.get(state, [])))
crop = st.sidebar.selectbox("Crop", sorted(crop_list))

year = st.sidebar.number_input("Crop Year", min_value=1990, max_value=2035, value=2023)
temperature = st.sidebar.number_input("Temperature (°C)", value=30.0)
humidity = st.sidebar.number_input("Humidity (%)", value=60.0)
soil_moisture = st.sidebar.number_input("Soil Moisture", value=0.30)

# -----------------------------------------------------
# WEATHER INDICATORS (COLOR BASED)
# -----------------------------------------------------
def get_color(value, low, high):
    if value < low:
        return "#FF8C00"  # orange
    elif value > high:
        return "#FF0000"  # red
    else:
        return "#2E8B57"  # green

temp_color = get_color(temperature, 20, 32)
hum_color = get_color(humidity, 30, 70)
sm_color = get_color(soil_moisture, 0.2, 0.45)

colA, colB, colC = st.columns(3)

with colA:
    st.markdown(f"<div class='indicator-box' style='background:{temp_color}'>🌡 Temp: {temperature}°C</div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<div class='indicator-box' style='background:{hum_color}'>💧 Humidity: {humidity}%</div>", unsafe_allow_html=True)
with colC:
    st.markdown(f"<div class='indicator-box' style='background:{sm_color}'>🌱 Soil Moisture: {soil_moisture}</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# CLIMATE TREND GRAPH (SIMULATED OR REAL DATA)
# -----------------------------------------------------
trend_data = pd.DataFrame({
    "Year": list(range(year-6, year+1)),
    "Temperature": [temperature-3, temperature-2, temperature-1, temperature, temperature+1, temperature-1, temperature],
    "Humidity": [humidity-5, humidity-3, humidity, humidity+2, humidity-1, humidity, humidity+1],
    "Soil_Moisture": [soil_moisture-0.05, soil_moisture-0.03, soil_moisture, soil_moisture+0.02, soil_moisture, soil_moisture+0.03, soil_moisture]
})

st.subheader("📈 Climate Trend (Past 7 Years)")

fig1 = px.line(trend_data, x="Year", y=["Temperature", "Humidity", "Soil_Moisture"],
               markers=True, title="Climate Trend Over Time")
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
st.write("")
center_col = st.columns([1, 1, 1])[1]

with center_col:
    predict_btn = st.button("🚀 Predict Yield", use_container_width=True)

if predict_btn:
    try:
        X = pd.DataFrame([{
            "State_Name": state,
            "District_Name": district,
            "Crop": crop,
            "Crop_Year": int(year),
            "Temperature": float(temperature),
            "Humidity": float(humidity),
            "Soil_Moisture": float(soil_moisture)
        }])

        pred = model.predict(X)[0]

        st.markdown(
            f"""
            <div class='prediction-card'>
                🌱 Predicted Yield <br> 
                <span style='font-size:2.5rem;'>{round(pred,2)} t/ha</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write(e)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.write("---")
st.markdown("<center>🛠 Developed by <b>Chetan Bajiya</b> • AI for Agriculture 🌍</center>", unsafe_allow_html=True)

