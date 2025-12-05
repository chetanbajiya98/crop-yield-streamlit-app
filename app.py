import streamlit as st
import joblib
import requests
import os

MODEL_URL = "https://huggingface.co/spaces/chetanbajiya/crop-yield-api/resolve/main/yield_model5.pkl"

@st.cache_resource
def load_model():
    local_model = "yield_model5.pkl"

    if not os.path.exists(local_model):
        st.write("⬇️ Downloading ML model from Hugging Face...")
        r = requests.get(MODEL_URL)
        if r.status_code == 200:
            with open(local_model, "wb") as f:
                f.write(r.content)
        else:
            st.error("❌ Failed to download model.")
            st.stop()

    # Try loading the model
    try:
        model = joblib.load(local_model)
        return model
    except Exception as e:
        st.error("❌ Model loading failed. Likely due to sklearn version mismatch.")
        st.write(e)
        st.stop()
