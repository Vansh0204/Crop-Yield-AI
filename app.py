import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import plotly.express as px
from pdf_exporter import export_advisory_pdf
import agent
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="AgriAI | Precision Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_resource
def load_assets():
    try:
        # Debug: List current directory
        # st.write(f"CWD: {os.getcwd()}")
        # st.write(f"Files: {os.listdir('.')}")
        
        model_path = 'model/crop_model.pkl'
        data_path = 'data/crop_yield.csv'
        
        if not os.path.exists(model_path):
            st.error(f"Missing model file: {model_path}")
            return None, None, None
            
        if not os.path.exists(data_path):
            st.error(f"Missing data file: {data_path}")
            return None, None, None

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        return None, None, None

    try:
        with open('known_values.json', 'r') as f:
            known = json.load(f)
    except:
        known = {"countries": sorted(df['Area'].unique()), "crops": sorted(df['Item'].unique())}
    return model, df, known

model, df, known_values = load_assets()
# ... (rest of app remains the same)
