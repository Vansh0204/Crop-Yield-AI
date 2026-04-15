import streamlit as st
import pandas as pd
import joblib
import json
from pdf_exporter import export_advisory_pdf
import agent

st.set_page_config(page_title="AgriLogistics Premium", layout="wide")
nav = st.sidebar.radio("Navigation", ["📈 Yield Dashboard", "🤖 Farm Advisory (AI)"])

if nav == "🤖 Farm Advisory (AI)":
    st.header("🤖 Pro-Grade Farm Advisory")
    area = st.selectbox("Area", ["India", "USA"])
    crop = st.selectbox("Crop", ["Wheat", "Maize"])
    if st.button("Generate Farm Advisory"):
        with st.spinner("Consulting AI..."):
            result = agent.run_advisory({"area": area, "crop": crop})
            st.markdown(result["advisory_report"])
            st.download_button("📄 Download PDF", export_advisory_pdf(result))
