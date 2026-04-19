import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import plotly.express as px
from pdf_exporter import export_advisory_pdf
import agent

# Page Config
st.set_page_config(
    page_title="AgriLogistics Premium | Yield AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_resource
def load_assets():
    model = joblib.load('model/crop_model.pkl')
    # Using the verified data path from the project directory
    data = pd.read_csv('data/crop_yield.csv')
    try:
        with open('known_values.json', 'r') as f:
            known = json.load(f)
    except:
        known = {"countries": ["India"], "crops": ["Wheat"]}
    return model, data, known

model, df, known_values = load_assets()

# Sidebar Navigation with custom branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913520.png", width=80)
    st.title("AgriLogistics AI")
    st.markdown("---")
    nav = st.radio("Navigation", ["📈 Yield Dashboard", "🤖 Farm Advisory (AI)"])
    st.markdown("---")
    st.info("Milestone 2: Agentic AI & RAG Integration Complete.")

if nav == "📈 Yield Dashboard":
    st.title("🌿 Agricultural Yield Analytics")
    
    # Dashboard Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Dataset Size", f"{len(df):,}")
    m2.metric("Total Countries", len(df['Area'].unique()))
    m3.metric("Crop Varieties", len(df['Item'].unique()))
    
    st.markdown("---")
    
    # Simple Visualization Example
    st.subheader("Historical Yield Trends")
    selected_area = st.selectbox("Select Region", df['Area'].unique()[:10])
    area_df = df[df['Area'] == selected_area]
    fig = px.line(area_df, x="Year", y="Value", color="Item", title=f"Yield in {selected_area}")
    st.plotly_chart(fig, use_container_width=True)

elif nav == "🤖 Farm Advisory (AI)":
    st.header("🤖 Pro-Grade Farm Advisory")
    st.markdown("Consult our AI agronomist for personalized crop strategies grounded in real-time data.")
    
    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Area", known_values["countries"])
        item = st.selectbox("Crop Type", known_values["crops"])
    with col2:
        year = st.slider("Target Implementation Year", 2024, 2030, 2024)
        rainfall = st.number_input("Expected Rainfall (mm)", 0, 5000, 1050)
    
    if st.button("Generate Farm Advisory Report"):
        with st.spinner("Consulting agricultural AI advisor..."):
            # Execute the LangGraph Agent
            inputs = {
                "area": area, 
                "crop": item, 
                "year": year, 
                "rainfall": float(rainfall),
                "temperature": 20.0, 
                "pesticides": 12000.0
            }
            result = agent.run_advisory(inputs)
            
            # Show Warnings if any (Data Sanitization Feedback)
            if result.get("warnings"):
                with st.expander("⚠️ Data Quality Adjustments"):
                    for w in result["warnings"]:
                        st.warning(w)

            # Display the result in a styled card
            st.success("Advisory Report Generated Successfully!")
            st.markdown("---")
            st.markdown(result.get("advisory_report", "Error generating report."))
            
            # PDF Export
            st.markdown("---")
            pdf_bytes = export_advisory_pdf(result)
            st.download_button(
                label="📄 Download Institutional PDF Report",
                data=pdf_bytes,
                file_name=f"AgriAI_Report_{item}_{area}.pdf",
                mime="application/pdf"
            )
