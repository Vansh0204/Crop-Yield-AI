import streamlit as st
import agent
from pdf_exporter import export_advisory_pdf

st.set_page_config(page_title="AgriLogistics Premium")
nav = st.sidebar.radio("Navigation", ["Dashboard", "AI Advisor"])

if nav == "AI Advisor":
    st.header("🤖 AI Farm Advisor")
    if st.button("Generate"):
        res = agent.run_advisory({"crop": "Wheat"})
        st.write(res["advisory_report"])
        st.download_button("Download PDF", export_advisory_pdf(res))
