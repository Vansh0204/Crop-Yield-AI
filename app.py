import streamlit as st
import pandas as pd
import joblib

def get_multi_season_predictions(model, row_template, start_year, end_year):
    results = []
    for y in range(start_year, end_year + 1):
        row_enc = row_template.copy()
        row_enc['Year'] = y
        results.append({"Year": y, "Predicted_Yield": float(model.predict(row_enc)[0])})
    return pd.DataFrame(results)

st.title("AgriLogistics Dashboard")
