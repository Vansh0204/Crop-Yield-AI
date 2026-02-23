# Milestone 1 Report: Intelligent Crop Yield Prediction

## Project Understanding
This project addresses the lack of data-driven yield estimation tools for farmers. By analyzing historical agricultural, soil, and weather data, the system provides accurate yield predictions to help in planning and strategy.

## Deliverables Status

| Requirement | Status | Description |
|-------------|--------|-------------|
| Problem Understanding | ✅ | Documented in `idea.md` and this report. |
| Input-Output Spec | ✅ | Documented in `input_output.md`. |
| Architecture Diagram | ✅ | Created in `architecture.md`. |
| Working Application | ✅ | Streamlit app with upload and prediction features. |
| Model Evaluation | ✅ | Metrics calculated and displayed during training. |
| Feature Analysis | ✅ | Feature importance extracted and visualized in UI. |

## Model Performance Results

- **Model Used**: Random Forest Regressor
- **MAE (Mean Absolute Error)**: 3509.59
- **RMSE (Root Mean Squared Error)**: 9534.03
- **R² Score**: 0.987

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python train.py`
3. Launch UI: `streamlit run app.py`

