# Intelligent Crop Yield Prediction System 🌾

This project is an AI-driven agricultural analytics system designed to help farmers and agricultural planners predict crop yields using historical data. It analyzes factors such as rainfall, temperature, fertilizer usage, and crop types to provide accurate, data-backed yield estimations.

## 🚀 Milestone 1: ML-Based Yield Prediction

The first phase of the system focuses on a robust machine learning pipeline for yield forecasting.

### Key Features
- **Data Preprocessing**: Handles raw agricultural data and prepares it for modeling.
- **Random Forest Regressor**: Uses a powerful ensemble learning method for accurate predictions.
- **Factor Identification**: Identifies and visualizes the most significant factors driving crop yield.
- **Batch Processing**: Allows users to upload CSV files for bulk predictions.

### 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "AIML Crop Project copy"
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 📈 Usage

1. **Train the Model**:
   Generate the latest model and feature importance data.
   ```bash
   python train.py
   ```

2. **Launch the Web Dashboard**:
   Start the interactive Streamlit interface.
   ```bash
   streamlit run app.py
   ```

3. **Make Predictions**:
   Upload your farm data in CSV format through the dashboard to get instant yield predictions and analysis.

## 🏗️ Project Structure
- `train.py`: The machine learning training pipeline.
- `app.py`: Streamlit-based user interface.
- `data/`: Contains the dataset used for training and testing.
- `model/`: Stores the trained model (`.pkl`) and analytical data (`.json`).
- `architecture.md`: Detailed system architecture documentation.
- `MILESTONE1_REPORT.md`: Formal report on Milestone 1 progress and performance.

---
*Developed as part of the Intelligent Crop Yield Prediction and Agentic Farm Advisory System project.*

