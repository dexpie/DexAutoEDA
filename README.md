# 🚀 DexAutoEDA - Gacor Edition 

**DexAutoEDA** is the ultimate Python application for Data Scientists and Competition participants. It automates EDA, Data Cleaning, and Machine Learning Baselines in one click.

## ✨ Features (v5 - Advanced AutoML)
- **🤖 Advanced AutoML & Model Comparison**:
    - **Multi-Model Training**: Train **Random Forest**, **XGBoost**, **LightGBM**, and **Linear/Logistic Regression** simultaneously.
    - **Leaderboard**: Compare models based on Accuracy/F1 (Classification) or R2/RMSE (Regression).
    - **Download Model**: One-click download of the best performing model (`.pkl`).
- **📈 Time Series Analysis**:
    - **Interactive Trends**: Visualize data over time with zoomable Range Sliders.
    - **Resampling**: Automatically aggregates data.
- **🛠️ Advanced Feature Engineering**:
    - **One-Hot & Label Encoding**.
    - **Standard & MinMax Scaling**.
- **Interactive Visualizations**: Zoom, pan, and hover over charts using **Plotly**.
- **Actionable Data Cleaning**: Smart imputation and duplicate removal.

## 📂 Project Structure
```
autoeda/
├── autoeda/           # Core Logic Package
│   ├── ml_utils.py    # AutoML & Modeling Logic
│   ├── loader.py      # Data ingestion
│   ├── cleaner.py     # Quality & Preprocessing
│   ├── eda.py         # Stats analysis
│   ├── visualizer.py  # Plotly Visualization
│   ├── insights.py    # Insight generation
│   └── reporter.py    # HTML export
├── app.py             # Main Streamlit App
├── examples/          # Example Datasets
└── requirements.txt   # Dependencies
```

## 🛠️ Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage
Run the application using Streamlit:
```bash
streamlit run app.py
```
1. Go to **"🤖 AutoML"** tab.
2. Select your Target Column and choose models to compare.
3. Click **"Train & Compare"** 🚀.
