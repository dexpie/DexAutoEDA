# 🚀 DexAutoEDA - Gacor Edition 

**DexAutoEDA** is the ultimate Python application for Data Scientists and Competition participants. It automates EDA, Data Cleaning, and Machine Learning Baselines in one click.

## ✨ "Gacor" Features (v4 - Time Series)
- **📈 Time Series Analysis**:
    - **Interactive Trends**: Visualize data over time with zoomable Range Sliders.
    - **Resampling**: Automatically aggregates data (e.g., Monthly averages) to see long-term patterns.
    - **Smart Date Detection**: One-click conversion of text to datetime.
- **🤖 Zero-Code AutoML**: 
    - Automatically detects if your problem is **Classification** or **Regression**.
    - Trains a baseline **Random Forest** model.
    - Displays **Feature Importance**.
- **🛠️ Advanced Feature Engineering**:
    - **One-Hot & Label Encoding** for categorical data.
    - **Standard & MinMax Scaling** for numeric data.
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
1. Upload your CSV.
2. For Time Series, go to **"📈 Time Series"** tab and convert your date column.
