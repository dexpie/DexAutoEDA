# 🚀 DexAutoEDA - Gacor Edition 

**DexAutoEDA** is the ultimate Python application for Data Scientists and Competition participants. It automates EDA, Data Cleaning, and Machine Learning Baselines in one click.

## ✨ "Gacor" Features (v3)
- **🤖 Zero-Code AutoML**: 
    - Automatically detects if your problem is **Classification** or **Regression**.
    - Trains a baseline **Random Forest** model.
    - Displays **Feature Importance** to help you select the best features.
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
2. Go to **"🤖 AutoML"** tab.
3. Select your Target Column and click **"Train Baseline Model"**.
4. Get instant results!
