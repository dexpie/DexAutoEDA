# 🚀 DexAutoEDA - Gacor Edition 

**DexAutoEDA** is the ultimate Python application for Data Scientists and Competition participants. It automates EDA, Data Cleaning, and Machine Learning Baselines in one click.

## ✨ Features (v6 - Chat with Data)
- **💬 NLP Chat Interface ("Jarvis")**:
    - Ask questions in plain English: *"What is the average Sales per Region?"*
    - Auto-generate Plots: *"Plot a histogram of Age"*
    - Powered by **PandasAI** and OpenAI.
- **🤖 Advanced AutoML (v5)**:
    - Train & Compare **XGBoost**, **LightGBM**, **Random Forest**.
    - Leaderboard & Model Download.
- **📈 Time Series Analysis (v4)**:
    - Interactive Line Charts & Resampling.
- **🛠️ Feature Engineering**: One-Hot, Label Encoding, Scaling.
- **Interactive Visualizations**: Plotly integration.

## 📂 Project Structure
```
autoeda/
├── autoeda/           # Core Logic Package
│   ├── chat_utils.py  # NLP Interface (PandasAI)
│   ├── ml_utils.py    # AutoML & Defense
│   ├── loader.py      # Data ingestion
│   ├── cleaner.py     # Quality & Preprocessing
│   ├── eda.py         # Stats analysis
│   ├── visualizer.py  # Plotly Visualization
│   ├── insights.py    # Insight generation
│   └── reporter.py    # HTML export
├── app.py             # Main Streamlit App
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
**For Chat Feature:**
1. Enter your **OpenAI API Key** in the sidebar.
2. Go to **"💬 Chat"** tab.
3. Ask away!
