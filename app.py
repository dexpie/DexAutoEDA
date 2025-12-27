import streamlit as st
import pandas as pd
import numpy as np
from autoeda import loader, cleaner, eda, visualizer, insights, reporter, ml_utils

# Page Config
st.set_page_config(
    page_title="DexAutoEDA v2 - ML Ready",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("🚀 DexAutoEDA - Gacor Edition")
st.markdown("### Interactive EDA & Zero-Code AutoML")

# Sidebar
st.sidebar.header("📁 Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Initialize session state for dataframe if not invalid
if 'df' not in st.session_state:
    st.session_state['df'] = None

if uploaded_file is not None:
    try:
        current_file = uploaded_file
        
        # Check if we need to reload (e.g. initial load)
        if st.session_state['df'] is None:
            df_load = loader.load_data(current_file)
            st.session_state['df'] = df_load
            st.sidebar.success("File Uploaded Successfully!")
            st.rerun() 
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.session_state['df'] = None
    st.info("👆 Please upload a CSV file in the sidebar to begin.")
    st.stop()

# Use Session State Data
df = st.session_state['df']
if df is None:
    st.stop()

# Determine numeric and categorical columns locally
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Main Layout with Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🔍 Overview", 
    "🛠️ Processing",
    "🤖 AutoML",
    "📈 Time Series",
    "🧹 Data Quality", 
    "📊 Statistics", 
    "📉 Visualizations", 
    "💡 Insights", 
    "📑 Report"
])

# Tab 1: Overview
with tab1:
    st.header("Dataset Overview")
    info = loader.get_dataset_info(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Rows</h3><h2>{info['rows']}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Columns</h3><h2>{info['columns']}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Columns Types</h3><p>{len(numeric_cols)} Num, {len(categorical_cols)} Cat</p></div>", unsafe_allow_html=True)

    st.subheader("First 5 Rows")
    st.dataframe(df.head())
    
    st.subheader("Data Types")
    st.json(info['dtypes'])

# Tab 2: Processing
with tab2:
    st.header("🛠️ Feature Engineering & Cleaning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Cleaning")
        
        # 1. Drop Duplicates
        if st.button("🗑️ Remove Duplicates"):
            before = len(df)
            df = cleaner.drop_duplicates(df)
            st.session_state['df'] = df
            st.success(f"Removed {before - len(df)} duplicate rows.")
            st.rerun()
            
        # 2. Impute Missing
        if st.button("✨ Impute Missing Values"):
            df = cleaner.impute_missing(df, strategy='mean')
            st.session_state['df'] = df
            st.success("Missing values imputed.")
            st.rerun()
            
        # 3. Drop Columns
        st.divider()
        st.markdown("##### Drop Columns")
        cols_to_drop = st.multiselect("Select columns to drop", df.columns)
        if st.button("❌ Drop Selected Columns"):
            if cols_to_drop:
                df = cleaner.drop_columns(df, cols_to_drop)
                st.session_state['df'] = df
                st.success(f"Dropped: {', '.join(cols_to_drop)}")
                st.rerun()
            else:
                st.warning("Please select columns first.")

    with col2:
        st.subheader("Advanced Preprocessing (ML Ready)")
        
        # 4. Encoding
        st.markdown("##### Encode Categorical")
        enc_cols = st.multiselect("Select Categorical Columns", categorical_cols)
        enc_method = st.selectbox("Encoding Method", ["One-Hot Encoding", "Label Encoding"])
        if st.button("🔢 Apply Encoding"):
            if enc_cols:
                method_code = 'onehot' if 'One-Hot' in enc_method else 'label'
                df = cleaner.encode_categorical(df, enc_cols, method=method_code)
                st.session_state['df'] = df
                st.success(f"Applied {enc_method} on {len(enc_cols)} columns.")
                st.rerun()
                
        # 5. Scaling
        st.divider()
        st.markdown("##### Scale Numeric Features")
        scale_cols = st.multiselect("Select Numeric columns", numeric_cols)
        scale_method = st.selectbox("Scaling Method", ["Standard Scaler (Z-Score)", "MinMax Scaler (0-1)"])
        if st.button("📏 Apply Scaling"):
            if scale_cols:
                method_code = 'standard' if 'Standard' in scale_method else 'minmax'
                df = cleaner.scale_features(df, scale_cols, method=method_code)
                st.session_state['df'] = df
                st.success(f"Applied {scale_method} on {len(scale_cols)} columns.")
                st.rerun()
        
        st.divider()
        st.subheader("Download Processed Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download ML-Ready CSV",
            data=csv,
            file_name="dexautoeda_processed.csv",
            mime="text/csv"
        )
        
        st.subheader("Preview")
        st.dataframe(df.head(5))

# Tab 3: AutoML (Advanced)
with tab3:
    st.header("🤖 Advanced AutoML")
    st.info("Train and compare multiple models (Random Forest, XGBoost, LightGBM) to find the best performer.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        target_col = st.selectbox("Select Target Column (Y)", df.columns)
        
        st.markdown("##### Select Models")
        models_to_train = []
        if st.checkbox("Random Forest", value=True): models_to_train.append("Random Forest")
        if st.checkbox("XGBoost", value=True): models_to_train.append("XGBoost")
        if st.checkbox("LightGBM", value=True): models_to_train.append("LightGBM")
        if st.checkbox("Linear/Logistic Regression", value=True): models_to_train.append("Linear/Logistic")
        
        train_btn = st.button("🚀 Train & Compare")
        
    with col2:
        if train_btn:
             if not models_to_train:
                 st.warning("Please select at least one model.")
             else:
                 with st.spinner(f"Training {len(models_to_train)} models..."):
                    try:
                        results_pkg = ml_utils.train_models(df, target_col, models_to_train)
                        
                        # 1. Comparison Table
                        st.subheader("🏆 Model Leaderboard")
                        results_df = pd.DataFrame(results_pkg['results']).sort_values(
                            by='Accuracy' if results_pkg['problem_type'] == 'classification' else 'R2 Score', 
                            ascending=False
                        )
                        st.dataframe(results_df, use_container_width=True)
                        
                        best_model_name = results_pkg['best_model_name']
                        st.success(f"**Best Model:** {best_model_name}")
                        
                        # 2. Download Best Model
                        model_bytes = ml_utils.save_model(results_pkg['best_model'])
                        st.download_button(
                            label=f"💾 Download {best_model_name} Model (.pkl)",
                            data=model_bytes,
                            file_name=f"best_model_{best_model_name}_dexautoeda.pkl",
                            mime="application/octet-stream"
                        )
                        
                        # 3. Feature Importance (Best Model)
                        if not results_pkg['feature_importance'].empty:
                            st.subheader(f"Feature Importance ({best_model_name})")
                            fig_imp = visualizer.plot_feature_importance(results_pkg['feature_importance'])
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Training failed: {e}")

# Tab 4: Time Series (New!)
with tab4:
    st.header("📈 Time Series Analysis")
    
    # Date Conversion
    dt_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    possible_dt_cols = [col for col in df.columns if df[col].dtype == 'object' or 'date' in col.lower() or 'time' in col.lower()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Date Column Selection")
        # Allow user to pick a column to convert if not already identified
        if dt_cols:
            selected_date_col = st.selectbox("Select Date Column", dt_cols)
        else:
            selected_date_col = st.selectbox("Select Column to Convert to Date", possible_dt_cols)
            if st.button("📅 Convert to Datetime"):
                df = loader.convert_to_datetime(df, selected_date_col)
                st.session_state['df'] = df
                st.success(f"Converted {selected_date_col} to Datetime.")
                st.rerun()

    with col2:
         if pd.api.types.is_datetime64_any_dtype(df[selected_date_col]):
             st.markdown("##### Visualizations")
             value_col_ts = st.selectbox("Select Value to Plot", numeric_cols)
             if value_col_ts:
                 fig_ts = visualizer.plot_time_series(df, selected_date_col, value_col_ts)
                 st.plotly_chart(fig_ts, use_container_width=True)
                 
                 st.markdown("##### Resampled Analysis (Monthly Mean)")
                 resampled_data = eda.get_time_series_summary(df, selected_date_col, value_col_ts, freq='M')
                 if resampled_data is not None:
                     st.line_chart(resampled_data)
         else:
             st.info("Please select or convert a date column first.")

# Tab 5: Data Quality
with tab5:
    st.header("Data Quality Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        missing_df = cleaner.check_missing(df)
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            st.warning("⚠️ Missing values detected!")
        else:
            st.success("✅ No missing values found.")  
    with col2:
        st.subheader("Duplicates")
        dup_count, dup_rows = cleaner.check_duplicates(df)
        st.metric("Duplicate Rows", dup_count)
        if dup_count > 0:
            st.dataframe(dup_rows.head(10))
    st.subheader("Outlier Detection (Numeric)")
    outlier_summary = cleaner.check_outliers(df)
    if outlier_summary:
        st.write(outlier_summary)

# Tab 6: Statistics
with tab6:
    st.header("Descriptive Statistics")
    st.subheader("Numeric Summary")
    st.dataframe(eda.get_descriptive_stats(df), use_container_width=True)
    st.subheader("Distribution (Skewness & Kurtosis)")
    st.dataframe(eda.get_numeric_distribution(df), use_container_width=True)
    st.subheader("Categorical Summary")
    cat_summary = eda.get_categorical_summary(df)
    if cat_summary:
        for col, val_counts in cat_summary.items():
            st.text(f"Column: {col}")
            st.dataframe(val_counts)

# Tab 7: Visualizations
with tab7:
    st.header("Interactive Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Histograms")
        selected_hist_col = st.selectbox("Select Column for Histogram", numeric_cols)
        if selected_hist_col:
            fig_hist = visualizer.plot_histograms(df, selected_hist_col)
            st.plotly_chart(fig_hist, use_container_width=True)   
    with col2:
        st.subheader("Boxplots")
        selected_box_col = st.selectbox("Select Column for Boxplot", numeric_cols)
        if selected_box_col:
            fig_box = visualizer.plot_boxplot(df, selected_box_col)
            st.plotly_chart(fig_box, use_container_width=True)
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
         st.subheader("Bar Charts (Categorical)")
         if categorical_cols:
             selected_bar = st.selectbox("Select Category", categorical_cols)
             fig_bar = visualizer.plot_bar_chart(df, selected_bar)
             st.plotly_chart(fig_bar, use_container_width=True) 
    with col4:
        st.subheader("Scatter Plots")
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X Axis", numeric_cols)
            y_axis = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            color_dim = st.selectbox("Color By (Optional)", [None] + categorical_cols + numeric_cols)
            fig_scat = visualizer.plot_scatter(df, x_axis, y_axis, color_dim)
            st.plotly_chart(fig_scat, use_container_width=True)
            
    st.subheader("Correlation Matrix")
    if len(numeric_cols) > 1:
        corr_matrix = eda.get_correlation(df)
        fig_corr = visualizer.plot_correlation_heatmap(corr_matrix)
        st.plotly_chart(fig_corr, use_container_width=True)

# Tab 8: Insights
with tab8:
    st.header("🤖 Automated Insights")
    missing_df = cleaner.check_missing(df)
    corr_matrix = eda.get_correlation(df) if len(numeric_cols) > 1 else pd.DataFrame()
    outlier_summary = cleaner.check_outliers(df)
    dup_count, _ = cleaner.check_duplicates(df)
    
    st.subheader("Cleaning Recommendations")
    recommendations = cleaner.get_recommendations(missing_df, dup_count, outlier_summary)
    for rec in recommendations:
        st.info(f"💡 {rec}")
        
    st.subheader("Data Findings")
    general_insights = insights.generate_insights(df, missing_df, corr_matrix)
    for ins in general_insights:
        st.success(f"📌 {ins}")

# Tab 9: Report
with tab9:
    st.header("Export Interactive Report")
    st.write("Generate a standalone HTML report containing all analysis and interactive Plotly charts.")
    if st.button("Generate HTML Report"):
        dataset_info = loader.get_dataset_info(df)
        missing_df = cleaner.check_missing(df)
        corr_matrix = eda.get_correlation(df) if len(numeric_cols) > 1 else pd.DataFrame()
        dup_count, _ = cleaner.check_duplicates(df)
        outlier_summary = cleaner.check_outliers(df)
        all_insights = cleaner.get_recommendations(missing_df, dup_count, outlier_summary) + insights.generate_insights(df, missing_df, corr_matrix)
        report_figures = []
        if len(numeric_cols) > 1:
            fig_curr = visualizer.plot_correlation_heatmap(corr_matrix)
            report_figures.append({'title': 'Correlation Heatmap', 'fig': fig_curr})
        for col in numeric_cols[:3]:
            fig = visualizer.plot_histograms(df, col)
            report_figures.append({'title': f'Histogram: {col}', 'fig': fig})
        if len(numeric_cols) >= 2:
             fig_scat = visualizer.plot_scatter(df, numeric_cols[0], numeric_cols[1])
             report_figures.append({'title': f'Scatter: {numeric_cols[0]} vs {numeric_cols[1]}', 'fig': fig_scat})
        html_report = reporter.export_to_html(dataset_info, all_insights, report_figures)
        st.download_button(
            label="Download HTML Report",
            data=html_report,
            file_name="DexAutoEDA_Report.html",
            mime="text/html"
        )
