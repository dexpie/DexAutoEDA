import pandas as pd
import numpy as np

def get_descriptive_stats(df):
    """
    Returns descriptive statistics for numeric columns.
    """
    return df.describe()

def get_numeric_distribution(df):
    """
    Returns Skewness and Kurtosis for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
        
    dist_df = pd.DataFrame({
        'Skewness': numeric_df.skew(),
        'Kurtosis': numeric_df.kurtosis()
    })
    return dist_df

def get_categorical_summary(df):
    """
    Returns value counts for categorical columns.
    
    Returns:
        dict: {column_name: value_counts_series}
    """
    cat_df = df.select_dtypes(include=['object', 'category'])
    summary = {}
    for col in cat_df.columns:
        summary[col] = df[col].value_counts().head(10) # Limit to top 10
    return summary

def get_correlation(df):
    """
    Returns correlation matrix for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr()

def get_time_series_summary(df, date_col, value_col, freq='M'):
    """
    Resamples time series and calculates rolling stats.
    freq: 'D', 'W', 'M', 'Y'
    """
    ts_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(ts_df[date_col]):
        return None
        
    ts_df = ts_df.set_index(date_col).sort_index()
    
    # Resample
    resampled = ts_df[value_col].resample(freq).mean()
    return resampled
