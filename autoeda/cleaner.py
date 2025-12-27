import pandas as pd
import numpy as np

def check_missing(df):
    """
    Checks for missing values in the dataframe.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return pd.DataFrame()
    
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': (missing / len(df)) * 100
    })
    return missing_df.sort_values(by='Percentage', ascending=False)

def check_duplicates(df):
    """
    Checks for duplicate rows.
    """
    duplicates = df[df.duplicated()]
    return len(duplicates), duplicates

def check_outliers(df):
    """
    Detects outliers in numeric columns using IQR method.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            outlier_summary[col] = len(outliers)
            
    return outlier_summary

def get_recommendations(missing_df, duplicates_count, outlier_summary):
    """
    Generates text recommendations.
    """
    recommendations = []
    
    if not missing_df.empty:
        for index, row in missing_df.iterrows():
            if row['Percentage'] > 50:
                recommendations.append(f"Column '{index}' has >50% missing values. Consider dropping it.")
            else:
                recommendations.append(f"Column '{index}' has missing values. Consider imputation.")
    
    if duplicates_count > 0:
        recommendations.append(f"Dataset has {duplicates_count} duplicate rows. Consider removing them.")
        
    if outlier_summary:
        cols_with_many_outliers = [k for k, v in outlier_summary.items() if v > 0]
        if cols_with_many_outliers:
            recommendations.append(f"Outliers detected in columns: {', '.join(cols_with_many_outliers)}. Check distribution.")
            
    if not recommendations:
        recommendations.append("Data looks clean! No critical issues detected.")
        
    return recommendations

# --- Actionable Functions ---

def drop_duplicates(df):
    """
    Removes duplicate rows.
    """
    return df.drop_duplicates()

def drop_columns(df, columns):
    """
    Drops specified columns.
    """
    return df.drop(columns=columns, errors='ignore')

def impute_missing(df, strategy='mean'):
    """
    Imputes missing values in numeric columns.
    Strategy: 'mean', 'median', 'mode' (zero for mode fallback)
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if strategy == 'mean':
                val = df_clean[col].mean()
            elif strategy == 'median':
                val = df_clean[col].median()
            else:
                val = 0 # Default fallback
            df_clean[col].fillna(val, inplace=True)
            
    # For categorical, fill with 'Missing'
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Missing', inplace=True)
            

# --- Feature Engineering ---

def encode_categorical(df, columns, method='onehot'):
    """
    Encodes categorical columns.
    method: 'onehot' or 'label'
    """
    df_enc = df.copy()
    if method == 'onehot':
        df_enc = pd.get_dummies(df_enc, columns=columns, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            
    return df_enc

def scale_features(df, columns, method='standard'):
    """
    Scales numeric features.
    method: 'standard' or 'minmax'
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled
