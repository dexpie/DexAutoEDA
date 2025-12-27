import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, classification_report

def detect_problem_type(df, target_col):
    """
    Detects if the problem is Classification or Regression based on target column.
    """
    if df[target_col].nunique() < 20 or df[target_col].dtype == 'object':
        return 'classification'
    else:
        return 'regression'

def train_baseline_model(df, target_col):
    """
    Trains a baseline Random Forest model and returns metrics & feature importance.
    
    Args:
        df: pandas DataFrame
        target_col: name of the target column
        
    Returns:
        dict: {
            'model_type': str,
            'metrics': dict,
            'feature_importance': DataFrame
        }
    """
    # 1. Prepare Data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Simple Preprocessing for Baseline (Handle Missing & Categorical)
    # Fill missing numeric with mean
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Label Encode Categorical (Simple for Tree Models)
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fill missing with 'Unknown' then encode
        X[col] = X[col].fillna('Unknown').astype(str)
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
        
    # Handle Target Missing
    if y.isnull().any():
        # Drop rows where target is missing
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
    # Detect Problem Type
    problem_type = detect_problem_type(df, target_col)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    if problem_type == 'classification':
        # If target is not numeric, encode it
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y_train = le_target.fit_transform(y_train)
            y_test = le_target.transform(y_test)
            
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, preds)
        }
    else:
        # Handle missing target (already dropped rows, but good check)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        metrics = {
            'R2 Score': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds)
        }
        
    # 4. Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return {
        'model_type': problem_type,
        'metrics': metrics,
        'feature_importance': feature_importance
    }
