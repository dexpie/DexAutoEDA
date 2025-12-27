import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error

# Import XGBoost and LightGBM gracefully
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier, LGBMRegressor = None, None

def detect_problem_type(df, target_col):
    """
    Detects if the problem is Classification or Regression based on target column.
    """
    if df[target_col].nunique() < 20 or df[target_col].dtype == 'object':
        return 'classification'
    else:
        return 'regression'

def train_models(df, target_col, selected_models=None):
    """
    Trains selected models and returns results for comparison.
    
    Args:
        df: pandas DataFrame
        target_col: name of the target column
        selected_models: list of model names ['Random Forest', 'XGBoost', 'LightGBM', 'Linear/Logistic']
        
    Returns:
        dict: {
            'problem_type': str,
            'results': list of dicts (metrics),
            'best_model_name': str,
            'best_model': object,
            'feature_importance': DataFrame (from best model if applicable)
        }
    """
    if selected_models is None:
        selected_models = ['Random Forest']
        
    # 1. Prepare Data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Simple Preprocessing
    # Fill missing numeric
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    
    # Encode Categorical
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown').astype(str)
        X[col] = le.fit_transform(X[col])
        
    # Handle Target Missing
    if y.isnull().any():
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
    # Detect Problem Type
    problem_type = detect_problem_type(df, target_col)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    models_obj = {}
    best_score = -float('inf')
    best_model_name = ""
    
    # Label Encode Target for Classification if needed
    if problem_type == 'classification' and y.dtype == 'object':
        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train)
        y_test = le_target.transform(y_test)

    # 3. Model Training Loop
    for model_name in selected_models:
        model = None
        
        if model_name == 'Random Forest':
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
        elif model_name == 'XGBoost':
            if XGBClassifier:
                if problem_type == 'classification':
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                else:
                    model = XGBRegressor(random_state=42)
            else:
                continue # Skip if not installed

        elif model_name == 'LightGBM':
            if LGBMClassifier:
                if problem_type == 'classification':
                    model = LGBMClassifier(random_state=42, verbose=-1)
                else:
                    model = LGBMRegressor(random_state=42, verbose=-1)
            else:
                continue

        elif model_name == 'Linear/Logistic':
            if problem_type == 'classification':
                model = LogisticRegression(max_iter=1000)
            else:
                model = LinearRegression()
                
        if model:
            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Evaluate
            if problem_type == 'classification':
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                
                res = {'Model': model_name, 'Accuracy': acc, 'F1 Score': f1}
                score = acc # Use Accuracy for selection
                
            else: # Regression
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                
                res = {'Model': model_name, 'R2 Score': r2, 'MAE': mae, 'MSE': mse}
                score = r2 # Use R2 for selection
            
            results.append(res)
            models_obj[model_name] = model
            
            if score > best_score:
                best_score = score
                best_model_name = model_name

    # 4. Feature Importance (Best Model)
    feature_importance = pd.DataFrame()
    best_model = models_obj.get(best_model_name)
    
    if best_model and hasattr(best_model, 'feature_importances_'):
         feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    elif best_model and model_name == 'Linear/Logistic':
         # For linear models, coefficients
         if problem_type == 'regression':
             imps = np.abs(best_model.coef_)
         else:
             imps = np.abs(best_model.coef_[0])
             
         feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': imps
        }).sort_values(by='Importance', ascending=False)
        
    return {
        'problem_type': problem_type,
        'results': results,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'feature_importance': feature_importance
    }

def save_model(model):
    """
    Serializes model to pickle.
    """
    return pickle.dumps(model)
