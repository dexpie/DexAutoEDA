import pandas as pd
import io

def load_data(file):
    """
    Loads data from a CSV file.
    
    Args:
        file: file-like object or path to CSV file.
    
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        return None

def get_dataset_info(df):
    """
    Returns basic information about the dataset.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: containing shape, columns, and dtypes
    """
    if df is None:
        return {}
        
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_list': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'info_str': info_str
    }

def convert_to_datetime(df, col):
    """
    Converts a column to datetime objects.
    """
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception:
        return df
