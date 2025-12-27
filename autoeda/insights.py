def generate_insights(df, missing_df, correlation_matrix):
    """
    Generates text-based insights based on data analysis.
    
    Args:
        df: pandas DataFrame
        missing_df: DataFrame summary of missing values
        correlation_matrix: DataFrame correlation matrix
        
    Returns:
        list: List of insight strings
    """
    insights = []
    
    # Dataset Shape
    insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Missing Values
    if not missing_df.empty:
        high_missing = missing_df[missing_df['Percentage'] > 30]
        if not high_missing.empty:
            cols = ", ".join(high_missing.index.tolist())
            insights.append(f"High missing values (>30%) detected in: {cols}.")
    else:
        insights.append("No missing values detected.")
        
    # High Correlation
    if not correlation_matrix.empty:
        # Create a boolean mask for the upper triangle
        import numpy as np
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.8 or < -0.8
        high_corr = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.8)]
        
        for col in high_corr:
            # Find the index (row) that correlates with this column
            correlated_rows = upper_tri.index[upper_tri[col].abs() > 0.8].tolist()
            for row in correlated_rows:
                val = upper_tri.loc[row, col]
                insights.append(f"Strong correlation ({val:.2f}) between '{row}' and '{col}'.")
                
    return insights
