import plotly.express as px
import pandas as pd
import numpy as np

def plot_histograms(df, column):
    """
    Plots interactive histogram for a numeric column using Plotly.
    """
    fig = px.histogram(df, x=column, title=f'Distribution of {column}', marginal="box")
    return fig

def plot_boxplot(df, column):
    """
    Plots interactive boxplot for a numeric column using Plotly.
    """
    fig = px.box(df, y=column, title=f'Boxplot of {column}')
    return fig

def plot_bar_chart(df, column):
    """
    Plots interactive bar chart for a categorical column (Top 10) using Plotly.
    """
    # Get top 10 counts
    counts = df[column].value_counts().head(10).reset_index()
    counts.columns = [column, 'Count']
    
    fig = px.bar(counts, x=column, y='Count', title=f'Top 10 Categories in {column}', color='Count')
    return fig

def plot_correlation_heatmap(corr_matrix):
    """
    Plots interactive correlation heatmap using Plotly.
    """
    if corr_matrix.empty:
        return None
        
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap', color_continuous_scale='RdBu_r')
    return fig

def plot_scatter(df, x_col, y_col, color_col=None):
    """
    Plots interactive scatter plot.
    """
    # Check if color column exists and is valid
    color = color_col if color_col in df.columns else None
    
    fig = px.scatter(df, x=x_col, y=y_col, color=color, title=f'Scatter Plot: {x_col} vs {y_col}')
    return fig

def plot_feature_importance(importance_df):
    """
    Plots feature importance bar chart.
    """
    fig = px.bar(importance_df.head(15), x='Importance', y='Feature', orientation='h', 
                 title='Top 15 Feature Importance', color='Importance')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig
