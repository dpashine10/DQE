import pandas as pd
import numpy as np

def get_basic_info(df):
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return info

def get_summary_statistics(df):
    summary_stats = df.describe(include='all').to_dict()
    return summary_stats

def get_correlation_matrix(df):
    correlation_matrix = df.corr().to_dict()
    return correlation_matrix

def get_value_counts(df, column):
    if column in df.columns:
        value_counts = df[column].value_counts().to_dict()
        return value_counts
    else:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

def get_missing_value_percentage(df):
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    return missing_percentage.to_dict()

def get_unique_values(df, column):
    if column in df.columns:
        unique_values = df[column].unique().tolist()
        return unique_values
    else:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

