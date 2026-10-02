import pandas as pd
import numpy as np

def mean(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    mean_value = df[column].mean()
    return mean_value

def mode(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    mode_value = df[column].mode()
    return mode_value.tolist()  # Return as a list to handle multiple modes

def median(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    median_value = df[column].median()
    return median_value

def std(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    std_value = df[column].std()
    return std_value

def variance(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    variance_value = df[column].var()
    return variance_value