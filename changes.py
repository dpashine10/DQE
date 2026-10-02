import pandas as pd
import numpy as np

def fillna_foracolumn(df, column, value):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df_filled = df.copy()
    df_filled[column].fillna(value, inplace=True)
    
    return df_filled

def fillna_forallcols(df, value):
    df_filled = df.copy()
    df_filled.fillna(value, inplace=True)
    
    return df_filled

def dropna_foracolumn(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df_dropped = df.copy()
    df_dropped.dropna(subset=[column], inplace=True)
    
    return df_dropped

def dropna_forallcols(df):
    df_dropped = df.copy()
    df_dropped.dropna(inplace=True)
    
    return df_dropped

def replace_values(df, column, to_replace, value):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df_replaced = df.copy()
    df_replaced[column].replace(to_replace, value, inplace=True)
    
    return df_replaced

def drop_duplicates(df, subset=None):
    df_deduplicated = df.copy()
    df_deduplicated.drop_duplicates(subset=subset, inplace=True)
    
    return df_deduplicated

def rename_column(df, old_name, new_name):
    if old_name not in df.columns:
        raise ValueError(f"Column '{old_name}' does not exist in the DataFrame.")
    
    df_renamed = df.copy()
    df_renamed.rename(columns={old_name: new_name}, inplace=True)
    
    return df_renamed

def convert_dtype(df, column, new_dtype):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df_converted = df.copy()
    df_converted[column] = df_converted[column].astype(new_dtype)
    
    return df_converted

def sort_dataframe(df, by, ascending=True):
    if by not in df.columns:
        raise ValueError(f"Column '{by}' does not exist in the DataFrame.")
    
    df_sorted = df.copy()
    df_sorted.sort_values(by=by, ascending=ascending, inplace=True)
    
    return df_sorted

def filter_dataframe(df, condition):
    df_filtered = df.query(condition)
    return df_filtered

def groupby_aggregate(df, groupby_columns, agg_dict):
    for col in groupby_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    df_grouped = df.groupby(groupby_columns).agg(agg_dict).reset_index()
    return df_grouped

def pivot_table(df, index, columns, values, aggfunc='mean'):
    for col in [index, columns, values]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    pivot_df = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
    return pivot_df
