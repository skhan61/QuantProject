import pandas as pd
import gc

def optimize_dataframe(df):
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # Convert bytes to MB
    
    # Remove columns with all NaN values
    df.dropna(axis=1, how='all', inplace=True)
    
    # Remove static columns (after removing NaN columns)
    cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Convert float64 to float32
    for col in df.select_dtypes(include=['float64']).columns:
        # Ensure the column values are within the range of float32 before downcasting
        max_val = df[col].max()
        min_val = df[col].min()
        if (-3.4e38 <= min_val <= 3.4e38) and (-3.4e38 <= max_val <= 3.4e38):
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            print(f"Column '{col}' not downcasted to float32 due to its range.")
    
    # Convert int64 to int32 (or smaller)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Fill NaN values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Remove rows that still have NaN values
    df.dropna(inplace=True)
    
    # Explicitly run the garbage collector
    gc.collect()
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2  # Convert bytes to MB
    
    print(f"Data memory before optimization: {initial_memory:.2f} MB")
    print(f"Data memory after optimization: {final_memory:.2f} MB")
    print(f"Reduced by: {100 * (initial_memory - final_memory) / initial_memory:.2f}%")
    
    return df

# df_optimized = optimize_dataframe(alphas_p.copy())

import pandas as pd

def CustomBackwardMultipleTimeSeriesCV(dataframe, train_period_length=63*3, \
    test_period_length=63, lookahead=1, date_idx='date'):
    """
    Yields train and test indices for time series cross-validation.
    Iterates over the data frame from the end to the beginning.
    
    Parameters:
        dataframe: The data to be split.
        train_period_length: The number of business days in the training set.
        test_period_length: The number of business days in the validation set.
        lookahead: The gap between training and validation sets.
        date_idx: The name of the date index in the dataframe.
    """
    
    unique_dates = dataframe.index.get_level_values(date_idx).unique()
    end_date_idx = len(unique_dates)
    
    while end_date_idx > 0:
        train_end = end_date_idx
        train_start = train_end - train_period_length
        
        test_end = train_start - lookahead
        test_start = max(test_end - test_period_length, 0)  # Ensure the index doesn't go negative
        
        if test_start == 0:
            break  # Break the loop if the test set would start at the beginning of the data
        
        # Get the train and test date ranges
        train_dates = unique_dates[train_start:train_end]
        test_dates = unique_dates[test_start:test_end]
        
        train_idx = dataframe.index.get_level_values(date_idx).isin(train_dates)
        test_idx = dataframe.index.get_level_values(date_idx).isin(test_dates)
        
        yield train_idx, test_idx
        
        end_date_idx = test_start - lookahead

import gc
import pandas as pd

def drop_na_cols(df):
    return df.dropna(axis=1, how='all')

def drop_static_cols(df):
    cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=cols_to_drop)

def downcast_numerics(df):
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    return df

def convert_objects_to_category(df):
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = df[col].nunique()
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    return df

def fill_and_drop_na(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df.dropna()

import gc
import pandas as pd

CHUNK_SIZE = 50000  # This can be adjusted based on your system's memory

def process_chunkwise(df, func):
    # Split dataframe into chunks
    chunks = [df.iloc[i:i + CHUNK_SIZE] for i in range(0, df.shape[0], CHUNK_SIZE)]
    
    processed_chunks = []
    for chunk in chunks:
        processed_chunks.append(func(chunk))
        gc.collect()
    
    return pd.concat(processed_chunks, axis=0)

def optimize_dataframe_new(df):
    df_copy = df.copy()
    
    df_copy = process_chunkwise(df_copy, drop_na_cols)
    df_copy = process_chunkwise(df_copy, drop_static_cols)
    df_copy = process_chunkwise(df_copy, downcast_numerics)
    df_copy = process_chunkwise(df_copy, convert_objects_to_category)
    df_copy = process_chunkwise(df_copy, fill_and_drop_na)

    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    final_memory = df_copy.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Data memory before optimization: {initial_memory:.2f} MB")
    print(f"Data memory after optimization: {final_memory:.2f} MB")
    print(f"Reduced by: {100 * (initial_memory - final_memory) / initial_memory:.2f}%")

    gc.collect()

    return df_copy


import sys

def clear_large_vars(threshold_size_in_MB=100):
    """
    Clears variables that are above the specified threshold size.
    
    Parameters:
    - threshold_size_in_MB (int): Size threshold in MB. Defaults to 100MB.
    """
    for var_name in list(globals().keys()):
        size = sys.getsizeof(globals()[var_name]) / (1024 ** 2)  # Convert to MB
        if size > threshold_size_in_MB:
            print(f"Clearing {var_name}, Size: {size:.2f}MB")
            del globals()[var_name]



import pandas as pd

def save_to_hdf(data, file_path, key_prefix):
    # print('=============Called===============')
    """
    Efficiently save data to an HDF5 file and return the key under which it's stored.
    
    Parameters:
    - data: The DataFrame you want to store.
    - file_path: The path of the HDF5 file.
    - key_prefix: Prefix for the key under which data will be stored.
    
    Returns:
    - key: The key under which the data is stored in the HDF5 file.
    """
    # Extract the date range from the 'date' level of the multi-index
    date_level_values = data.index.get_level_values('date')
    start_date = date_level_values.min().strftime('%Y%m%d')
    end_date = date_level_values.max().strftime('%Y%m%d')
    
    # Construct the key
    key = f"{key_prefix}_{start_date}_{end_date}"
    
    # Use compression for efficient storage and append mode to not overwrite existing data
    data.to_hdf(file_path, key=key, mode='a', \
        complib='blosc', complevel=9, format='table')
    # print(f'Data saved in {key}\n')
    
    return key




import pandas as pd
import numpy as np

def rank_and_quantize(df, TARGET_col='TARGET_ret_fwd_frac_order'):
    """
    Ranks stocks within each date based on the specified TARGET column and then bucket them into quantiles.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with MultiIndex (date, ticker).
        TARGET_col (str): Column name based on which the ranking needs to be done.
    
    Returns:
        pd.DataFrame: Dataframe with original column, an additional column for ranks, and quantized values.
    """
    rank_col_name = TARGET_col + '_rank'
    quant_col_name = TARGET_col + '_quantiled'
    
    # Ranking stocks within each date so that the highest value gets rank 1 (considered best)
    df[rank_col_name] = df.groupby('date')[TARGET_col].rank(method="average", ascending=False).astype(int) # Change to descending

    # Bucketing the ranks into quantiles such that rank 1 is in the uppermost bucket (labelled as 1.0)
    quantile_labels = [1.0, 0.75, 0.5, 0.25, 0.0]
    df[quant_col_name] = pd.qcut(df[rank_col_name], q=5, labels=quantile_labels).astype(float)
    
    # # Sorting by MultiIndex levels to preserve the original structure
    # df.sort_index(level=['date', 'ticker'], ascending=[True, True], inplace=True)
    
    return df
