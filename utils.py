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
        df[col] = pd.to_numeric(df[col], downcast='float')
    
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
