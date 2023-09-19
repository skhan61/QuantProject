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


import pandas as pd
import gc

def optimize_dataframe_new(df):
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # Convert bytes to MB
    
    # Remove columns with all NaN values
    df.dropna(axis=1, how='all', inplace=True)
    
    # Remove static columns (after removing NaN columns)
    cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Convert float64 to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert int64 to appropriate smaller types
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    # Convert object types to category if number of unique values is less than 50% of total values
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = df[col].nunique()
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    
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

# df_optimized = optimize_dataframe(your_dataframe.copy())
