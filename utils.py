import pandas as pd
import gc

def optimize_dataframe(df):
    # Convert float64 to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Explicitly run the garbage collector
    gc.collect()
        
    return df