import os
import tables
import pandas as pd
import numpy as np
import dask.dataframe as dd
from joblib import Parallel, delayed
from numba import jit
from utils import optimize_dataframe, save_to_hdf

@jit(nopython=True)
def compute_correlation(data1, data2):
    n = len(data1)
    mean_x = np.mean(data1)
    mean_y = np.mean(data2)
    
    num = np.sum((data1 - mean_x) * (data2 - mean_y))
    den = np.sqrt(np.sum((data1 - mean_x)**2) * np.sum((data2 - mean_y)**2))
    
    if den == 0:
        return np.nan
    else:
        return num / den


def calculate_ic(dataframe, target_column, target_ranked, n_jobs=-1):
    features = [col for col in dataframe.columns.tolist() if col != target_column]
    correlations = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation)(dataframe[column].values, target_ranked) for column in features
    )
    ic_original = pd.Series(dict(zip(features, correlations))).sort_values(ascending=False)
    return ic_original


def calculate_ic_batched(dataframe, target_column, batch_size=50, corr_threshold=0.5, n_jobs=-1):
    df_ranked = dataframe.rank()
    target_ranked = df_ranked[target_column].values
    columns = [col for col in dataframe.columns.tolist() if col != target_column]

    ic_aggregated = pd.Series(dtype=float)
    num_batches = len(columns) // batch_size + 1
    for i in range(num_batches):
        start_col = i * batch_size
        end_col = start_col + batch_size

        subset_cols = columns[start_col:end_col]
        subset = df_ranked[subset_cols]
        ic_original = calculate_ic(subset, target_column, target_ranked, n_jobs=n_jobs)
        ic_aggregated = ic_aggregated.add(ic_original, fill_value=0)

    correlation_matrix = df_ranked[ic_aggregated.index].corr()
    dropped_features = set()
    for col in ic_aggregated.index:
        if col not in dropped_features:
            correlated_features = correlation_matrix[col][(correlation_matrix[col].abs() \
                > corr_threshold) & (correlation_matrix[col].index != col)].index
            for feature in correlated_features:
                if ic_aggregated[col] < ic_aggregated[feature]:
                    dropped_features.add(col)
                else:
                    dropped_features.add(feature)

    ic_reduced = ic_aggregated.drop(labels=dropped_features)
    columns_to_include = ic_reduced.index.tolist() + [target_column, 'open', 'high', 'low', 'volume', 'close']
    reduced_dataframe = dataframe[columns_to_include]
    selected_correlation_matrix = correlation_matrix.loc[ic_reduced.index, ic_reduced.index]
    
    return reduced_dataframe, ic_reduced, selected_correlation_matrix

def preprocess_dataframe(df):
    df = optimize_dataframe(df)
    df.drop_duplicates(inplace=True)
    return df.loc[:, ~df.columns.duplicated()]


def list_all_keys_in_h5(file_path):
    with pd.HDFStore(file_path) as store:
        keys = [k for k in store.keys() if k.startswith('/data/YEAR')]
    return keys

import tables

def find_key_for_date_range(start_date, end_date, file_path):
    pattern = f"/data/YEAR_{start_date.replace('-', '_')}_to_{end_date.replace('-', '_')}"
    return pattern if pattern in list_all_keys_in_h5(file_path) else None


def process_data_using_key(file_name, \
    file_path="/home/sayem/Desktop/Project/data/dataset.h5"):
    # Check if the key is valid
    if not file_name:
        print(f"Invalid key provided: {file_name}")
        return
    
    # Read the dataset using Dask
    data = dd.read_hdf(file_path, file_name)
    
    # Compute the result (this will load data into memory)
    result = data.compute()

    # Optimize memory and clean dataframe
    data = optimize_dataframe(result)

    # Drop duplicated rows and columns from the data
    data.drop_duplicates(inplace=True)
    data = data.loc[:, ~data.columns.duplicated()]

    del result

    # Check for the TARGET column name in a case-insensitive manner
    TARGET = 'RET_FWD_FRAC_ORDER'
    column_to_use = next((col for col in data.columns if col.lower() == TARGET.lower()), None)
    
    if not column_to_use:
        print(f"Column {TARGET} or its lowercase version not found in dataframe.")
        return

    reduced_dataframe, selected_ics, \
        selected_corr_matrix = calculate_ic_batched(data, column_to_use, batch_size=100)

    # Find and remove duplicate columns
    duplicated_cols = reduced_dataframe.columns[reduced_dataframe.columns.duplicated()].to_list()
    reduced_dataframe = reduced_dataframe.loc[:, ~reduced_dataframe.columns.duplicated()]
    print(f"Removed duplicate columns: {duplicated_cols}")
    print(reduced_dataframe.shape)

    del data

    # Save to HDF5
    file_name_prefix = f'/data/ic_based_reduced_features_YEAR'  # You might want to modify this if needed
    key = save_to_hdf(reduced_dataframe, file_path, file_name_prefix)
    print(f'Data saved in {key}\n')


import sys

def main_process():
    # The first argument (sys.argv[0]) is always the script name itself.
    top = int(sys.argv[1])  # Capture the 'top' value passed as the first argument.
    file_path = f"/home/sayem/Desktop/Project/data/{top}_dataset.h5"
    
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    year_keys = list_all_keys_in_h5(file_path)
    # print(year_keys)

    # Process each key
    for key in year_keys:
        print(f"Processing data for key: {key}")
        process_data_using_key(key, file_path)

if __name__ == "__main__":
    main_process()
