import pandas as pd
import numpy as np
import dask.dataframe as dd  # Correct import for Dask's DataFrame module
from joblib import Parallel, delayed
from numba import jit
import sys

from utils import optimize_dataframe, save_to_hdf

@jit(nopython=True)
def compute_correlation(data1, data2):
    mean_x = np.mean(data1)
    mean_y = np.mean(data2)
    num = np.sum((data1 - mean_x) * (data2 - mean_y))
    den = np.sqrt(np.sum((data1 - mean_x) ** 2) * np.sum((data2 - mean_y) ** 2))
    
    return num / den if den != 0 else np.nan


# def calculate_ic(dataframe, target_column, target_ranked, n_jobs=-1):
#     features = [col for col in dataframe.columns if col != target_column]
#     correlations = Parallel(n_jobs=n_jobs)(
#         delayed(compute_correlation)(dataframe[column].values, target_ranked) for column in features
#     )
#     return pd.Series(dict(zip(features, correlations))).sort_values(ascending=False)

def calculate_ic(dataframe, target_column, target_ranked, n_jobs=-1):
    # Exclude the target column from the feature list
    features = [col for col in dataframe.columns.tolist() if col != target_column]
    correlations = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation)(dataframe[column].values, target_ranked) for column in features
    )
    
    # Explicitly specifying dtype=float64 for the Series
    return pd.Series(dict(zip(features, correlations)), dtype=float).sort_values(ascending=False)



def calculate_ic_batched(dataframe, target_column, batch_size=50, corr_threshold=0.5, n_jobs=-1):
    df_ranked = dataframe.rank()
    target_ranked = df_ranked[target_column].values
    columns = [col for col in dataframe.columns if col != target_column]
    
    ic_aggregated = pd.Series(dtype=float)

    num_batches = len(columns) // batch_size + 1
    for batch_no in range(num_batches):
        print(f"Processing batch {batch_no + 1}/{num_batches}...")
        subset_cols = columns[batch_no * batch_size: (batch_no + 1) * batch_size]
        ic_original = calculate_ic(df_ranked[subset_cols], target_column, target_ranked, n_jobs=n_jobs)
        ic_aggregated = ic_aggregated.add(ic_original, fill_value=0)
    
    correlation_matrix = df_ranked[ic_aggregated.index].corr()
    dropped_features = set()
    for col in ic_aggregated.index:
        if col not in dropped_features:
            correlated_features = correlation_matrix[col][(correlation_matrix[col].abs() > corr_threshold) & (correlation_matrix[col].index != col)].index
            for feature in correlated_features:
                dropped_features.add(col if ic_aggregated[col] < ic_aggregated[feature] else feature)

    ic_reduced = ic_aggregated.drop(labels=dropped_features)
    columns_to_include = ic_reduced.index.tolist() + [target_column] + [f'FEATURE_{attr}' for attr in ['open', 'high', 'low', 'volume', 'close']]
    
    return dataframe[columns_to_include], ic_reduced, correlation_matrix.loc[ic_reduced.index, ic_reduced.index]


def preprocess_dataframe(df):
    df = optimize_dataframe(df)
    df.drop_duplicates(inplace=True)
    return df.loc[:, ~df.columns.duplicated()]


def process_data_using_key(file_name, TARGET, file_path):
    data = dd.read_hdf(file_path, file_name)
    result = data.compute()
    data = preprocess_dataframe(result)
    del result

    column_to_use = next((col for col in data.columns if col.lower() == TARGET.lower()), None)

    if not column_to_use:
        print(f"Column {TARGET} or its lowercase version not found in dataframe.")
        return

    reduced_dataframe, selected_ics, selected_corr_matrix = calculate_ic_batched(data, column_to_use, batch_size=100)
    
    duplicated_cols = reduced_dataframe.columns[reduced_dataframe.columns.duplicated()].tolist()
    reduced_dataframe = reduced_dataframe.loc[:, ~reduced_dataframe.columns.duplicated()]

    print(f"Removed duplicate columns: {duplicated_cols}")
    print(reduced_dataframe.shape)

    del data

    file_name_prefix = f'/data/ic_based_reduced_features_YEAR'
    key = save_to_hdf(reduced_dataframe, file_path, file_name_prefix)
    print(f'Data saved in {key}\n')


def list_all_keys_in_h5(file_path):
    with pd.HDFStore(file_path) as store:
        keys = [k for k in store.keys() if k.startswith('/data/YEAR')]
    return keys


if __name__ == "__main__":
    top = int(sys.argv[1])
    file_path = f"/home/sayem/Desktop/Project/data/{top}_dataset.h5"

    year_keys = list_all_keys_in_h5(file_path)
    TARGET = 'TARGET_ret_fwd_frac_order'

    for key in year_keys:
        print(f"Processing data for key: {key}")
        process_data_using_key(key, TARGET, file_path)
