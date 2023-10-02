from pathlib import Path
import pandas as pd
from filelock import FileLock
import dask.dataframe as dd

import dask.dataframe as dd
from filelock import FileLock

def read_hdf_filtered_by_date(data_store, start_date, end_date, top=250):
    print("Reading data from HDF5...")
    lock_path = "/tmp/assets_h5_file.lock"
    with FileLock(lock_path):
        alpha_101_data = dd.read_hdf(data_store, key=f'factor/top{top}_dataset_alpha101').compute()
        ta_data = dd.read_hdf(data_store, key=f'factor/top{top}_dataset_with_TA').compute()
        beta_proxy_data = dd.read_hdf(data_store, key=f'factor/top{top}_dataset_with_rolling_beta_size_proxy').compute()

    print("Filtering data by date...")
    
    alpha_101_data = alpha_101_data.loc[
        (alpha_101_data.index.get_level_values("date") >= start_date) &
        (alpha_101_data.index.get_level_values("date") <= end_date)
    ]

    ta_data = ta_data.loc[
        (ta_data.index.get_level_values("date") >= start_date) &
        (ta_data.index.get_level_values("date") <= end_date)
    ]
    
    beta_proxy_data = beta_proxy_data.loc[
        (beta_proxy_data.index.get_level_values("date") >= start_date) &
        (beta_proxy_data.index.get_level_values("date") <= end_date)
    ]

    print("Data filtering completed.")
    return alpha_101_data, ta_data, beta_proxy_data
    


def process_and_merge(alpha_101_data, common_data, top, chunk_size=100000):
    print("Processing and merging data...")
    processed_data_list = []
    total_chunks = (common_data.shape[0] // chunk_size) + 1
    
    for idx, start in enumerate(range(0, common_data.shape[0], chunk_size)):
        print(f"Processing chunk {idx + 1} of {total_chunks}...")
        end = start + chunk_size
        common_data_chunk = common_data.iloc[start:end]
        
        tickers_in_chunk = common_data_chunk.index.get_level_values('ticker').unique()
        dates_in_chunk = common_data_chunk.index.get_level_values('date').unique()

        filtered_alpha_101_data = alpha_101_data[
            alpha_101_data.index.get_level_values('ticker').isin(tickers_in_chunk) &
            alpha_101_data.index.get_level_values('date').isin(dates_in_chunk)
        ]

        merged_chunk = common_data_chunk.merge(
            filtered_alpha_101_data, left_index=True, right_index=True, how='inner', suffixes=('', '_y')
        )
        
        # Drop columns with unwanted suffixes
        merged_chunk = merged_chunk.drop(columns=[col for col in merged_chunk if col.endswith(('_y', '_x'))])
        
        # Add "FEATURE_" prefix to columns that don't start with 'ret_fwd_'
        feature_cols = [col for col in merged_chunk.columns if not col.startswith('ret_fwd_')]
        rename_dict_feature = {col: f'FEATURE_{col}' for col in feature_cols}
        merged_chunk.rename(columns=rename_dict_feature, inplace=True)
        
        # Add "TARGET_" prefix to columns that start with 'ret_fwd_'
        target_cols = [col for col in merged_chunk.columns if col.startswith('ret_fwd_')]
        rename_dict_target = {col: f'TARGET_{col}' for col in target_cols}
        merged_chunk.rename(columns=rename_dict_target, inplace=True)
        
        processed_data_list.append(merged_chunk)
    
    final_data = pd.concat(processed_data_list)
    print("Processing and merging completed.")
    return final_data


import pandas as pd

def save_to_hdf(data, file_path, key_prefix):
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
    
    return key


def main(data_store, file_path, start_date, end_date, top):
    alpha_101_data, ta_data, beta_proxy_data \
        = read_hdf_filtered_by_date(data_store, start_date, end_date, top)
    common_data = ta_data
    final_data = process_and_merge(alpha_101_data, common_data, top)
    
    # Swap levels if needed
    if final_data.index.names[0] == 'ticker':
        final_data = final_data.swaplevel('ticker', 'date')

    # Sort the dataset by date and then ticker
    final_data.sort_index(level=['date', 'ticker'], ascending=[True, True], inplace=True)
    
        # Check and localize the datetime level of the MultiIndex to UTC if needed
    datetime_level = 0  # Assuming the datetime is the first level
    if final_data.index.levels[datetime_level].tz is None:
        localized_level = final_data.index.levels[datetime_level].tz_localize('UTC')
        final_data.index = final_data.index.set_levels(localized_level, level=datetime_level)
        
        
    # print(final_data.head())

    KEY_NAME_PREFIX = 'data/YEAR'
    key = save_to_hdf(final_data, file_path, KEY_NAME_PREFIX)

    print(f"Shape of the final combined data: {final_data.shape}")
    print("Processing completed.")
    print(f'key is: {key}')


def get_max_date_from_store(path):
    """Load a small chunk of data and retrieve the max date."""
    with pd.HDFStore(path) as store:
        # Load a small chunk
        chunk = store.select('/stooq/us/nyse/stocks/prices', start=-10)  # Last 10 records
        max_date = chunk.index.get_level_values(1).max()
    return max_date


import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the 'top' value as an argument.")
        sys.exit(1)

    top = int(sys.argv[1])
    DATA_STORE = Path('/home/sayem/Desktop/Project/data/assets.h5')
    
    # Get the maximum date from DATA_STORE
    max_date_in_store = get_max_date_from_store(DATA_STORE)

    # For the unseen dataset
    FINAL_UNSEEN = max_date_in_store.strftime('%Y-%m-%d')
    INITIAL_UNSEEN = (max_date_in_store - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    FILE_PATH_UNSEEN = f"/home/sayem/Desktop/Project/data/{top}_unseen_dataset.h5"

    business_days_offset = pd.tseries.offsets.BDay(2 * 252)

    current_start_date = pd.Timestamp(INITIAL_UNSEEN)
    current_end_date = pd.Timestamp(FINAL_UNSEEN)
    
    while current_end_date <= pd.Timestamp(FINAL_UNSEEN):
        main(DATA_STORE, FILE_PATH_UNSEEN, current_start_date, current_end_date, top)
        
        # Update dates for next iteration
        current_start_date = current_end_date + pd.tseries.offsets.BDay(1)
        current_end_date = current_start_date + business_days_offset

    # For the regular dataset
    # Initialize INITIAL to the start date of your dataset (assuming it's the minimum date in your dataset)
    INITIAL = pd.Timestamp('1970-01-01')  # This is just a placeholder. Replace with your actual start date.
    FINAL = INITIAL_UNSEEN  # Start the regular dataset just before the unseen dataset
    FILE_PATH = f"/home/sayem/Desktop/Project/data/{top}_dataset.h5"

    current_start_date = pd.Timestamp(INITIAL)
    current_end_date = current_start_date + business_days_offset

    while current_end_date < pd.Timestamp(FINAL_UNSEEN):
        main(DATA_STORE, FILE_PATH, current_start_date, current_end_date, top)
        
        # Update dates for next iteration
        current_start_date = current_end_date + pd.tseries.offsets.BDay(1)
        current_end_date = current_start_date + business_days_offset

    # Process the remainder if any
    if current_start_date < pd.Timestamp(FINAL_UNSEEN):
        main(DATA_STORE, FILE_PATH, current_start_date, pd.Timestamp(FINAL_UNSEEN) - pd.tseries.offsets.BDay(1), top)



# import sys

# if __name__ == "__main__":
#     # The first argument (sys.argv[0]) is always the script name itself.
#     if len(sys.argv) < 2:
#         print("Please provide the 'top' value as an argument.")
#         sys.exit(1)

#     top = int(sys.argv[1])
    
#     DATA_STORE = Path('/home/sayem/Desktop/Project/data/assets.h5')
#     FILE_PATH = f"/home/sayem/Desktop/Project/data/{top}_unseen_dataset.h5"

#     FINAL = '2023-08-11'
#     INITIAL = '2022-08-12'

#     FINAL_END_DATE = pd.Timestamp(FINAL)
#     INITIAL_START_DATE = pd.Timestamp(INITIAL)
#     business_days_offset = pd.tseries.offsets.BDay(2 * 252)  # Approximation for 2 business years

#     current_start_date = INITIAL_START_DATE
#     current_end_date = current_start_date + business_days_offset

#     while current_end_date <= FINAL_END_DATE:
#         main(DATA_STORE, FILE_PATH, current_start_date, current_end_date, top)
        
#         # Update dates for next iteration
#         current_start_date = current_end_date + pd.tseries.offsets.BDay(1)  # Start the next day
#         current_end_date = current_start_date + business_days_offset

#     # Process the remainder if any
#     if current_start_date < FINAL_END_DATE:
#         main(DATA_STORE, FILE_PATH, current_start_date, FINAL_END_DATE, top)



