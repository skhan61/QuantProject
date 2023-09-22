from pathlib import Path
import pandas as pd
from filelock import FileLock
import dask.dataframe as dd

def read_hdf_filtered_by_date(data_store, start_date, end_date):
    print("Reading data from HDF5...")
    lock_path = "/tmp/assets_h5_file.lock"
    
    with FileLock(lock_path):
        alpha_101_data = dd.read_hdf(data_store, key='factors/alpha_101').compute()
        common_data = dd.read_hdf(data_store, key='factors/common').compute()

    print("Filtering data by date...")
    alpha_101_data = alpha_101_data.loc[
        (alpha_101_data.index.get_level_values("date") >= start_date) &
        (alpha_101_data.index.get_level_values("date") <= end_date)
    ]

    common_data = common_data.loc[
        (common_data.index.get_level_values("date") >= start_date) &
        (common_data.index.get_level_values("date") <= end_date)
    ]

    print("Data filtering completed.")
    return alpha_101_data, common_data

def process_and_merge(alpha_101_data, common_data, chunk_size=100000):
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
        merged_chunk = merged_chunk.drop(columns=[col for col in merged_chunk if col.endswith(('_y', '_x'))])
        
        processed_data_list.append(merged_chunk)
    
    final_data = pd.concat(processed_data_list)
    print("Processing and merging completed.")
    return final_data

# from utils import save_to_hdf

def main(data_store, file_path, start_date, end_date):
    alpha_101_data, common_data = read_hdf_filtered_by_date(data_store, start_date, end_date)
    final_data = process_and_merge(alpha_101_data, common_data)
    
    KEY_NAME_PREFIX = 'data/YEAR'
    key = save_to_hdf(final_data, file_path, KEY_NAME_PREFIX)

    print(f"Shape of the final combined data: {final_data.shape}")
    print("Processing completed.")
    print(f'key is: {key}')

if __name__ == "__main__":
    from utils import save_to_hdf
    DATA_STORE = Path('/home/sayem/Desktop/Project/data/assets.h5')
    FILE_PATH = "/home/sayem/Desktop/Project/data/dataset.h5"
    
    FINAL_END_DATE = pd.Timestamp('2023-08-11')
    INITIAL_START_DATE = pd.Timestamp('2013-01-01')
    business_days_offset = pd.tseries.offsets.BDay(2 * 252)  # Approximation for 2 business years

    current_start_date = INITIAL_START_DATE
    current_end_date = current_start_date + business_days_offset

    while current_end_date <= FINAL_END_DATE:
        main(DATA_STORE, FILE_PATH, current_start_date, current_end_date)
        
        # Update dates for next iteration
        current_start_date = current_end_date + pd.tseries.offsets.BDay(1)  # Start the next day
        current_end_date = current_start_date + business_days_offset

    # Process the remainder if any
    if current_start_date < FINAL_END_DATE:
        main(DATA_STORE, FILE_PATH, current_start_date, FINAL_END_DATE)