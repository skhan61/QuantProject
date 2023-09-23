from concurrent.futures import ThreadPoolExecutor, as_completed
import papermill as pm
from datetime import datetime
from pathlib import Path
from filelock import FileLock
import dask.dataframe as dd
import pandas as pd

def run_notebooks(input_notebook):
    def run_notebook(start_date_str, end_date_str):
        print(f"Running notebook {input_notebook} for start_date: {start_date_str}, end_date: {end_date_str}")
        
        pm.execute_notebook(
            input_notebook,
            "/dev/null",  # Discard output
            parameters={
                'start_date': start_date_str,
                'end_date': end_date_str,
            },
            report_mode=True
        )
        
    # Define the path to your data
    DATA_STORE = Path('/home/sayem/Desktop/Project/data/assets.h5')
    lock_path = "/tmp/assets_h5_file.lock"  # Lock path

    # Lock the file and retrieve the data
    with FileLock(lock_path):
        alpha_101_data_full = dd.read_hdf(DATA_STORE, key='factors/alpha_101').compute()

    # Get the maximum and minimum dates from your data
    max_date = alpha_101_data_full.index.get_level_values('date').max()
    min_date = datetime.strptime('2013-01-01', '%Y-%m-%d')

    # Define the time delta: approximately 2 years in business days (252 business days in a year)
    delta = pd.DateOffset(days=504)

    date_ranges = []
    
    while max_date >= min_date:
        end_date_str = max_date.strftime('%Y-%m-%d')
        start_date_str = (max_date - delta).strftime('%Y-%m-%d')
        date_ranges.append((start_date_str, end_date_str))
        max_date -= delta

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(run_notebook, start_date, end_date): (start_date, end_date) for start_date, end_date in date_ranges}
        
        for future in as_completed(futures):
            start_date, end_date = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{input_notebook} generated an exception: {exc}")

if __name__ == "__main__":
    # run_notebooks("04_dataset_building.ipynb")
    run_notebooks("05_ic_based_feature_selection.ipynb")





# # Parallel Code
# from concurrent.futures import ThreadPoolExecutor
# import papermill as pm
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# import dask.dataframe as dd
# import pandas as pd

# from concurrent.futures import ThreadPoolExecutor, as_completed  # Added as_completed here

# def run_notebook(start_date_str, end_date_str):
#     print(f"Running notebook for start_date: \
#         {start_date_str}, end_date: {end_date_str}")

#     pm.execute_notebook(
#         input_notebook,
#         "/dev/null",  # Discard output
#         parameters={
#             'start_date': start_date_str,
#             'end_date': end_date_str,
#         },
#         report_mode=True
#     )

# # Define the notebook paths
# input_notebook = "04_dataset_building.ipynb"

# # Define the path to your data
# DATA_STORE = Path('/home/sayem/Desktop/Project/data/assets.h5')
# lock_path = "/tmp/assets_h5_file.lock"  # Lock path

# # Lock the file and retrieve the data
# with FileLock(lock_path):
#     alpha_101_data_full = dd.read_hdf(DATA_STORE, key='factors/alpha_101').compute()

# # Get the maximum and minimum dates from your data
# max_date = alpha_101_data_full.index.get_level_values('date').max()
# min_date = datetime.strptime('2013-01-01', '%Y-%m-%d')  # Fixed min date

# # Define the time delta: approximately 2 years in business days (252 business days in a year)
# delta = pd.DateOffset(days=504)

# date_ranges = []

# # Create a list of date ranges
# while max_date >= min_date:
#     end_date_str = max_date.strftime('%Y-%m-%d')
#     start_date_str = (max_date - delta).strftime('%Y-%m-%d')
#     date_ranges.append((start_date_str, end_date_str))

#     # Update max_date
#     max_date -= delta

# # # Run the notebooks in parallel using a thread pool
# # with ThreadPoolExecutor(max_workers=20) as executor:
# #     futures = {executor.submit(run_notebook, start_date, end_date): (start_date, end_date) for start_date, end_date in date_ranges}

# #     for future in concurrent.futures.as_completed(futures):
# #         start_date, end_date = futures[future]
# #         try:
# #             future.result()
# #         except Exception as exc:
# #             print(f"Notebook generated an exception: {exc}")

# # Run the notebooks in parallel using a thread pool
# with ThreadPoolExecutor(max_workers=20) as executor:
#     futures = {executor.submit(run_notebook, start_date, end_date): (start_date, end_date) for start_date, end_date in date_ranges}

#     for future in as_completed(futures):  # Changed this line
#         start_date, end_date = futures[future]
#         try:
#             future.result()
#         except Exception as exc:
#             print(f"Notebook generated an exception: {exc}")




# # Serial Code
# # import papermill as pm
# # from datetime import datetime, timedelta
# # from pathlib import Path
# # from filelock import FileLock
# # import dask.dataframe as dd
# # import pandas as pd

# # # Define the notebook paths
# # input_notebook = "04_dataset_building.ipynb"

# # # Define the path to your data
# # DATA_STORE = Path('/home/sayem/Desktop/Project/data/assets.h5')
# # lock_path = "/tmp/assets_h5_file.lock"  # Lock path

# # # Lock the file and retrieve the data
# # with FileLock(lock_path):
# #     # Use Dask to directly read the HDF5 files
# #     alpha_101_data_full = dd.read_hdf(DATA_STORE, key='factors/alpha_101').compute()

# # # Get the maximum and minimum dates from your data
# # max_date = alpha_101_data_full.index.get_level_values('date').max()
# # min_date = datetime.strptime('2013-01-01', '%Y-%m-%d')  # Fixed min date

# # # Define the time delta: approximately 2 years in business days (252 business days in a year)
# # delta = pd.DateOffset(days=504)

# # # Loop to execute the notebook with different parameters
# # while max_date >= min_date:
# #     end_date_str = max_date.strftime('%Y-%m-%d')
# #     start_date_str = (max_date - delta).strftime('%Y-%m-%d')

# #     print(f"Running notebook for start_date: {start_date_str}, end_date: {end_date_str}")

# #     pm.execute_notebook(
# #         input_notebook,
# #         "/dev/null",  # We don't need to save the output
# #         parameters={
# #             'start_date': start_date_str,
# #             'end_date': end_date_str,
# #         },
# #         report_mode=True  # This will avoid saving any output from the cells
# #     )

# #     # Update max_date
# #     max_date -= delta