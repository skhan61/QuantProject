import papermill as pm
import os
import logging
import warnings
import sys

# Set up logging
logging.basicConfig(level=logging.WARNING)

# Function to capture warnings
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    logging.warning(f"Warning captured: {message}")

warnings.showwarning = warn_with_traceback
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Add custom filter to ignore specific messages
warnings.filterwarnings("always", category=DeprecationWarning, module=".*traitlets.*")
warnings.filterwarnings("always", category=UserWarning, module=".*debugpy.*")
warnings.filterwarnings(action="ignore", category=UserWarning, module=".*debugpy.*", lineno=0)
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=".*traitlets.*", lineno=0)

# # Define the top values
# top_values = [250, 500]
top_values = [500]
# File paths
data_path = "/home/sayem/Desktop/Project/data/"

# Loop through each top value
for top in top_values:
    print(f"=== Running for top value: {top} ===")
    
    # Execute notebooks with papermill
    try:
        pm.execute_notebook(
           '02_sample_selection.ipynb',
           '/dev/null',
           parameters={'top': top}
        )
        
        pm.execute_notebook(
           '03_common_alpha_factors.ipynb',
           '/dev/null',
           parameters={'top': top}
        )

        pm.execute_notebook(
           '04_101_formulaic_alphas.ipynb',
           '/dev/null',
           parameters={'top': top}
        )
        
    except pm.PapermillExecutionError as e:
        logging.error(f"Error while executing notebook: {str(e)}")
    
    # Execute Python scripts
    result_05 = os.system(f"python notebook_05.py {top}")
    if result_05 != 0:
        logging.error(f"Error while executing notebook_05.py with top value {top}")


    # Execute notebooks with papermill
    # try:

    #     pm.execute_notebook(
    #        '06_ic_based_feature_selection.ipynb',
    #        '/dev/null',
    #        parameters={'top': top}
    #     )
        
    # except pm.PapermillExecutionError as e:
    #     logging.error(f"Error while executing notebook: {str(e)}")
        
    result_06 = os.system(f"python notebook_06.py {top}")
    if result_06 != 0:
        logging.error(f"Error while executing notebook_06.py with top value {top}")

print("All tasks completed!")


    # # Check for the existence of dataset.h5 and rename if it exists
    # if os.path.exists(os.path.join(data_path, "dataset.h5")):
    #     os.rename(
    #         os.path.join(data_path, "dataset.h5"), 
    #         os.path.join(data_path, f"{top}_dataset.h5")
    #     )
    # else:
    #     logging.warning(f"Warning: {os.path.join(data_path, 'dataset.h5')} does not exist and cannot be renamed.")

# print("All tasks completed!")



