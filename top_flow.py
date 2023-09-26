import papermill as pm
import os

# Define the top values
top_values = [250, 500]

# File paths
data_path = "/home/sayem/Desktop/Project/data/"

# Loop through each top value
for top in top_values:
    print(f"Running for top value: {top}")
    
    # Execute notebooks with papermill
    pm.execute_notebook(
       '02_sample_selection.ipynb',
       '/dev/null',  # This means no output notebook will be saved
       parameters={'top': top}
    )
    
    # Uncomment these when you're ready to run the other notebooks and Python files.
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
    
    # Execute Python scripts
    os.system(f"python notebook_05.py {top}")
    os.system(f"python notebook_06.py {top}")
    
    # Check for the existence of dataset.h5 and rename if it exists
    if os.path.exists(os.path.join(data_path, "dataset.h5")):
        os.rename(
            os.path.join(data_path, "dataset.h5"), 
            os.path.join(data_path, f"{top}_dataset.h5")
        )
    else:
        print(f"Warning: {os.path.join(data_path, 'dataset.h5')} does not exist and cannot be renamed.")

print("All tasks completed!")
