import mlflow
import papermill as pm
import pandas as pd
from pathlib import Path
import numpy as np
import gc

top = "250"  # default value, can be overwritten by Papermill

# Set MLflow experiment
mlflow.set_experiment("ML_Models_Experiments")

# Read the keys
DATA_STORE = Path(f'data/{top}_dataset.h5')
with pd.HDFStore(DATA_STORE) as store:
    keys = store.keys()

# Filter the keys
relevant_keys = [key for key in keys if key.startswith('/data/YEAR_')]


# Loop over the relevant keys and run the strategy building notebook
for key in relevant_keys:
    with mlflow.start_run():
        # Log the dataset key as a parameter
        mlflow.log_param("dataset_key", key)

        result = pm.execute_notebook(
            '07_a_strategy_building_(ML_models).ipynb',
            None,
            parameters={'dataset_key': key}
        )

        era_scores = result['papermill_era_scores']