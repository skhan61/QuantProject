import mlflow
import papermill as pm
import pandas as pd
from pathlib import Path
import scrapbook as sb
from io import StringIO

# Define list of top values
top_values = ["250", "500"]

def run_experiment(top, target):
    mlflow.set_experiment(f"LightGBM_Experiments_with_{top}_dataset")

    DATA_STORE = Path(f'data/{top}_dataset.h5')
    with pd.HDFStore(DATA_STORE) as store:
        keys = store.keys()

    relevant_keys = [key for key in keys if key.startswith('/data/YEAR_')]

    for key in relevant_keys:
        print(f'Running for top: {top} target: {target} key: {key}')
        with mlflow.start_run():
            mlflow.log_param("dataset_key", key)
            output_notebook_path = Path(f'./output_notebook/{top}{key}_07_a_ML _models_forecasting.ipynb')
            output_notebook_path.parent.mkdir(parents=True, exist_ok=True)

            pm.execute_notebook(
                '07_a_ML _models_forecasting.ipynb',
                str(output_notebook_path),
                parameters={'dataset_key': key, 'top': top, 'label': target}
            )

            notebook_data = sb.read_notebook(str(output_notebook_path))
            scraps = notebook_data.scraps

            def extract_scrap_data(scrap_name):
                scrap = scraps[scrap_name]
                if scrap.display is None:
                    return scrap.data
                return scrap.display['data']['text/plain'].strip("'")

            era_scores_string = extract_scrap_data('papermill_era_scores')
            era_scores_data = pd.read_csv(StringIO(era_scores_string), index_col=0)  # Convert back to DataFrame
            # plot_path_data = extract_scrap_data('papermill_plot_path')
            information_coefficient_data = float(extract_scrap_data('information_coefficient'))
            p_value_data = float(extract_scrap_data('p_value'))

            # # Log artifacts and metrics to MLflow
            # mlflow.log_artifact(plot_path_data)
            mlflow.log_metric("information_coefficient", information_coefficient_data)
            mlflow.log_metric("p_value", p_value_data)

            # Extract just the filename from the key
            key_name = Path(key).name

            # Save the era_scores_data DataFrame as a CSV in the era_scores folder and log it as an artifact
            era_scores_folder = Path("./fold_scores")
            era_scores_folder.mkdir(parents=True, exist_ok=True)
            era_scores_file = era_scores_folder / f"{key_name}_temp_fold_scores.csv"
            era_scores_data.to_csv(era_scores_file)
            mlflow.log_artifact(str(era_scores_file), era_scores_file.name)

# Iterate over different top and target values
for top in top_values:
    DATA_STORE = Path(f'data/{top}_dataset.h5')
    with pd.HDFStore(DATA_STORE) as store:
        # Assuming the dataset is a DataFrame stored under one of the keys
        first_key = [k for k in store.keys() if k.startswith('/data/YEAR_')][0]
        df = store[first_key]
        target_values = [col for col in df.columns if col.startswith('TARGET_ret_fwd_')]

    for target in target_values:
        try:
            # print(f'Running for top: {top} target: {target}')
            run_experiment(top, target)
        except Exception as e:
            print(f"Error encountered for top: {top}, target: {target}. Error: {str(e)}")
            # Optionally, you can log the exception to MLflow or another logging system
            continue  # Go to the next iteration