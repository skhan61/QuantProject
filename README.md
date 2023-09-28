# Trading Strategy Development: An ML-Driven Approach

Welcome to the Trading Strategy Project. This repository showcases an end-to-end Machine Learning and Deep Learning driven approach for crafting sophisticated trading strategies. By leveraging the power of data science and advanced algorithms, we guide you through the intricate pipeline, starting from initial data processing all the way to comprehensive strategy backtesting. Dive in and experience the modern evolution of trading techniques.

Data Processing ---> Sample Selection ---> Factor Research ---> Feature Engineering ---> Dataset Building ---> ML/DL Model Training ---> Strategy Backtesting ---> Deployment

## Directory Structure

### Data
Contains raw OHLCV data and processed datasets used across the notebooks.

### Codes and Notebooks (in root directory)

1. **01_raw_data_to_create_dataset.ipynb**: Processing raw data to create a structured dataset.
2. **02_sample_selection.ipynb**: Techniques and logic behind selecting the right data samples.
3. **03_common_alpha_factors.ipynb**: Investigating common alpha factors in finance.
4. **04_101_formulaic_alphas.ipynb**: Deriving alphas using predefined formulae.
5. **05_dataset_building.ipynb**: Crafting the dataset suitable for ML modeling.
6. **06_ic_based_feature_selection.ipynb**: Feature selection based on Information Coefficient (IC).
7. **07_a_ML_models_forecasting.ipynb**: Training and evaluating machine learning forecasting models.
8. **07_b_DL_models_forecasting.ipynb**: Delving deep with deep learning models for forecasting.
9. **07_c_DL_models_generative.ipynb**: Generative models using deep learning.
10. **08_strategy_building_and_backtesting.ipynb**: The culmination - building and backtesting the trading strategy.
11. **formulaic_alphas.py**: Python script with formulae for deriving alphas.
12. **notebook_05.py, notebook_06.py**: Python scripts corresponding to specific notebooks.
13. **top_mlflow_experiment.py**: MLflow experiment tracking script.
14. **top_stock_dataset_pipeline.py**: Data pipeline script for stock datasets.
15. **utils.py**: Utility functions and common operations used across the project.
16. **auto_commit.sh, push_to_dockerhub.sh**: Automation scripts for repository management.

### Old (Archived)

Contains older versions or deprecated files.

## .gitignore Rules

The repository follows specific gitignore rules:

```plaintext
# Ignore all files by default
*
*_creds.txt
./output_notebook/**

# Whitelist only .py and .ipynb files, and other specified files/folders
!*.py
!*.ipynb
!.gitignore
!*.sh
!./Old/
!./Old/**
