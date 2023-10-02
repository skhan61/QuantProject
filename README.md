[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/skhan61/QuantProject.git/main)

# Trading Strategy Development: An ML-Driven Approach

Welcome to the Trading Strategy Project. This repository unfolds a comprehensive end-to-end Machine Learning and Deep Learning framework dedicated to the creation of intricate trading strategies. With the combined prowess of data science and innovative algorithms, we walk you through a detailed pipeline that spans from the initial stages of data processing to the meticulous act of strategy backtesting. Explore and immerse yourself in the contemporary advances of trading methodologies.

## Project Pipeline:

`Data Processing` ---> `Sample Selection` ---> `Factor Research` ---> `Feature Engineering & Dataset Building` ---> `ML/DL Model Training` ---> `Strategy Backtesting` ---> `Deployment`

## Codes and Notebooks:

Contained within each stage of the pipeline are meticulously crafted notebooks that delve into the details of each step. A breakdown is provided below:

### Data Processing

- [01_raw_data_to_create_dataset.ipynb](01_raw_data_to_create_dataset.ipynb): Transforming raw data into a cohesive dataset structure.

### Sample Selection

- [02_sample_selection.ipynb](02_sample_selection.ipynb): Insights and techniques for optimal data sample selection.

### Factor Research and Feature Engineering

- [03_common_alpha_factors.ipynb](03_common_alpha_factors.ipynb): An exploration into the prevalent alpha factors in finance.
- [04_101_formulaic_alphas.ipynb](04_101_formulaic_alphas.ipynb): Constructing alphas from predefined formulae.

### Dataset Building and Pre-modeling Feature Selection

- [05_dataset_building.ipynb](05_dataset_building.ipynb): Streamlining the dataset for machine learning applications.
- [06_ic_based_feature_selection.ipynb](06_ic_based_feature_selection.ipynb): Harnessing the Information Coefficient (IC) for effective feature selection.

### ML/DL Model Training

- [07_a_ML_models_forecasting.ipynb](07_a_ML_models_forecasting.ipynb): Diving into machine learning models for precise forecasting.
- [07_b_DL_models_forecasting.ipynb](07_b_DL_models_forecasting.ipynb): A deep dive into the world of deep learning forecasting models.
- [07_c_DL_models_generative.ipynb](07_c_DL_models_generative.ipynb): Leveraging deep learning for generative model constructs.

### Strategy Backtesting

- [08_strategy_building_and_backtesting.ipynb](08_strategy_building_and_backtesting.ipynb): The apex - constructing and assessing the trading strategy.

## Strategy Results:
```
                    Strategy
------------------  ----------
Start Period        2022-11-01
End Period          2023-08-03
Risk-Free Rate      0.0%
Time in Market      89.0%

Cumulative Return   12.72%
CAGR﹪              11.6%

Sharpe              1.18
Prob. Sharpe Ratio  84.51%
Sortino             1.77
Sortino/√2          1.25
Omega               1.21

Max Drawdown        -7.63%
Longest DD Days     51

Gain/Pain Ratio     0.21
Gain/Pain (1M)      1.04

Payoff Ratio        1.18
Profit Factor       1.21
Common Sense Ratio  1.24
CPC Index           0.73
Tail Ratio          1.02
Outlier Win Ratio   2.69
Outlier Loss Ratio  2.81

MTD                 -2.3%
3M                  2.27%
6M                  8.62%
YTD                 20.07%
1Y                  12.72%
3Y (ann.)           11.6%
5Y (ann.)           11.6%
10Y (ann.)          11.6%
All-time (ann.)     11.6%

Avg. Drawdown       -2.68%
Avg. Drawdown Days  16
Recovery Factor     1.67
Ulcer Index         0.03
Serenity Index      0.82

```

### Additional Resources

- [formulaic_alphas.py](formulaic_alphas.py): An array of formulae dedicated to alpha derivation.
- [notebook_05.py](notebook_05.py), [notebook_06.py](notebook_06.py): Scripts complementing specific notebooks.
- [top_mlflow_experiment.py](top_mlflow_experiment.py): Script for MLflow experiment tracking.
- [top_stock_dataset_pipeline.py](top_stock_dataset_pipeline.py): Scripted pipeline for stock dataset management.
- [utils.py](utils.py): A collection of utility functions and procedures for project-wide use.
- [auto_commit.sh](auto_commit.sh), [push_to_dockerhub.sh](push_to_dockerhub.sh): Automation scripts enhancing repository management.

### Old (Archived)

This section houses deprecated versions or previously used files, preserved for reference.

## Repository Configuration

For seamless integration and collaboration, our repository adheres to certain `.gitignore` rules:

```plaintext
# Default rule - ignore all files
*
# Exclude credential files and output notebooks
*_creds.txt
./output_notebook/**

# Explicitly allow only .py, .ipynb, and a few other specified files/folders
!*.py
!*.ipynb
!.gitignore
!*.sh
!./Old/
!./Old/** 
!.yml
