
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/skhan61/QuantProject.git/main)

# End-to-End ML/DL Based Trade Strategy Development

## Overview

This project offers a comprehensive framework for the development and deployment of machine learning and deep learning models in trading strategies. Beginning with raw stock OHLCV data, the pipeline extends through data preprocessing, research, model training, and ultimately to live trading. The aim is to provide a full spectrum view of trading model development, inclusive of various skills and tools employed at different stages.

## Skills Demonstrated

### Data Preprocessing

The project commences with an efficient preprocessing of raw stock data, implemented in the notebook `01_raw_data_to_create_dataset.ipynb`. This stage is designed for handling large datasets, optimized to provide clean data for subsequent steps.

### Research and Dataset Building

Research-oriented notebooks focus on sample selection and factor research. They are covered in notebooks `02_sample_selection_and_returns_calculation.ipynb` and `03_common_alpha_factors.ipynb`. An additional notebook, `04_101_formulaic_alphas.ipynb`, explores a variety of alpha factors.

### ML/DL Model Development

Machine Learning and Deep Learning models are then trained and tested. Experiments are managed via MLflow for transparency and reproducibility. Relevant notebooks include `06_a_ML_models_forecasting.ipynb`, `06_b_DL_models_forecasting.ipynb`, and `06_c_DL_models_generative.ipynb`.

### Operational Tools

The project is supported by a series of scripts and utilities. These range from dataset management (`top_stock_dataset_pipeline.py`) to MLflow experiment tracking (`top_mlflow_experiment.py`). Operational aspects are further enhanced by scripts for Git and Docker automation (`auto_commit.sh`, `push_to_dockerhub.sh`).

## Project Objective

The overarching objective is to offer a robust, end-to-end example of machine learning and deep learning in trading. It is intended to serve as a display of various skills: from data management and feature engineering to model development and deployment. These skills are supplemented by proficiency in various development tools, such as Git, Docker, and Bash, among others.

For further inquiries or potential collaborations, please contact me at sayem.eee.kuet@gmail.com.

## End to End ML/DL Trade Pipeline:

```mermaid

graph LR
  style RawData fill:#0000000,stroke:#fffffa,stroke-width:4px;
  RawData[Raw Data]
  subgraph "Data Handling"
    RawData --> A[Data Preprocessing]
  end

subgraph "Research and Dataset Building"

A --> B[Sample Selection]

B --> C[Factor Research]

end

subgraph "ML/DL Model Training and Strategy Development"

C --> D[ML/DL Model Developement and Taining]

D --> E[Strategy Building & Backtesting]

end

subgraph "Deployment & Live Trade"

E --> F[Paper Trading]

F --> G[Live Trading]

end

  

%% Feedback Loops

G -.->|Feedback| B

G -.->|Feedback| C

G -.->|Feedback| E

F -.->|Feedback| C

F -.->|Feedback| B

F -.->|Feedback| C

F -.->|Feedback| C

F -.->|Feedback| D

F -.->|Feedback| E

```

# Strategy Tear Sheet

# Project Workflow and Notebooks

This document outlines the various stages of the project pipeline along with the corresponding Jupyter notebooks that provide detailed steps for each stage.

---

## Table of Contents
1. [Data Handling](#data-handling)
2. [Research and Dataset Building](#research-and-dataset-building)
3. [ML/DL Model Training and Strategy Development](#mldl-model-training-and-strategy-development)
4. [Deployment & Live Trade](#deployment--live-trade)

---

## Data Handling
- [01_raw_data_to_create_dataset.ipynb](01_raw_data_to_create_dataset.ipynb)  
  **Description**: Transforming raw data into a cohesive dataset structure.

---

## Research and Dataset Building
- [02_sample_selection_and_returns_calculation.ipynb](02_sample_selection_and_returns_calculation.ipynb)  
  **Description**: Insights and techniques for optimal data sample selection and calculating returns.
  
- [03_common_alpha_factors.ipynb](03_common_alpha_factors.ipynb)  
  **Description**: An exploration into the prevalent alpha factors in finance.
  
- [04_101_formulaic_alphas.ipynb](04_101_formulaic_alphas.ipynb)  
  **Description**: Constructing alphas from predefined formulae.
- [05_a_dataset_building.ipynb](05_a_dataset_building.ipynb)  
  **Description**: Streamlining the dataset for machine learning applications.
- [05_b_ic_based_feature_selection.ipynb](05_b_ic_based_feature_selection.ipynb)  
  **Description**: Harnessing the Information Coefficient (IC) for effective feature selection.
---

## ML/DL Model Training and Strategy Development

- [06_a_ML_models_forecasting.ipynb](06_a_ML_models_forecasting.ipynb)  
  **Description**: Diving into machine learning models for precise forecasting.

- [06_b_DL_models_forecasting.ipynb](06_b_DL_models_forecasting.ipynb)  
  **Description**: A deep dive into the world of deep learning forecasting models.
  
- [06_c_DL_models_generative.ipynb](06_c_DL_models_generative.ipynb)  
  **Description**: Leveraging deep learning for generative model constructs.

- [07_strategy_building_and_backtesting.ipynb](07_strategy_building_and_backtesting.ipynb)  
  **Description**: Constructing and assessing the trading strategy.

---

<-- ## Deployment & Live Trade
*(No notebooks as of now for this stage)* -->

## Strategy Results

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

```