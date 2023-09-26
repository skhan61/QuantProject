#!/bin/bash

# Define the top values
top_values=(250 500)

# Loop through each top value
for top in "${top_values[@]}"; do
    echo "Running for top value: $top"
    
    # Execute notebooks with papermill
    papermill 02_sample_selection.ipynb output_02_$top.ipynb -p top $top
    papermill 03_common_alpha_factors.ipynb output_03_$top.ipynb -p top $top
    papermill 04_101_formulaic_alphas.ipynb output_04_$top.ipynb -p top $top
    
    # Delete the output notebooks
    rm output_02_$top.ipynb
    rm output_03_$top.ipynb
    rm output_04_$top.ipynb
    
    # Execute Python scripts
    python notebook_05.py $top
    python notebook_06.py $top
    
    # Rename the dataset.h5 if it exists
    if [[ -f "/home/sayem/Desktop/Project/data/dataset.h5" ]]; then
        mv /home/sayem/Desktop/Project/data/dataset.h5 /home/sayem/Desktop/Project/data/${top}_dataset.h5
    else
        echo "Warning: /home/sayem/Desktop/Project/data/dataset.h5 does not exist and cannot be renamed."
    fi
done
