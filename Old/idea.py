## Idea: Convrting returns as a image shapshot: IPCV project
import numpy as np
import math

# Assuming 'dataset_ranked' is your dataframe and has already been loaded into the current workspace.
unique_dates = dataset_ranked.index.get_level_values(0).unique()

images = []

# Determine the dimensions of the 2D image based on the number of tickers.
num_tickers = len(dataset_ranked.index.get_level_values(1).unique())
side_length = math.floor(math.sqrt(num_tickers))  # Largest square less than num_tickers

# Adjust the number of tickers
num_tickers = side_length * side_length

for idx, date in enumerate(unique_dates):
    day_data = dataset_ranked.loc[date].iloc[:num_tickers]  # Keep only the number of rows we need
    
    if all(feature in day_data.columns for feature in ['FEATURE_ret_frac_order', 'FEATURE_ret_01d', 'FEATURE_ret_02d']):
        
        channels = []
        
        for feature in ['FEATURE_ret_frac_order', 'FEATURE_ret_01d', 'FEATURE_ret_02d']:
            # Convert the data to a 2D image representation
            feature_data = day_data[feature].values
            # Reshape it to a square matrix
            feature_image = feature_data.reshape(side_length, side_length)
            channels.append(feature_image)
        
        # Stack the channels to form a multi-channel image
        image = np.stack(channels, axis=-1)
        images.append(image)
    else:
        print(f"Skipping date {date} due to inadequate data.")

print("Number of images:", len(images))
if images:
    print(images[0].shape)  # This should give (side_length, side_length, 3)
else:
    print("No images created.")