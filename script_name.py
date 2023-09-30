import pandas as pd
import sys

def is_desired_structure(dataset):
    """
    Check if the dataset matches the desired structure.
    """
    if not isinstance(dataset.index, pd.MultiIndex):
        return False
    
    if dataset.index.names != ['date', 'ticker']:
        return False
    
    return True

def main(top_value):
    file_path = f"/home/sayem/Desktop/Project/data/{top_value}_dataset.h5"
    
    with pd.HDFStore(file_path) as store:
        keys = store.keys()
        for key in keys:
            dataset = store[key]
            if is_desired_structure(dataset):
                print(f"{key} matches the desired structure.")
            else:
                print(f"{key} does NOT match the desired structure.")
                
            # Print the head of the dataset
            print(dataset.head())
            
            print("--------------------------------------------------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the 'top' value as an argument.")
        sys.exit(1)
    
    top = sys.argv[1]
    main(top)
