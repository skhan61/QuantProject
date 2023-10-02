# DATA_UNSEEN = '/home/sayem/Desktop/Project/data/total_unseen_data.h5'

# # Resetting the multi-index
# test_data_reset = test_data.reset_index()

# test_data_reset.to_hdf(DATA_UNSEEN, key='df', mode='w')

# print("Test data saved!")

# with pd.HDFStore(DATA_UNSEEN) as store:
#     keys = store.keys()

# print("Keys in the HDF5 file:", keys)