import h5py
import os

# Get the list of all files and directories in the current directory
contents = os.listdir('.')

# Print the contents
for item in contents:
    print(item)
    if h5py.is_hdf5(item):
        with h5py.File(item, 'r') as f:
        # Read a dataset from the file
            # keys = list(f.keys())
            # print(keys)
            data = f['0']
            print(data.keys())
            print(data['encodec_frame_len'][()])
            print(data['encodec_latents'][:])
            print(data['encodec_rvq'][:])
            print(data['spectrogram'][:])

            # Print the data
            # print(data[:])
# Open the HDF5 file
