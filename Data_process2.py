import numpy as np
import os

data_dir = 'data_npy'
output_path = 'data/merged_data.npy'

npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
data_list = [np.load(os.path.join(data_dir, f)) for f in npy_files]
merged_data = np.array(data_list)
np.save(output_path, merged_data)
