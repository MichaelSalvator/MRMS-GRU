import os
import cv2
import numpy as np
from skimage import exposure
import imageio
import warnings
import pandas as pd
import rasterio

warnings.filterwarnings("ignore", category=UserWarning)

data_dir = "C:/Users/!Z#Y#C!/Desktop/new2/data10"
output_npy = "data_npy"
output_look = "data_look"
os.makedirs(output_npy, exist_ok=True)
os.makedirs(output_look, exist_ok=True)


cut_list = [[[5100, 8560, 1000, 700]]]


folder_names = os.listdir(data_dir)
for folder_index in range(len(folder_names)):
    folder_name = folder_names[folder_index]
    bands_name = os.listdir(f"{data_dir}/{folder_name}")


    cuts = cut_list[0]

    for cut_index in range(len(cuts)):

        [x, y, w, h] = cuts[cut_index]

        bands_data = []
        for band_name in bands_name:
            print(band_name)
            dataset = rasterio.open(f"{data_dir}/{folder_name}/{band_name}")
            band_data = dataset.read(1)
            band_data = band_data[y:y + h, x:x + w]
            bands_data.append(band_data)


        mid_B, mid_G, mid_R = bands_data[0] / bands_data[0].max(), bands_data[1] / bands_data[1].max(), bands_data[2] / bands_data[2].max()
        result_rgb = np.dstack((mid_B, mid_G, mid_R))
        result_rgb = exposure.equalize_hist(result_rgb) * 255
        result_rgb = result_rgb.astype(np.uint8)

        np.save(f"{output_npy}/{folder_names[folder_index][19:27]}.npy", np.array(bands_data))
        cv2.imwrite(f"{output_look}/{folder_names[folder_index][19:27]}.jpg", result_rgb)
