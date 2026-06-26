import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import pickle
import os

os.makedirs("predict", exist_ok=True)
models = ["RandomForest_full",
            # "XGBoost_full"
            # "SVM_full"
            ]


merged_data = np.load('postresearch/data/merged_data.npy')
time_steps, bands, h, w = merged_data.shape
merged_data = merged_data.reshape(time_steps, bands, h * w)
merged_data = merged_data.transpose(2, 1, 0)
merged_data = merged_data.reshape(h * w, -1)

dataloader = DataLoader(merged_data, batch_size=2048, shuffle=False)

for model_name in models:
    model_path = f'weights/{model_name}.pth'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predictions = []
    with tqdm(dataloader, desc=f"Predicting {model_name}") as pbar:
        for batch in dataloader:
            batch = batch.numpy()
            preds = model.predict(batch)
            predictions.extend(preds)
            pbar.update(1)

    predictions = np.array(predictions).reshape(h, w)
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    color_map[predictions == 0] = [0, 0, 0]
    color_map[predictions == 1] = [0, 128, 128]
    color_map[predictions == 2] = [255, 99, 71]

    output_image_path = f'predict/{model_name}.png'
    cv2.imwrite(output_image_path, color_map)
    print(f"{model_name} predicted image saved as {output_image_path}")
