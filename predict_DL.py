import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from model import DPTSC
from model3 import TSExplorer
from model_others import DNN, FCN, Transformer, GRU, LSTM
import os

os.makedirs("predict", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


merged_data_np = np.load('postresearch/data/merged_data.npy')
time_steps, bands, h, w = merged_data_np.shape

models = {
    #"TSExplorer_quarter": TSExplorer(time_steps=time_steps, bands=bands),
    #"DNN_full": DNN(time_steps=time_steps, bands=bands),
    #"FCN_full": FCN(time_steps=time_steps, bands=bands),
    #"Transformer": Transformer(time_steps=time_steps, bands=bands),
    #"GRU": GRU(time_steps=time_steps, bands=bands),
    "LSTM": LSTM(time_steps=time_steps, bands=bands)
}


merged_data = merged_data_np.reshape(time_steps, bands, h * w).transpose(2, 1, 0)
merged_data = torch.tensor(merged_data, dtype=torch.float32).to(device)

dataloader = DataLoader(merged_data, batch_size=2048 * 8, shuffle=False)


for model_name, model in models.items():
    model.to(device)
    model.load_state_dict(torch.load(f'weights/{model_name}.pth'))
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {model_name}"):
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    predictions = np.array(predictions).reshape(h, w)

    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    color_map[predictions == 0] = [0, 0, 0]
    color_map[predictions == 1] = [0, 128, 128]
    color_map[predictions == 2] = [255, 99, 71]

    output_image_path = f'predict/{model_name}.png'
    cv2.imwrite(output_image_path, color_map)
    print(f"{model_name} predicted image saved as {output_image_path}")
