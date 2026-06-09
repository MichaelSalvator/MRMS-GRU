import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import os

from model import DPTSC
from model_others import DNN, FCN, Transformer, GRU, LSTM
from model3 import TSExplorer
from Dataloader import RemoteSensingDataset

model_name = "TSExplorer"
batch_size = 1024 * 8
length_names = ['full', 'half', 'quarter', 'eighth']
# length_names = ['quarter']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


os.makedirs('weights', exist_ok=True)
os.makedirs('logs', exist_ok=True)
Epoch = 200

for length in length_names:
    dataset_obj = RemoteSensingDataset(
        f'postresearch/data/val_samples_{length}.npy',
        f'postresearch/data/val_labels_{length}.npy'
    )
    train_size = int(0.6 * len(dataset_obj))
    val_size = len(dataset_obj) - train_size
    train_dataset, val_dataset = random_split(dataset_obj, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    bands, time_steps = dataset_obj[0][0].shape
    ######
    model = TSExplorer(time_steps, bands).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = FocalLoss()

    best_val_f1 = 0
    log_data = []

    for epoch in range(Epoch):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for data, labels in train_loader:
            data, labels = data.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = precision_score(train_labels, train_preds, average='macro')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_kappa = cohen_kappa_score(train_labels, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.squeeze().to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = precision_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_kappa = cohen_kappa_score(val_labels, val_preds)

        print(f"Length: {length}, Epoch {epoch + 1}/{Epoch}: "
              f"Train Loss: {train_loss:.2f}, Train F1: {train_f1:.2f}, "
              f"Val Loss: {val_loss:.2f}, Val F1: {val_f1:.2f}")

        round_len = 4
        log_data.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, round_len),
            'train_acc': round(train_acc, round_len),
            'train_recall': round(train_recall, round_len),
            'train_f1': round(train_f1, round_len),
            'train_kappa': round(train_kappa, round_len),
            'val_loss': round(val_loss, round_len),
            'val_acc': round(val_acc, round_len),
            'val_recall': round(val_recall, round_len),
            'val_f1': round(val_f1, round_len),
            'val_kappa': round(val_kappa, round_len)
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'weights/{model_name}_{length}.pth')

        pd.DataFrame(log_data).to_csv(f'logs/{model_name}_{length}.csv', index=False)

print("训练完成")