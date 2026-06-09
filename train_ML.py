import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import os
import pickle
from torch.utils.data import DataLoader, random_split

from Dataloader import RemoteSensingDataset

model_name = "RandomForest"
batch_size = 256
length_names = ['full', 'half', 'quarter', 'eighth']

os.makedirs('weights', exist_ok=True)
os.makedirs('logs', exist_ok=True)

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

    # SVM model (commented out)
    # model = SVC(kernel='rbf', probability=True, random_state=42)

    # RandomForest model (commented out)
    # model = RandomForestClassifier(n_estimators=25, random_state=42)

    # XGBoost model
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    best_val_acc = 0
    log_data = []

    train_data = []
    train_labels = []
    for data, labels in train_loader:
        data = data.numpy().reshape(data.shape[0], -1)
        labels = labels.numpy().squeeze()
        train_data.append(data)
        train_labels.append(labels)
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    model.fit(train_data, train_labels)

    for epoch in range(1):
        train_preds = model.predict(train_data)
        train_acc = precision_score(train_labels, train_preds, average='macro')

        val_data = []
        val_labels = []
        for data, labels in val_loader:
            data = data.numpy().reshape(data.shape[0], -1)
            labels = labels.numpy().squeeze()
            val_data.append(data)
            val_labels.append(labels)
        val_data = np.concatenate(val_data, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        train_preds = model.predict(train_data)
        train_acc = precision_score(train_labels, train_preds, average='macro')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_kappa = cohen_kappa_score(train_labels, train_preds)

        val_preds = model.predict(val_data)
        val_acc = precision_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_kappa = cohen_kappa_score(val_labels, val_preds)

        print(f"Length: {length}, "
              f"Epoch {epoch + 1}/1: Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")

        round_len = 4
        log_data.append({
            'train_acc': round(train_acc, round_len),
            'train_recall': round(train_recall, round_len),
            'train_f1': round(train_f1, round_len),
            'train_kappa': round(train_kappa, round_len),
            'val_acc': round(val_acc, round_len),
            'val_recall': round(val_recall, round_len),
            'val_f1': round(val_f1, round_len),
            'val_kappa': round(val_kappa, round_len)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(f'weights/{model_name}_{length}.pth', 'wb') as f:
                pickle.dump(model, f)

        pd.DataFrame(log_data).to_csv(f'logs/{model_name}_{length}.csv', index=False)

print("训练完成")