import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, time_steps, bands, d_model=128, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(bands * time_steps, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.selu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.selu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(d_model, d_model)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.selu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.linear4 = nn.Linear(d_model, d_model)
        self.bn4 = nn.BatchNorm1d(d_model)
        self.selu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.linear5 = nn.Linear(d_model, d_model)
        self.bn5 = nn.BatchNorm1d(d_model)
        self.selu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout)

        self.linear7 = nn.Linear(d_model, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.selu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(dropout)

        self.linear8 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.selu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.selu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = self.selu3(x)
        x = self.dropout3(x)

        x = self.linear4(x)
        x = self.bn4(x)
        x = self.selu4(x)
        x = self.dropout4(x)

        x = self.linear5(x)
        x = self.bn5(x)
        x = self.selu5(x)
        x = self.dropout5(x)

        x = self.linear7(x)
        x = self.bn7(x)
        x = self.selu7(x)
        x = self.dropout7(x)

        x = self.linear8(x)
        return x

class FCN(nn.Module):
    def __init__(self, time_steps, bands):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(bands, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv1d(32, 8, kernel_size=3, padding=1)
        self.linear = nn.Linear(8*time_steps, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.dropout(x)

        x = self.conv8(x).flatten(1)
        x = self.linear(x)

        return x


class Transformer(nn.Module):
    def __init__(self, time_steps, bands, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(bands, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=128,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=3
        )

        self.output_proj = nn.Linear(d_model, 3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time_steps, bands)
        x = self.input_proj(x)  # (batch, time_steps, d_model)

        x = self.transformer(x)  # (batch, time_steps, d_model)

        x = x.transpose(1, 2)  # (batch, d_model, time_steps)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)

        x = self.output_proj(x)  # (batch, 3)
        return x


class GRU(nn.Module):
    def __init__(self, time_steps, bands, hidden_size=64, dropout=0.1):
        super().__init__()
        self.gru1 = nn.GRU(bands, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru3 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.gru4 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.gru5 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.gru6 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.gru7 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru8 = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time_steps, bands)

        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)
        x, _ = self.gru3(x)
        x = self.dropout(x)
        # x, _ = self.gru4(x)
        # x = self.dropout(x)
        # x, _ = self.gru5(x)
        # x = self.dropout(x)
        # x, _ = self.gru6(x)
        # x = self.dropout(x)
        # x, _ = self.gru7(x)
        # x = self.dropout(x)
        x, _ = self.gru8(x)

        x = x[:, -1, :]  # Take last time step
        x = self.output_proj(x)

        return x


class LSTM(nn.Module):
    def __init__(self, time_steps, bands, hidden_size=64, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(bands, hidden_size, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.lstm5 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.lstm6 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm7 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm8 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time_steps, bands)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        # x, _ = self.lstm2(x)
        # x = self.dropout(x)
        # x, _ = self.lstm3(x)
        # x = self.dropout(x)
        # x, _ = self.lstm4(x)
        # x = self.dropout(x)
        # x, _ = self.lstm5(x)
        # x = self.dropout(x)
        # x, _ = self.lstm6(x)
        # x = self.dropout(x)
        x, _ = self.lstm7(x)
        x = self.dropout(x)
        x, _ = self.lstm8(x)

        x = x[:, -1, :]
        x = self.output_proj(x)

        return x