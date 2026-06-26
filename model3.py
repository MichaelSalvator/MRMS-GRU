import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HalfPool1d(nn.Module):
    def forward(self, x):
        L = x.size(-1)
        out_L = max(1, math.ceil(L / 2))
        return F.adaptive_max_pool1d(x, output_size=out_L)

class TSExplorer(nn.Module):
    def __init__(self, time_steps, bands):
        super().__init__()
        self.conv1_s1 = nn.Conv1d(bands, 16, kernel_size=1)
        self.conv1_s2 = nn.Conv1d(bands, 16, kernel_size=3, padding=1)
        self.conv1_s3a = nn.Conv1d(bands, 16, kernel_size=3, padding=1)
        self.conv1_s3b = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.gru1_conv = nn.GRU(input_size=bands, hidden_size=16, batch_first=True)
        self.bn1 = nn.BatchNorm1d(16)
        self.selu1 = nn.SELU()
        self.pool1 = HalfPool1d()
        self.gru1 = nn.GRU(input_size=16, hidden_size=32, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2_s1 = nn.Conv1d(64, 32, kernel_size=1)
        self.conv2_s2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv2_s3a = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv2_s3b = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.gru2_conv = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.selu2 = nn.SELU()
        self.pool2 = HalfPool1d()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gru3_conv = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        # Add a projection layer for residual connection
        self.residual_proj = nn.Conv1d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.selu3 = nn.SELU()
        self.pool3 = HalfPool1d()
        self.dropout3 = nn.Dropout(0.1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x_s1 = self.conv1_s1(x)
        x_s2 = self.conv1_s2(x)
        x_s3 = self.conv1_s3b(self.conv1_s3a(x))
        x_t = x.permute(0, 2, 1)
        x_s4, _ = self.gru1_conv(x_t)
        x_s4 = x_s4.permute(0, 2, 1)
        x = x_s1 + x_s2 + x_s3 + x_s4
        x = self.selu1(self.bn1(x))
        x = self.pool1(x) ###
        x_t = x.permute(0, 2, 1)
        x_t, _ = self.gru1(x_t)
        x = x_t.permute(0, 2, 1)
        x = self.dropout1(x)

        x_s1 = self.conv2_s1(x)
        x_s2 = self.conv2_s2(x)
        x_s3 = self.conv2_s3b(self.conv2_s3a(x))
        x_t = x.permute(0, 2, 1)
        x_s4, _ = self.gru2_conv(x_t)
        x_s4 = x_s4.permute(0, 2, 1)
        x = x_s1 + x_s2 + x_s3 + x_s4
        x = self.selu2(self.bn2(x))
        x_residual = x
        x = self.pool2(x) ###
        x = self.dropout2(x)

        x_conv = self.conv3(x)
        x_t = x.permute(0, 2, 1)
        x_gru, _ = self.gru3_conv(x_t)
        x_gru = x_gru.permute(0, 2, 1)
        x_residual = self.residual_proj(x_residual)
        x_residual = self.pool3(x_residual)  ###

        x = x_conv + x_gru + x_residual
        x = self.selu3(self.bn3(x))
        x = self.pool3(x) ###
        x = self.dropout3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(x)

        x = self.selu3(x)
        x = self.fc2(x)
        return x
