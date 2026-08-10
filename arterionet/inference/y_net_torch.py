"""
Y-NET: Dual-encoder U-Net architecture for ABP reconstruction from ECG+PPG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YNETEncoder(nn.Module):
    """CNN encoder for single biosignal (ECG or PPG)."""
    
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(out_channels, out_channels*2, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(out_channels*2, out_channels*4, kernel_size=15, stride=1, padding=7)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels*2)
        self.bn3 = nn.BatchNorm1d(out_channels*4)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x


class YNETDecoder(nn.Module):
    """U-Net decoder with upsampling and skip connections."""
    
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv1 = nn.Conv1d(in_channels, in_channels//2, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(in_channels//2, in_channels//4, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(in_channels//4, out_channels, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.bn2 = nn.BatchNorm1d(in_channels//4)
    
    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upsample(x)
        x = torch.tanh(self.conv3(x))
        return x


class ASIHead(nn.Module):
    """Arterial Stiffness Index prediction head."""
    
    def __init__(self, in_channels=256, hidden_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        asi = torch.sigmoid(self.fc3(x))
        return asi


class BPVHead(nn.Module):
    """Blood Pressure Variability prediction head."""
    
    def __init__(self, in_channels=256, hidden_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        bpv = F.softplus(self.fc3(x))
        return bpv


class YNETModelTorch(nn.Module):
    """
    Y-NET: Dual-encoder U-Net for ABP reconstruction from ECG + PPG.
    
    Input: ECG (250 samples) + PPG (250 samples)
    Output: ABP waveform (250 samples), ASI (scalar), BPV (2 scalars: SBP_SD, DBP_SD)
    """
    
    def __init__(self):
        super().__init__()
        self.ecg_encoder = YNETEncoder(in_channels=1, out_channels=64)
        self.ppg_encoder = YNETEncoder(in_channels=1, out_channels=64)
        self.decoder = YNETDecoder(in_channels=256, out_channels=1)
        self.asi_head = ASIHead(in_channels=256, hidden_dim=128)
        self.bpv_head = BPVHead(in_channels=256, hidden_dim=128)
    
    def forward(self, ecg, ppg):
        """
        Args:
            ecg: (batch_size, 1, 250)
            ppg: (batch_size, 1, 250)
        
        Returns:
            abp: (batch_size, 1, 250)
            asi: (batch_size, 1)
            bpv: (batch_size, 2)
        """
        # Encode
        ecg_feat = self.ecg_encoder(ecg)
        ppg_feat = self.ppg_encoder(ppg)
        
        # Simple fusion: 0.5/0.5 weighting
        fused = 0.5 * ecg_feat + 0.5 * ppg_feat
        
        # Decode and extract biomarkers
        abp_pred = self.decoder(fused)
        asi_pred = self.asi_head(fused)
        bpv_pred = self.bpv_head(fused)
        
        return abp_pred, asi_pred, bpv_pred
