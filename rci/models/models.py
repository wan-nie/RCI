import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class BitModel(nn.Module):
    def __init__(self, N_orb=18):
        super(BitModel, self).__init__()
        self.N_orb = N_orb
        dense_size = self.get_dense_size(N_orb)
        
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1, padding=0)
        
        flatten_size = 4 * (N_orb - 1)
        self.dense1 = nn.Linear(flatten_size, dense_size)
        self.dense2 = nn.Linear(dense_size, dense_size // 2)
        self.dense3 = nn.Linear(dense_size // 2, dense_size // 4)
        self.dense4 = nn.Linear(dense_size // 4, 1)

    @staticmethod
    def get_dense_size(N_orb):
        return int(7 * np.sqrt(2 * N_orb))
    
    def forward(self, x):
        #
        batch_size = x.size(0)
        x = x.view(batch_size, self.N_orb, 2)
        x = x.permute(0, 2, 1)
        
        #
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)

        return x


class BitModelTransformer(nn.Module):
    def __init__(self, N_orb, N_elec, hidden_dim=128, dropout=0.1, num_layers=2, nhead=2):
        super().__init__()
        self.N_orb = N_orb
        self.N_elec = N_elec
        self.hidden_dim = hidden_dim

        # Project input channels (2) to hidden_dim for transformer
        self.input_proj = nn.Embedding(N_orb*2, hidden_dim)

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        indices = torch.where(x == 1)[1].reshape(-1, self.N_elec)  # -> (batch_size, N_elec)
        x = self.input_proj(indices)  # -> (batch_size, N_elec, hidden_dim)
        x = self.transformer_encoder(x)  # -> (batch_size, N_elec, hidden_dim)

        x = torch.mean(x, dim=1)
        x = self.proj_head(x)
        # out = torch.sigmoid(x)
        out = x
        return out


class BitModelTransformerTwoChannels(nn.Module):
    def __init__(self, N_orb, N_elec, hidden_dim=128, dropout=0.1, num_layers=2, nhead=2):
        super().__init__()
        self.N_orb = N_orb
        self.N_elec = N_elec
        self.hidden_dim = hidden_dim

        #
        self.alpha_orb_emb = nn.Embedding(N_orb, hidden_dim)
        self.beta_orb_emb = nn.Embedding(N_orb, hidden_dim)

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        alpha_indices = torch.where(x[:, ::2] == 1)[1].reshape(-1, self.N_elec//2)  # -> (batch_size, N_elec//2)
        beta_indices = torch.where(x[:, 1::2] == 1)[1].reshape(-1, self.N_elec//2)  # -> (batch_size, N_elec//2)

        x_alpha = self.encoder(self.alpha_orb_emb(alpha_indices))  # -> (batch_size, N_elec//2, hidden_dim)
        x_beta = self.encoder(self.beta_orb_emb(beta_indices))  # -> (batch_size, N_elec//2, hidden_dim)

        x_alpha = torch.mean(x_alpha, dim=1)  # -> (batch_size, hidden_dim)
        x_beta = torch.mean(x_beta, dim=1)  # -> (batch_size, hidden_dim)

        x = torch.cat([x_alpha, x_beta], dim=-1)  # -> (batch_size, hidden_dim*2)
        out = self.proj_head(x)
        return out


class BitModelTransformerOneChannel(nn.Module):
    def __init__(self, N_orb, N_elec, hidden_dim=128, dropout=0.1, num_layers=2, nhead=2):
        super().__init__()
        self.N_orb = N_orb
        self.N_elec = N_elec
        self.hidden_dim = hidden_dim

        #
        self.occ_emb = nn.Embedding(2*N_orb, hidden_dim, padding_idx=None)
        self.empt_emb = nn.Embedding(2*N_orb, hidden_dim, padding_idx=None)

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        occ_w = self.occ_emb.weight.unsqueeze(0)  # (1, 2*N_orb, hidden_dim)
        empt_w = self.empt_emb.weight.unsqueeze(0)  # (1, 2*N_orb, hidden_dim)
        x_bool = (x == 1).unsqueeze(-1)  # (batch_size, 2*N_orb, 1)
        x = torch.where(x_bool, occ_w, empt_w)  # (batch_size, 2*N_orb, hidden_dim)

        x = self.encoder(x)  # (batch_size, 2*N_orb, hidden_dim)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)

        out = self.proj_head(x)
        return out
    
