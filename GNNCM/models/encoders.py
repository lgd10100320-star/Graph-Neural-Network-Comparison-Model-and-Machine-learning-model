
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool

class GCNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_pool = global_add_pool(x, batch)
        return x, x_pool

class GINEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(GINEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_pool = global_add_pool(x, batch)
        return x, x_pool

def get_encoder(name, input_dim, hidden_dim, num_layers, dropout=0.5):
    if name.lower() == 'gcn':
        return GCNEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif name.lower() == 'gin':
        return GINEncoder(input_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Encoder '{name}' not supported. Choose from 'gcn' or 'gin'.")
