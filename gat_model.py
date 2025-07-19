import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_linears = nn.ModuleList()

        # 第一层
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.res_linears.append(nn.Linear(in_channels, hidden_channels * heads) if in_channels != hidden_channels * heads else nn.Identity())

        # 第二层
        self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.res_linears.append(nn.Linear(hidden_channels * heads, hidden_channels * heads) if hidden_channels * heads != hidden_channels * heads else nn.Identity())

        # 第三层
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(out_channels))
        self.res_linears.append(nn.Linear(hidden_channels * heads, out_channels) if hidden_channels * heads != out_channels else nn.Identity())

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_input = x
            x = conv(x, edge_index)
            x = x + self.res_linears[i](x_input)
            x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x 