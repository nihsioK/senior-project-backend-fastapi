import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GraphConv, self).__init__()
        # Learnable adjacency matrix (initialized randomly)
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))  # (num_nodes x num_nodes)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)  # Batch normalization layer

    def forward(self, x):
        # x: (batch, in_channels, time, num_nodes)
        x = self.conv(x)

        # Apply the learnable adjacency matrix using einsum
        # x: (batch, out_channels, time, num_nodes)
        x = torch.einsum('nctv,vw->nctw', x, self.A)  # Propagate via the learnable graph

        x = self.bn(x)  # Normalize the output
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(padding, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)  # Batch normalization layer

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GestureGCN(nn.Module):
    def __init__(self, num_classes, num_nodes=21):
        super(GestureGCN, self).__init__()
        # Input has 3 channels (x, y, z)
        self.graph_conv1 = GraphConv(3, 64, num_nodes)
        self.temp_conv1 = TemporalConv(64, 64)
        self.graph_conv2 = GraphConv(64, 128, num_nodes)
        self.temp_conv2 = TemporalConv(128, 128)
        # Final fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, channels, time, num_nodes)
        x = F.relu(self.graph_conv1(x))
        x = F.relu(self.temp_conv1(x))
        x = F.relu(self.graph_conv2(x))
        x = F.relu(self.temp_conv2(x))
        # Global average pooling over the time and node dimensions
        x = x.mean(dim=2).mean(dim=2)
        x = self.fc(x)
        return x

