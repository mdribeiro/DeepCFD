import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init
import torch
from torch_geometric.nn import MessagePassing, GCNConv, GATConv  # , GraphSAGEConv, ChebConv, SAGEConv, GatedGraphConv, NNConv, ARMAConv, SGConv, SignedConv, AGNNConv, APPNP


class FeedForwardNN(nn.Module):
    def __init__(self, in_channels, out_channels, neurons_list=[10, 10, 10, 10],
                 activation=nn.ReLU(), normalize_weights=False, normalize_batch=False):
        super(FeedForwardNN, self).__init__()

        # Calculate the number of layers
        numLayers = len(neurons_list)

        # Calculate the total number of points per feature/output
        numPointsPerFeature = in_channels
        numPointsPerOutput = out_channels

        # Create a list to store the layers of the network
        self.layers = nn.ModuleList()

        # Add the input layer
        self.layers.append(nn.Linear(numPointsPerFeature, neurons_list[0]))

        # Add weight normalization if enabled
        if normalize_weights:
            self.layers[0] = nn.utils.weight_norm(self.layers[0])

        # Add batch normalization if enabled
        if normalize_batch:
            self.layers.append(nn.BatchNorm1d(neurons_list[0]))

        # Add the hidden layers
        for i in range(numLayers - 1):
            self.layers.append(nn.Linear(neurons_list[i], neurons_list[i+1]))

            # Add weight normalization if enabled
            if weight_norm:
                self.layers[-1] = nn.utils.weight_norm(self.layers[-1])

            # Add batch normalization if enabled
            if normalize_batch:
                self.layers.append(nn.BatchNorm1d(neurons_list[i+1]))

        # Add the output layer
        self.layers.append(nn.Linear(neurons_list[-1], numPointsPerOutput))

        self.activation = activation

        # initialize layers:
        for layer in self.layers:
            # init.xavier_uniform_(layer.weight)
            # init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            init.xavier_normal_(layer.weight)
            init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        return x


class GNNRegression(MessagePassing):
    def __init__(self, in_channels, out_channels, neurons_list=[10, 10, 10, 10], conv_type='gcn', activation=torch.tanh):
        super(GNNRegression, self).__init__(aggr='add')

        self.num_layers = len(neurons_list)
        self.layers = nn.ModuleList()

        if conv_type == 'gcn':
            self.conv = GCNConv  #(in_conv, out_conv)
        elif conv_type == 'gat':
            self.conv = GATConv  #(in_conv, out_conv, heads=1)
        # elif conv_type == 'graphsage':
        #     self.conv = GraphSAGEConv(in_conv, out_conv)
        # elif conv_type == 'cheb':
        #     self.conv = ChebConv(in_conv, out_conv, K=2)  # Set K as desired
        # elif conv_type == 'sage':
        #     self.conv = SAGEConv(in_conv, out_conv, aggr='mean')  # Set aggr as 'mean' or 'max'
        # elif conv_type == 'gated':
        #     self.conv = GatedGraphConv(in_conv, out_conv)
        # elif conv_type == 'nn':
        #     self.conv = NNConv(in_conv, out_conv, nn.Linear(in_channels, hidden_channels))  # Replace nn.Linear with desired function
        # elif conv_type == 'arma':
        #     self.conv = ARMAConv(in_conv, out_conv)
        # elif conv_type == 'sg':
        #     self.conv = SGConv(in_conv, out_conv, K=2)  # Set K as desired
        # elif conv_type == 'signed':
        #     self.conv = SignedConv(in_conv, out_conv, K=2)  # Set K as desired
        # elif conv_type == 'agnn':
        #     self.conv = AGNNConv(requires_grad=False)  # AGNNConv doesn't have weight parameters
        # elif conv_type == 'appnp':
        #     self.conv = APPNP(K=1)  # Set K as desired

        self.layers.append(self.conv(in_channels, neurons_list[0]))
        for layer in range(1, self.num_layers-1):
            self.layers.append(self.conv(neurons_list[layer], neurons_list[layer+1]))

        # self.layers.append(nn.Linear(neurons_list[-1], out_channels))
        self.layers.append(self.conv(neurons_list[-1], out_channels))
        self.activation = activation

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
        x = self.layers[-1](x, edge_index)
        return x
