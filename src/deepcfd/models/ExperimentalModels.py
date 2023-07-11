import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init
import torch


# class FeedForwardNN(nn.Module):
#     def __init__(self, in_channels, out_channels, neurons_list=[10, 10, 10, 10],
#                  activation=nn.ReLU(), normalize_weights=False, normalize_batch=False):
#         super(FeedForwardNN, self).__init__()

#         # Calculate the number of layers
#         numLayers = len(neurons_list)

#         # Calculate the total number of points per feature/output
#         numPointsPerFeature = in_channels
#         numPointsPerOutput = out_channels

#         # Create a list to store the layers of the network
#         self.layers = nn.ModuleList()

#         # Add the input layer
#         self.layers.append(nn.Linear(numPointsPerFeature + 2, neurons_list[0]))

#         # Add weight normalization if enabled
#         if normalize_weights:
#             self.layers[0] = nn.utils.weight_norm(self.layers[0])

#         # Add batch normalization if enabled
#         if normalize_batch:
#             self.layers.append(nn.BatchNorm1d(neurons_list[0]))

#         # Add the hidden layers with skip connections
#         for i in range(numLayers - 1):
#             self.layers.append(nn.Linear(neurons_list[i] * (i+2)  + numPointsPerFeature, neurons_list[i+1]))

#             # Add weight normalization if enabled
#             if weight_norm:
#                 self.layers[-1] = nn.utils.weight_norm(self.layers[-1])

#             # Add batch normalization if enabled
#             if normalize_batch:
#                 self.layers.append(nn.BatchNorm1d(neurons_list[i+1]))

#         # Add the output layer
#         self.layers.append(nn.Linear(neurons_list[-1] + numPointsPerFeature - 2, numPointsPerOutput))

#         self.activation = activation

#         # Initialize layers
#         for layer in self.layers:
#             init.xavier_normal_(layer.weight)
#             init.constant_(layer.bias, 0)

#     def forward(self, x):
#         skip_connections = []
#         for layer in self.layers[:-1]:
#             skip_connections.append(x)
#             x = torch.cat([x] + skip_connections, dim=1)
#             x = self.activation(layer(x))

#         # Last layer without skip connection
#         x = self.layers[-1](x)

#         return x



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
            init.xavier_normal_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        return x
