import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


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

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        return x
