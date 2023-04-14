import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .AutoEncoder import create_layer


class AutoEncoderEx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[8, 16, 16, 32],
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        super().__init__()
        assert len(filters) > 0
        encoder = []
        decoder = [[] for _ in range(out_channels)]
        for i in range(len(filters)):
            if i == 0:
                encoder_layer = create_layer(in_channels, filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = [create_layer(filters[i], 1, kernel_size, weight_norm, False, final_activation, nn.ConvTranspose2d) for _ in range(out_channels)]
            else:
                encoder_layer = create_layer(filters[i-1], filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = [create_layer(filters[i], filters[i-1], kernel_size, weight_norm, batch_norm, activation, nn.ConvTranspose2d) for _ in range(out_channels)]
            encoder = encoder + [encoder_layer]
            for c in range(out_channels):
                decoder[c] = [decoder_layer[c]] + decoder[c]
        self.encoder = nn.Sequential(*encoder)
        for c in range(out_channels):
            decoder[c] = nn.Sequential(*decoder[c])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        y = []
        for c in self.decoder:
            y.append(c(x))
        return torch.cat(y, dim=1)
