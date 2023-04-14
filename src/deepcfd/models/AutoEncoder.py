import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, convolution=nn.Conv2d):
    assert kernel_size % 2 == 1
    layer = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    if activation is not None:
        layer.append(activation())
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64],
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        super().__init__()
        assert len(filters) > 0
        encoder = []
        decoder = []
        for i in range(len(filters)):
            if i == 0:
                encoder_layer = create_layer(in_channels, filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], out_channels, kernel_size, weight_norm, False, final_activation, nn.ConvTranspose2d)
            else:
                encoder_layer = create_layer(filters[i-1], filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], filters[i-1], kernel_size, weight_norm, batch_norm, activation, nn.ConvTranspose2d)
            encoder = encoder + [encoder_layer]
            decoder = [decoder_layer] + decoder
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))
