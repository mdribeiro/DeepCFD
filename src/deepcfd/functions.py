import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


class ModifiedTensorDataset(torch.utils.data.Dataset):

    def __init__(self, *tensors):
        self.tensors_x = tensors[0]
        self.tensors_y = tensors[1]
        self.tensors = tensors
        self.lengths = tuple([tensor.size(0) for tensor in self.tensors_x])

    def __getitem__(self, index):
        return tuple([tensor[index % length] for tensor, length in zip(self.tensors, self.lengths)])

    def __len__(self):
        return max(self.lengths)


def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)

def visualize(sample_y, out_y, error, s, savePath="./run.png"):
    minu = np.min(sample_y[s, 0, :, :])
    maxu = np.max(sample_y[s, 0, :, :])

    minv = np.min(sample_y[s, 1, :, :])
    maxv = np.max(sample_y[s, 1, :, :])

    minp = np.min(sample_y[s, 2, :, :])
    maxp = np.max(sample_y[s, 2, :, :])

    mineu = np.min(error[s, 0, :, :])
    maxeu = np.max(error[s, 0, :, :])

    minev = np.min(error[s, 1, :, :])
    maxev = np.max(error[s, 1, :, :])

    minep = np.min(error[s, 2, :, :])
    maxep = np.max(error[s, 2, :, :])

    nx = sample_y.shape[2]
    ny = sample_y.shape[3]

    plot_options = {'cmap': 'jet', 'origin': 'lower', 'extent': [0,nx,0,ny]}

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.subplot(3, 3, 1)
    plt.title('CFD', fontsize=18)
    plt.imshow(np.transpose(sample_y[s, 0, :, :]), vmin = minu, vmax = maxu, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Ux', fontsize=18)
    plt.subplot(3, 3, 2)
    plt.title('CNN', fontsize=18)
    plt.imshow(np.transpose(out_y[s, 0, :, :]), vmin = minu, vmax =maxu, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 3)
    plt.title('Error', fontsize=18)
    plt.imshow(np.transpose(error[s, 0, :, :]), vmin = mineu, vmax = maxeu, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 4)
    plt.imshow(np.transpose(sample_y[s, 1, :, :]), vmin = minv, vmax = maxv, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Uy', fontsize=18)
    plt.subplot(3, 3, 5)
    plt.imshow(np.transpose(out_y[s, 1, :, :]), vmin = minv, vmax = maxv, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 6)
    plt.imshow(np.transpose(error[s, 1, :, :]), vmin = minev, vmax = maxev, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 7)
    plt.imshow(np.transpose(sample_y[s, 2, :, :]), vmin = minp, vmax = maxp, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('p', fontsize=18)
    plt.subplot(3, 3, 8)
    plt.imshow(np.transpose(out_y[s, 2, :, :]), vmin = minp, vmax = maxp, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 9)
    plt.imshow(np.transpose(error[s, 2, :, :]), vmin = minep, vmax = maxep, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.savefig(savePath)
    plt.show()


def visualizeScatter(sample_y, out_y, sample_x, savePath="./run.png"):
    error = out_y - sample_y

    minu = np.min(sample_y[:, 0])
    maxu = np.max(sample_y[:, 0])

    minv = np.min(sample_y[:, 1])
    maxv = np.max(sample_y[:, 1])

    minp = np.min(sample_y[:, 2])
    maxp = np.max(sample_y[:, 2])

    mineu = np.min(error[:, 0])
    maxeu = np.max(error[:, 0])

    minev = np.min(error[:, 1])
    maxev = np.max(error[:, 1])

    minep = np.min(error[:, 2])
    maxep = np.max(error[:, 2])

    plot_options = {'cmap': 'jet'}

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.subplot(3, 3, 1)
    plt.title('CFD', fontsize=18)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=sample_y[:, 0], vmin=minu, vmax=maxu, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Ux', fontsize=18)
    plt.subplot(3, 3, 2)
    plt.title('MLP', fontsize=18)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=out_y[:, 0], vmin=minu, vmax=maxu, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 3)
    plt.title('Error', fontsize=18)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=error[:, 0], vmin=minu, vmax=maxu, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 4)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=sample_y[:, 1], vmin=minv, vmax=maxv, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Uy', fontsize=18)
    plt.subplot(3, 3, 5)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=out_y[:, 1], vmin=minv, vmax=maxv, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 6)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=error[:, 1], vmin=minv, vmax=maxv, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 7)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=sample_y[:, 2], vmin=minp, vmax=maxp, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('p', fontsize=18)
    plt.subplot(3, 3, 8)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=out_y[:, 2], vmin=minp, vmax=maxp, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 9)
    plt.scatter(sample_x[:, 0], sample_x[:, 1], c=error[:, 2], vmin=minp, vmax=maxp, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.savefig(savePath)
    plt.show()
