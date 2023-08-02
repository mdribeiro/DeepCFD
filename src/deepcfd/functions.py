import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyDOE import lhs


def CreateCollocationPoints(xBounds, tBounds, numSamples, numRe=None, ReBounds=None):
    """
    Create collocation points for PINNs
    """
    x = xBounds[0] + (xBounds[1] - xBounds[0]) * lhs(1, samples=numSamples)
    t = tBounds[0] + (tBounds[1] - tBounds[0]) * lhs(1, samples=numSamples)
    if (ReBounds==None):
        pointsInside = np.concatenate((x, t), axis=1)
    else:
        # Re = np.expand_dims(np.random.choice(np.linspace(ReBounds[0],ReBounds[1],numRe), size=numSamples), axis=1)
        # Re = np.ones_like(x)*ReBounds
        Re = ReBounds[0] + (ReBounds[1] - ReBounds[0]) * lhs(1, samples=numRe)
        pointsInside = np.concatenate((x, t, Re), axis=1)

    x = np.asarray([-1.0 if number == 0 else 1.0 for number in np.random.randint(0, 2, numSamples)]
                   ).reshape((numSamples, 1))
    t = tBounds[0] + (tBounds[1] - tBounds[0]) * lhs(1, samples=numSamples)
    if (ReBounds==None):
        pointsBoundary = np.concatenate((x, t), axis=1)
    else:
        pointsBoundary = np.concatenate((x, t, Re), axis=1)

    x = xBounds[0] + (xBounds[1] - xBounds[0]) * lhs(1, samples=numSamples)
    t = np.zeros_like(x)
    if (ReBounds==None):
        pointsInitial = np.concatenate((x, t), axis=1)
    else:
        pointsInitial = np.concatenate((x, t, Re), axis=1)

    return pointsInside, pointsBoundary, pointsInitial


def Burgers(x, t, Re):
    """ analytical solution of Burgers equation """

    td = t + 1
    num = x/(td)

    t0 = np.exp(Re/8)
    den = 1 + np.sqrt(td/t0) * np.exp(Re * (x**2/(4*td)))

    u_xt = num/den

    return u_xt


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
        dataset_size = len(tensor)
        indices = list(range(dataset_size))
        split = int(np.floor(ratio * dataset_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        split1.append(tensor[train_indices])
        split2.append(tensor[test_indices])
        # split1.append(tensor[:int(len(tensor) * ratio)])
        # split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2


def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)


def visualize(sample_y, out_y, error, s, savePath="../tmp/run.png"):
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


def visualizeScatter(sample_y, out_y, sample_x, savePath="../tmp/run.png"):
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


def visualize1DBurgers(time_label, test_x, test_re, options, model, analyticial_function,
                       xBounds, tBounds, test_Renum, savePath="../tmp/10_Re_100_samplePoints_pinn1DBurgers.png"):
    outs, targets = [], []
    for t in time_label:
        test_t = torch.ones_like(test_x) * t
        test_points = torch.cat((test_x, test_t, test_re/1000.), dim=1).to(options["device"])
        outs.append(model(test_points).cpu().detach().numpy())
        targets.append(analyticial_function(
            test_points[:, 0], test_points[:, 1], test_re[:, 0]).cpu().detach().numpy())
    out_25, out_50, out_75 = outs
    target_25, target_50, target_75 = targets

    xPoints = 50
    spacePoints = np.linspace(xBounds[0], xBounds[1], xPoints)
    tPoints = 100
    timePoints = np.linspace(tBounds[0], tBounds[1], tPoints)

    xx, tt = np.meshgrid(spacePoints, timePoints)

    pointsX = xx.T.reshape((xPoints * tPoints * 1, 1))
    pointsT = tt.T.reshape((xPoints * tPoints * 1, 1))
    pointsRe = np.ones_like(pointsX) * test_Renum

    test_points = np.concatenate((pointsX, pointsT, pointsRe/1000.), axis=1)

    out = model(torch.tensor(test_points).float().to(options["device"])).cpu().detach().numpy()
    target = analyticial_function(torch.tensor(test_points[:, 0]),
                                  torch.tensor(test_points[:, 1]),
                                  torch.tensor(test_Renum)).reshape((xPoints * tPoints, 1)
                                  ).cpu().detach().numpy()

    fig, axs = plt.subplots(3, 2, figsize=(12, 9), gridspec_kw={'width_ratios': [0.65, 0.25], "wspace": 0.25, "hspace": 0.6, "top": 0.95, "bottom": 0.1, "left": 0.1, "right": 0.95})

    # Define the data for each subplot
    data = [
        {'type': 'image', 'image': out.reshape((50, 100)), 'cmap': 'rainbow'},
        {'type': 'image', 'image': target.reshape((50, 100)), 'cmap': 'rainbow'},
        {'type': 'image', 'image': (out - target).reshape((50, 100)), 'cmap': 'rainbow'},
        {'type': 'line1', 'x': out_25, 'y': target_25},
        {'type': 'line2', 'x': out_50, 'y': target_50},
        {'type': 'line3', 'x': out_75, 'y': target_75}
    ]

    # Create each subplot and colorbar
    key_names = ["PINN", "Truth", "Error"]
    idx = 0
    im1 = axs[0, 0].imshow(data[0]['image'], interpolation='nearest', cmap=data[0]['cmap'],
                     extent=[tBounds[0], tBounds[1], xBounds[0], xBounds[1]],
                     origin='lower', aspect='auto')
    divider1 = make_axes_locatable(axs[0, 0])
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    axs[0, 0].set_title(key_names[0])
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel("x")

    im2 = axs[1, 0].imshow(data[1]['image'], interpolation='nearest', cmap=data[1]['cmap'],
                     extent=[tBounds[0], tBounds[1], xBounds[0], xBounds[1]],
                     origin='lower', aspect='auto')
    divider2 = make_axes_locatable(axs[1, 0])
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    axs[1, 0].set_title(key_names[1])
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("x")

    im3 = axs[2, 0].imshow(data[2]['image'], interpolation='nearest', cmap=data[2]['cmap'],
                     extent=[tBounds[0], tBounds[1], xBounds[0], xBounds[1]],
                     origin='lower', aspect='auto')
    divider3 = make_axes_locatable(axs[2, 0])
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    axs[2, 0].set_title(key_names[2])
    axs[2, 0].set_xlabel("t")
    axs[2, 0].set_ylabel("x")

    x_line = np.linspace(-1, 1, 100)

    axs[0, 1].plot(x_line, data[3]['x'], '-k', label="pinn")
    axs[0, 1].plot(x_line, data[3]['y'], '--r', label="truth")
    axs[0, 1].set_title("t = 0.25s")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].legend()

    axs[1, 1].plot(x_line, data[4]['x'], '-k', label="pinn")
    axs[1, 1].plot(x_line, data[4]['y'], '--r', label="truth")
    axs[1, 1].set_title("t = 0.50s")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].legend()

    axs[2, 1].plot(x_line, data[5]['x'], '-k', label="pinn")
    axs[2, 1].plot(x_line, data[5]['y'], '--r', label="truth")
    axs[2, 1].set_title("t = 0.75s")
    axs[2, 1].set_xlabel("x")
    axs[2, 1].legend()

    plt.savefig(savePath)
    plt.show()
