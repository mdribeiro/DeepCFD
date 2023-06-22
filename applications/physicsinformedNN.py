import os
import json
import torch
import pickle
import random
import getopt
import time
import sys
import numpy as np
from deepcfd.train_functions import *
from deepcfd.functions import *
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from deepcfd.models.ExperimentalModels import FeedForwardNN
net = FeedForwardNN


def parseOpts(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = "mymodel.pt"
    learning_rate = 0.001
    epochs = 500
    batch_size = 32
    patience = 500
    visualize = False

    try:
        opts, args = getopt.getopt(
            argv, "hd:n:mi:mo:o:k:f:l:e:b:p:v",
            [
                "device=",
                "output=",
                "learning-rate=",
                "epochs=",
                "batch-size=",
                "patience=",
                "visualize"
            ]
        )
    except getopt.GetoptError as e:
        print(e)
        print("python -m deepcfd --help")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--device"):
            if (arg == "cpu" or arg.startswith("cuda")):
                device = arg
            else:
                print("Unkown device " + str(arg) + ", only 'cpu', 'cuda'"
                      "'cuda:index', or comma-separated list of 'cuda:index'"
                      "are supported")
                exit(0)
        elif opt in ("-l", "--learning-rate"):
            learning_rate = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch-size"):
            batch_size = int(arg)
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-p", "--patience"):
            patience = arg
        elif opt in ("-v", "--visualize"):
            visualize = True

    options = {
        'device': device,
        'net': net,
        'output': output,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
        'visualize': visualize,
    }

    return options


if __name__ == "__main__":
    options = parseOpts(sys.argv[1:])

    numSamples = 1000
    xBounds = (-1, 1)
    tBounds = (0, 1)
    x, xBdr, xInitial = CreateCollocationPoints(xBounds, tBounds, numSamples)

    neurons_list = [10, 10, 10, 10, 10]
    model = options["net"](
        2,
        1,
        neurons_list=neurons_list,
        activation=nn.Tanh(),
        normalize_weights=False,
        normalize_batch=False
    )

    model.nu = 0.01 / np.pi

    y = Burgers(x[:, 0], x[:, 1], 1/model.nu).reshape((numSamples, 1))

    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    xBdr = torch.FloatTensor(xBdr)
    xInitial = torch.FloatTensor(xInitial)

    # Spliting dataset into 70% train and 30% test
    train_data, test_data = split_tensors(x, y, xBdr, xInitial, ratio=0.7)

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    test_x, test_y, _, _ = test_dataset.tensors

    torch.manual_seed(0)

    # Define optimizers
    optimizerAdam = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=0.005
    )
    optimizerLFBGS = torch.optim.LBFGS(
        model.parameters(),
        lr=1e-2
    )

    config = {}
    train_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    test_mse_curve = []
    train_f_curve = []
    test_f_curve = []

    def after_epoch(scope):
        train_loss_curve.append(scope["train_loss"])
        test_loss_curve.append(scope["val_loss"])
        train_mse_curve.append(scope["train_metrics"]["mse"])
        test_mse_curve.append(scope["val_metrics"]["mse"])
        train_f_curve.append(scope["train_metrics"]["ux"])
        test_f_curve.append(scope["val_metrics"]["ux"])

    def BurgersExact(x, t, Re):
        td = t + 1
        num = x/(td)

        t0 = torch.exp(Re/8)
        den = 1 + torch.sqrt(td/t0) * torch.exp(Re * (x**2/(4*td)))

        u_xt = num/den

        return u_xt

    def loss_func(model, batch):
        xInside, _, xBdr, xInitial = batch
        xInside.requires_grad = True
        xBdr.requires_grad = True
        xInitial.requires_grad = True

        x = xInside[:, 0:1]
        t = xInside[:, 1:2]
        u = model(torch.cat([x, t], dim=1))
        xB = xBdr[:, 0:1]
        tB = xBdr[:, 1:2]
        outBdr = model(torch.cat([xB, tB], dim=1))
        xI = xInitial[:, 0:1]
        tI = xInitial[:, 1:2]
        outInitial = model(torch.cat([xI, tI], dim=1))

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        loss_f = u_t + u * u_x - model.nu * u_xx

        Re = torch.tensor(1/model.nu, requires_grad=True)

        yBdr = BurgersExact(xBdr[:, 0:1], xBdr[:, 1:2], Re)
        yInitial = BurgersExact(xInitial[:, 0:1], xInitial[:, 1:2], Re)

        loss_Bdr = yBdr - outBdr
        loss_Initial = yInitial - outInitial

        loss = loss_f**2 + loss_Bdr**2 + loss_Initial**2

        return torch.sum(loss), u

    # Training model with Adam
    pinnAdam, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
        model,
        loss_func,
        train_dataset,
        test_dataset,
        optimizerAdam,
        physics_informed=True,
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        device=options["device"],
        m_mse_name="Total MSE",
        m_mse_on_batch=lambda scope:
            float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
        m_mse_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        m_f_name="Residual And BCs",
        m_f_on_batch=lambda scope:
            float(torch.sum((scope["output"][:, 0] -
                             scope["batch"][1][:, 0]) ** 2)),
        m_f_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"])
    )

    # Training model with L-FBGS
    pinnModel, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
        pinnAdam,
        loss_func,
        train_dataset,
        test_dataset,
        optimizerLFBGS,
        physics_informed=True,
        epochs=options["epochs"],
        batch_size=len(train_dataset),
        initial_epoch=last_epoch + 1,
        device=options["device"],
        m_mse_name="Total MSE",
        m_mse_on_batch=lambda scope:
            float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
        m_mse_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        m_f_name="Residual And BCs",
        m_f_on_batch=lambda scope:
            float(torch.sum((scope["output"][:, 0] -
                             scope["batch"][1][:, 0]) ** 2)),
        m_f_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"])
    )

    state_dict = pinnModel.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, options["output"])

    if (options["visualize"]):
        time_label = [0.25, 0.5, 0.75]
        test_x = torch.linspace(-1, 1, 100).reshape((100, 1))
        test_re = (torch.ones_like(test_x) * 1 /
                   pinnModel.nu).to(options["device"])
        visualize1DBurgers(time_label, test_x, test_re, options, pinnModel, BurgersExact, xBounds, tBounds)

# %run applications/physicsinformedNN.py --device "gpu" --epochs 75 --batch-size 32 --visualize True
# %run applications/physicsinformedNN.py --device "cpu" --epochs 75 --batch-size 32 --visualize True