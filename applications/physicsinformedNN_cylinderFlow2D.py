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
from pyDOE import lhs
from sklearn.metrics import r2_score
from torch.nn.utils import parameters_to_vector


def calc_r2(target, output):
    value = r2_score(target, output)
    value = np.where(value < 0.0, 0., (np.where(value == np.inf, 0., value)))
    return value

from deepcfd.models.ExperimentalModels import FeedForwardNN
net = FeedForwardNN


def circle(r=0.02, xc=0, yc=0, n_samples=100):
    theta = np.linspace(0, 2*np.pi, n_samples)
    x = xc + r*np.cos(theta).reshape((n_samples, 1))
    y = yc + r*np.sin(theta).reshape((n_samples, 1))
    circ = np.concatenate((x, y), axis=1)
    return circ


def parseOpts(argv):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = "mymodel.pt"
    learning_rate = 1e-3  # 0.001
    epochs = [10, 10]
    batch_size = 1024  #21322 #1024
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
            epochs = list(map(int, list(arg.split(", "))))  # int(arg)
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
        'show_mesh': False,
        'visualize': visualize,
    }

    return options


if __name__ == "__main__":
    options = parseOpts(sys.argv[1:])

    import pandas as pd
    data = pd.read_csv('/home/dias_ma/Github/cylinderFlow2D.csv')
    cylinderFlow2D = data[["Points:0", "Points:1", "Points:2", "U:0", "U:1", "p"]].values

    idx = np.where((cylinderFlow2D[:, 2] == 0.0))
    points_f, output_f = cylinderFlow2D[idx][:, 0:2], cylinderFlow2D[idx][:, 3:]

    idx_shuffle = np.linspace(0, len(points_f) - 1, len(points_f), dtype=int)
    np.random.shuffle(idx_shuffle)
    np.random.shuffle(idx_shuffle)
    np.random.shuffle(idx_shuffle)
    points_f, output_f = points_f[idx_shuffle], output_f[idx_shuffle]

    n_samples = len(points_f)

    points_surface = circle(r=0.02, n_samples=n_samples, xc=0.13, yc=0.06)

    # plt.scatter(points_f[:, 0], points_f[:,1]), plt.scatter(points_surface[:,0], points_surface[:,1]),  plt.show()
    x_left = np.zeros((n_samples, 1))
    y_left = 0.0 + 0.12 * lhs(1, samples=n_samples)
    points_left = np.concatenate((x_left, y_left), axis=1)

    x_right = np.ones((n_samples, 1)) * 0.26
    y_right = 0.0 + 0.12 * lhs(1, samples=n_samples)
    points_right = np.concatenate((x_right, y_right), axis=1)

    x_topbottom = 0.0 + 0.26 * lhs(1, samples=n_samples)
    y_topbottom = np.asarray([0.0 if number == 0 else 0.12 for
                              number in np.random.randint(0, 2, n_samples)]).reshape((n_samples, 1))
    points_topbottom = np.concatenate((x_topbottom, y_topbottom), axis=1)

    if options["show_mesh"]:
        plt.scatter(points_f[:, 0], points_f[:,1]), plt.scatter(points_surface[:, 0], points_surface[:, 1], color="red"),  
        plt.scatter(points_left[:, 0], points_left[:, 1], color="magenta")
        plt.scatter(points_right[:, 0], points_right[:, 1], color="yellow")
        plt.scatter(points_topbottom[:, 0], points_topbottom[:, 1], color="black")
        plt.show()

    # neurons_list = [10, 10, 10, 10, 10]
    # neurons_list = [80, 80, 80, 80]
    neurons_list = [30, 30, 30, 30, 30]
    model = options["net"](
        2,
        3,
        neurons_list=neurons_list,
        activation=nn.Tanh(),
        # activation=nn.ReLU(),
        normalize_weights=False,
        normalize_batch=False
    )

    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    x = torch.FloatTensor(points_f)
    y = torch.FloatTensor(output_f)

    points_surface = torch.FloatTensor(points_surface)
    points_left = torch.FloatTensor(points_left)
    points_right = torch.FloatTensor(points_right)
    points_topbottom = torch.FloatTensor(points_topbottom)

    model.nu = 0.0001
    model.uLeft = 0.1
    model.vLeft = 0.0
    model.uSurface = 0.0
    model.vSurface = 0.0
    model.noSlip = 0.0
    model.pRight = 0.0
    idxLeft = np.where((points_f[:, 0] == 0.0))
    model.xLeft = points_f[idxLeft]
    model.pLeft = output_f[idxLeft][:, 2:3]
    model.xLeft = torch.FloatTensor(model.xLeft)
    model.pLeft = torch.FloatTensor(model.pLeft)
    model.xLeft.requires_grad = True
    model.pLeft.requires_grad = True
    idxRight = np.where((points_f[:, 0] == 0.26))
    model.xRight = points_f[idxRight]
    model.pRight = output_f[idxRight][:, 2:3]
    model.xRight = torch.FloatTensor(model.xRight)
    model.pRight = torch.FloatTensor(model.pRight)
    model.xRight.requires_grad = True
    model.pRight.requires_grad = True

    model.alpha = 1.0
    model.beta = 1.0

    # Spliting dataset into 80% train and 20% test
    train_data, test_data = split_tensors(x,
                                          y,
                                          points_surface,
                                          points_left,
                                          points_right,
                                          points_topbottom,
                                          ratio=0.7)

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    test_x, test_y, _, _, _, _ = test_dataset.tensors

    model.count = 0

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

    def weight_grads(model, loss):
        grads = []
        with torch.no_grad():
            for submodule in model.modules():
                if isinstance(submodule, torch.nn.Linear):
                    weight_grads = torch.autograd.grad(loss, submodule.weight, retain_graph=True, allow_unused=True)[0]
                    bias_grads = torch.autograd.grad(loss, submodule.bias, retain_graph=True, allow_unused=True)[0]
                    grads.extend([g.view(-1) for g in [weight_grads, bias_grads] if g is not None])
        return torch.cat(grads).detach()

    def loss_func(model, batch):
        xInside, yData, xSurface, xLeft, xRight, xTopBottom = batch
        xInside.requires_grad = True
        xSurface.requires_grad = True
        xLeft.requires_grad = True
        xRight.requires_grad = True
        xTopBottom.requires_grad = True

        x = xInside[:, 0:1]
        y = xInside[:, 1:2]
        out = model(torch.cat([x, y], dim=1))
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        xS = xSurface[:, 0:1]
        yS = xSurface[:, 1:2]
        outS = model(torch.cat([xS, yS], dim=1))
        uS = outS[:, 0:1]
        vS = outS[:, 1:2]

        xL = xLeft[:, 0:1]
        yL = xLeft[:, 1:2]
        outL = model(torch.cat([xL, yL], dim=1))
        uL = outL[:, 0:1]
        vL = outL[:, 1:2]

        xTB = xTopBottom[:, 0:1]
        yTB = xTopBottom[:, 1:2]
        topIdx = torch.where(yTB[:, 0] == 0.12)
        bottomIdx = torch.where(yTB[:, 0] == 0.0)
        xT, yT = xTB[topIdx], yTB[topIdx]
        outT = model(torch.cat([xT, yT], dim=1))
        uT = outT[:, 0:1]
        vT = outT[:, 1:2]

        xB, yB = xTB[bottomIdx], yTB[bottomIdx]
        outB = model(torch.cat([xB, yB], dim=1))
        uB = outB[:, 0:1]
        vB = outB[:, 1:2]

        rightIdx = torch.where(x[:, 0] == 0.26)
        xR, yR = x[rightIdx], y[rightIdx]
        outR = model(torch.cat([xR, yR], dim=1))
        pR = outR[:, 2:3]
        dataRight = yData[rightIdx]
        pRight = dataRight[:, 2:3]

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
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x, x,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v, y,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y, y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        p_x = torch.autograd.grad(
            p, x,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True,
        )[0]

        p_y = torch.autograd.grad(
            p, y,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True,
        )[0]

        rho = 1.184
        # Euler
        # Navier-Stokes
        mom_loss_x = u * u_x + v * u_y + (p_x - model.nu * (u_xx + u_yy)) / rho
        mom_loss_y = u * v_x + v * v_y + (p_y - model.nu * (v_xx + v_yy)) / rho
        # Euler
        # mom_loss_x = u * u_x + v * u_y + p_x / rho
        # mom_loss_y = u * v_x + v * v_y + p_y / rho
        mass_loss = u_x + v_y

        loss_f = mom_loss_x + mom_loss_y + mass_loss
        model.residual = loss_f

        loss_left = (uL - model.uLeft) + (vL - model.vLeft)

        loss_surface = (uS - model.uSurface) + (vS - model.vSurface)

        loss_noslip = (uT - model.noSlip) + (vT - model.noSlip)

        loss_p = pR - pRight

        loss_residual = loss_f**2
        loss_velocity = loss_left**2 + loss_surface**2
        loss_noslip = loss_noslip**2
        loss_pressure = loss_p**2

        # delta = 0.1
        # grads_f = weight_grads(model, torch.sum(loss_residual))
        # grads_vel = weight_grads(model, torch.sum(loss_velocity) + torch.sum(loss_noslip))
        # grads_pressure = weight_grads(model, torch.sum(loss_pressure))

        # alpha_new = torch.mean(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_vel)).item()
        # beta_new = torch.mean(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_pressure)).item()

        # alpha = (1 - delta) * model.alpha + delta * alpha_new
        # beta = (1 - delta) * model.beta + delta * beta_new
        # model.alpha = alpha_new
        # model.beta = beta_new
        # return torch.sum(loss_residual) + alpha * (torch.sum(loss_velocity) + torch.sum(loss_noslip)) \
        #     + beta * torch.sum(loss_pressure), out
        return torch.sum(loss_residual) + (torch.sum(loss_velocity) + torch.sum(loss_noslip)) \
            + torch.sum(loss_pressure), out
        # return 0.05 * torch.sum(loss_residual) + (torch.sum(loss_velocity) + torch.sum(loss_noslip)), out

    validation_metrics = {
        "m_mse_name": "Total MSE",
        "m_mse_on_batch": lambda scope:
            float(torch.sum((scope["output"][:, :3] - scope["batch"][1]) ** 2)),
        "m_mse_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        "m_r2_name": "Total R2 Score",
        "m_r2_on_batch": lambda scope:
            float(calc_r2(scope["batch"][1].cpu().detach().numpy(),
                          scope["output"].cpu().detach().numpy())),
        "m_r2_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        "m_f_name": "Total Residual",
        "m_f_on_batch": lambda scope:
            float(torch.sum((model.residual) ** 2)),
        "m_f_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        "m_ux_name": "Ux MSE",
        "m_ux_on_batch": lambda scope:
            float(torch.sum((scope["output"][:, 0] -
                             scope["batch"][1][:, 0]) ** 2)),
        "m_ux_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        "m_uy_name": "Uy MSE",
        "m_uy_on_batch": lambda scope:
            float(torch.sum((scope["output"][:, 1] -
                             scope["batch"][1][:, 1]) ** 2)),
        "m_uy_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        "m_p_name": "p MSE",
        "m_p_on_batch": lambda scope:
            float(torch.sum((scope["output"][:, 2] -
                             scope["batch"][1][:, 2]) ** 2)),
        "m_p_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"])

    }

    # Training model with Adam
    pinnAdam, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
        model,
        loss_func,
        train_dataset,
        test_dataset,
        optimizerAdam,
        physics_informed=True,
        epochs=options["epochs"][0],
        batch_size=options["batch_size"],
        device=options["device"],
        **validation_metrics
    )

    # Training model with L-FBGS
    pinnModel, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
        pinnAdam,
        loss_func,
        train_dataset,
        test_dataset,
        optimizerLFBGS,
        physics_informed=True,
        epochs=options["epochs"][1],
        batch_size=len(train_dataset),
        initial_epoch=last_epoch + 1,
        device=options["device"],
        **validation_metrics
    )

    state_dict = pinnModel.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, options["output"])

    if (options["visualize"]):
        test_x = data[["Points:0", "Points:1"]].values
        test_x = torch.FloatTensor(test_x)
        sample_y = data[["U:0", "U:1", "p"]].values
        out = pinnModel(test_x.to(options["device"]))
        sample_x = test_x.cpu().detach().numpy()
        out_y = out.cpu().detach().numpy()
        visualizeScatter(sample_y, out_y, sample_x, savePath="./runPINN2D.png")

# %run applications/physicsinformedNN_cylinderFlow2D.py --device "gpu" --epochs 75 --batch-size 1024 --visualize True
# %run applications/physicsinformedNN_cylinderFlow2D.py --device "cpu" --epochs 75 --batch-size 1024 --visualize True
# %run applications/physicsinformedNN_cylinderFlow2D.py --device "cpu" --epochs 10 --batch-size 1024 --visualize True