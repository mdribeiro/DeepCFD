import os
import json
import torch
import pickle
import random
import getopt
import time
import sys
import numpy as np
import pandas as pd
from deepcfd.train_functions import *
from deepcfd.functions import *
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from matplotlib.path import Path
import scipy.interpolate
from pyDOE import lhs
from sklearn.metrics import r2_score
from torch.nn.utils import parameters_to_vector
import torch.utils.tensorboard as tb
from deepcfd.models.ExperimentalModels import FeedForwardNN
net = FeedForwardNN


def calc_r2(data, model, device):
    with torch.no_grad():
        output = model.to("cpu")(data.tensors[0].to("cpu")).cpu().detach().numpy()
    model.to(device)
    target = data.tensors[1].cpu().detach().numpy()
    value = r2_score(target, output)
    value = np.where(value < 0.0, 0., value)
    value = np.where(value == np.inf, 0., value)
    return value


if __name__ == "__main__":

    options = {
        'device': "cuda",  # "cpu",
        'output': "mymodel.pt",
        'net': net,
        'learning_rate': 1e-3,
         # 'epochs': [40000, 1000],
         # 'epochs': [10000, 1000],
         'epochs': [1000, 100],
         # 'epochs': [100, 10],
        'batch_size': 1024,
        'patience': 500,
        'neurons_list': [40, 40, 40, 40, 40, 40, 40, 40],
        'activation': nn.Tanh(),
        'shuffle_train': True,
        'visualize': True,
        'fixed_w': False,
        'update_freq': 50,
        'init_w': [2.0, 2.0, 2.0],
        'max_w': 100.0,
        'min_w': 1.0,
        'show_points': True,
    }

    data = pd.read_csv('/home/dias_ma/Github/cylinderFlow2DCellData.csv')
    cylinderFlow2D = data[["x", "y", "U:0", "U:1", "p", "patchIDs"]].values

    points_f, output_f = cylinderFlow2D[:, 0:2], cylinderFlow2D[:, 2:5]
    patch_ids = cylinderFlow2D[:, 5:6]

    n_samples = len(points_f)

    points_inlet = points_f[np.where(patch_ids[:, 0] == 1)]
    output_inlet = output_f[np.where(patch_ids[:, 0] == 1)]

    points_outlet = points_f[np.where(patch_ids[:, 0] == 2)]
    output_outlet = output_f[np.where(patch_ids[:, 0] == 2)]

    points_top = points_f[np.where(patch_ids[:, 0] == 3)]
    output_top = output_f[np.where(patch_ids[:, 0] == 3)]

    points_bottom = points_f[np.where(patch_ids[:, 0] == 4)]
    output_bottom = output_f[np.where(patch_ids[:, 0] == 4)]

    points_surface = points_f[np.where(patch_ids[:, 0] == 5)]
    output_surface = output_f[np.where(patch_ids[:, 0] == 5)]

    points_f = points_f[np.where(patch_ids[:, 0] == 0)]
    output_f = output_f[np.where(patch_ids[:, 0] == 0)]

    # Mesh internal points
    idx_shuffle = np.linspace(0, len(points_f) - 1, len(points_f), dtype=int)
    np.random.shuffle(idx_shuffle)
    np.random.shuffle(idx_shuffle)
    np.random.shuffle(idx_shuffle)
    points_f, output_f = points_f[idx_shuffle], output_f[idx_shuffle]

    # Residual training points
    xmin, xmax = -0.98, 0.98
    ymin, ymax = -0.98, 0.98
    n_samples = 25000

    samples = lhs(2, samples=n_samples)
    x_coord = xmin + (xmax - xmin) * samples[:, 0]
    y_coord = ymin + (ymax - ymin) * samples[:, 1]
    coordinates = np.column_stack((x_coord, y_coord))

    center = np.array([0.0, 0.0])
    radius = 0.202845 + 0.001
    theta = np.linspace(0, 2 * np.pi, n_samples)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    points_cylinder = np.column_stack((circle_x, circle_y))

    cylinder_path = Path(points_cylinder)
    mask = ~cylinder_path.contains_points(coordinates)
    coordinates = coordinates[mask]

    if options["show_points"]:
        plt.scatter(coordinates[:, 0], coordinates[:,1]),
        plt.scatter(points_surface[:, 0], points_surface[:, 1], color="red"),
        plt.scatter(points_inlet[:, 0], points_inlet[:, 1], color="magenta")
        plt.scatter(points_outlet[:, 0], points_outlet[:, 1], color="yellow")
        plt.scatter(points_top[:, 0], points_top[:, 1], color="black")
        plt.scatter(points_bottom[:, 0], points_bottom[:, 1], color="black")
        plt.show()

    device = options["device"]
    neurons_list = options["neurons_list"]
    model = options["net"](
        2,
        3,
        neurons_list=neurons_list,
        activation=options["activation"],
        normalize_weights=False,
        normalize_batch=False
    )

    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    # These are the points used for training! NOT the same as the reference solution!
    x = torch.FloatTensor(coordinates)
    # Need to interpolate on mesh grid points so we can calcualte validation metrics during training!
    y = scipy.interpolate.griddata(points_f, output_f, (coordinates[:, 0], coordinates[:, 1]), method='nearest')
    y = torch.FloatTensor(y)

    points_surface = torch.FloatTensor(points_surface)
    output_surface = torch.FloatTensor(output_surface)

    points_inlet = torch.FloatTensor(points_inlet)
    output_inlet = torch.FloatTensor(output_inlet)

    points_outlet = torch.FloatTensor(points_outlet)
    output_outlet = torch.FloatTensor(output_outlet)

    points_top = torch.FloatTensor(points_top)
    output_top = torch.FloatTensor(output_top)

    points_bottom = torch.FloatTensor(points_bottom)
    output_bottom = torch.FloatTensor(output_bottom)

    points_surface.requires_grad = True
    output_surface.requires_grad = True

    points_inlet.requires_grad = True
    output_inlet.requires_grad = True

    points_outlet.requires_grad = True
    output_outlet.requires_grad = True

    points_top.requires_grad = True
    output_top.requires_grad = True

    points_bottom.requires_grad = True
    output_bottom.requires_grad = True

    model.nu = 0.02
    model.rho = 1.0
    model.alpha = options["init_w"][0]
    model.beta = options["init_w"][1]
    model.gamma = options["init_w"][2]
    model.count = 0

    model.points_surface = points_surface.to(device)
    model.output_surface = output_surface.to(device)

    model.points_inlet = points_inlet.to(device)
    model.output_inlet = output_inlet.to(device)

    model.points_outlet = points_outlet.to(device)
    model.output_outlet = output_outlet.to(device)

    model.points_top = points_top.to(device)
    model.output_top = output_top.to(device)

    model.points_bottom = points_bottom.to(device)
    model.output_bottom = output_bottom.to(device)

    # Spliting dataset into 80% train and 20% test
    train_data, test_data = split_tensors(x,
                                          y,
                                          ratio=0.8)

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    test_x, test_y = test_dataset.tensors

    torch.manual_seed(0)

    # Define optimizers
    optimizerAdam = torch.optim.AdamW(
    # optimizerAdam = torch.optim.Adam(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=0.0
    )
    optimizerLFBGS = torch.optim.LBFGS(
        model.parameters(),
        lr=1e-2
    )

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerAdam, milestones=[15000, 20000, 25000], gamma=0.25)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerAdam, milestones=[15000, 25000, 35000], gamma=0.1)
    m1, m2, m3 = int(0.3 * options["epochs"][0]), int(0.6 * options["epochs"][0]), int(0.9 * options["epochs"][0])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerAdam, milestones=[m1, m2, m3], gamma=0.1)

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

    def loss_weights(model, loss_residual, loss_left, loss_right, loss_top, loss_bottom, loss_surface):
        delta = 0.1
        grads_f = weight_grads(model, loss_residual)
        grads_left = weight_grads(model, loss_left)
        grads_right = weight_grads(model, loss_right)
        grads_noslip = weight_grads(model, (loss_top + loss_bottom + loss_surface) / 3)

        # alpha_new = torch.max(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_left)).item()
        # beta_new = torch.max(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_right)).item()
        # gamma_new = torch.max(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_noslip)).item()

        alpha_new = torch.mean(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_left))# .item(
        beta_new = torch.mean(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_right))# .item()
        gamma_new = torch.mean(torch.abs(grads_f)).item() / torch.mean(torch.abs(grads_noslip))# .item()

        max_value = torch.tensor(options["max_w"])
        min_value = torch.tensor(options["min_w"])
        alpha_new = torch.minimum(alpha_new, max_value)
        alpha_new = torch.maximum(alpha_new, min_value).item()

        beta_new = torch.minimum(beta_new, max_value)
        beta_new = torch.maximum(beta_new, min_value).item()

        gamma_new = torch.minimum(gamma_new, max_value)
        gamma_new = torch.maximum(gamma_new, min_value).item()

        alpha = (1 - delta) * model.alpha + delta * alpha_new
        beta = (1 - delta) * model.beta + delta * beta_new
        gamma = (1 - delta) * model.gamma + delta * gamma_new
        model.alpha = alpha_new
        model.beta = beta_new
        model.gamma = gamma_new
        model.count = 0

        return alpha, beta, gamma

    def loss_func(model, batch):
        xInside, _ = batch
        xInside.requires_grad = True

        x = xInside[:, 0:1]
        y = xInside[:, 1:2]
        out = model(torch.cat([x, y], dim=1))
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        xS = model.points_surface[:, 0:1]
        yS = model.points_surface[:, 1:2]
        outS = model(torch.cat([xS, yS], dim=1))
        uS = outS[:, 0:1]
        vS = outS[:, 1:2]
        pS = outS[:, 2:3]

        xL = model.points_inlet[:, 0:1]
        yL = model.points_inlet[:, 1:2]
        outL = model(torch.cat([xL, yL], dim=1))
        uL = outL[:, 0:1]
        vL = outL[:, 1:2]
        pL = outL[:, 2:3]

        xR = model.points_outlet[:, 0:1]
        yR = model.points_outlet[:, 1:2]
        outR = model(torch.cat([xR, yR], dim=1))
        uR = outR[:, 0:1]
        vR = outR[:, 1:2]
        pR = outR[:, 2:3]

        xT = model.points_top[:, 0:1]
        yT = model.points_top[:, 1:2]
        outT = model(torch.cat([xT, yT], dim=1))
        uT = outT[:, 0:1]
        vT = outT[:, 1:2]
        pT = outT[:, 2:3]

        xB = model.points_bottom[:, 0:1]
        yB = model.points_bottom[:, 1:2]
        outB = model(torch.cat([xB, yB], dim=1))
        uB = outB[:, 0:1]
        vB = outB[:, 1:2]
        pB = outB[:, 2:3]

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

        # Navier-Stokes
        mom_loss_x = u * u_x + v * u_y + p_x / model.rho - model.nu * (u_xx + u_yy)
        mom_loss_y = u * v_x + v * v_y + p_y / model.rho - model.nu * (v_xx + v_yy)
        mass_loss = u_x + v_y

        loss_f = mom_loss_x + mom_loss_y + mass_loss
        model.residual = loss_f

        loss_left = outL - model.output_inlet
        loss_right = outR - model.output_outlet

        loss_top = outT - model.output_top
        loss_bottom = outB - model.output_bottom
        loss_surface = outS - model.output_surface

        loss_residual = torch.sum(loss_f**2)
        loss_left = torch.sum(loss_left**2)
        loss_right = torch.sum(loss_right**2)
        loss_top = torch.sum(loss_top**2)
        loss_bottom = torch.sum(loss_bottom**2)
        loss_surface = torch.sum(loss_surface**2)

        model.count += 1
        if options["fixed_w"]:
            alpha, beta, gamma = model.alpha, model.beta, model.gamma
        else:
            if model.count == options["update_freq"]:
                alpha, beta, gamma = loss_weights(model, loss_residual, loss_left, loss_right, loss_top, loss_bottom, loss_surface)
            else:
                alpha, beta, gamma = model.alpha, model.beta, model.gamma

        return loss_residual + alpha * loss_left + beta * loss_right \
            + gamma * (loss_top + loss_bottom + loss_surface), out

    validation_metrics = {
        "m_mse_name": "Total MSE",
        "m_mse_on_batch": lambda scope:
            float(torch.sum((scope["output"][:, :3] - scope["batch"][1]) ** 2)),
        "m_mse_on_epoch": lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        "m_r2_name": "Total R2 Score",
        "m_r2_on_batch": lambda scope: float(0.0),
        "m_r2_on_epoch": lambda scope:
            float(calc_r2(scope["dataset"], scope["model"], scope["device"])),
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

    writerAdam = tb.SummaryWriter(comment="Adam_training")
    # # Training model with Adam
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
        shuffle_train=options["shuffle_train"],
        scheduler=scheduler,
        writer=writerAdam,
        **validation_metrics
    )

    state_dict = pinnAdam.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, options["output"])

    writerFullbatch = tb.SummaryWriter(comment="Fullbatch_training")
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
        shuffle_train=True,
        writer=writerFullbatch,
        **validation_metrics
    )
    writerAdam.close()
    writerFullbatch.close()
    state_dict = pinnModel.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, options["output"])

    if (options["visualize"]):
        # Evaluate on the mesh cell points (NOT USED IN TRAINING)
        test_x = data[["x", "y"]].values
        test_x = torch.FloatTensor(test_x)
        sample_y = data[["U:0", "U:1", "p"]].values
        out = pinnModel(test_x.to(options["device"]))
        sample_x = test_x.cpu().detach().numpy()
        out_y = out.cpu().detach().numpy()
        visualize2DNavierStokes(sample_y, out_y, sample_x, savePath="./runPINN2D_LongRun.png")
