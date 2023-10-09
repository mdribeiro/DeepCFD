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
from torch_geometric.data import Data
import torch_geometric
import torch.utils.tensorboard as tb
from sklearn import neighbors
from deepcfd.models.ExperimentalModels import FeedForwardNN, GNNRegression
net = GNNRegression

torch.set_num_threads(16)

def create_graph(data_x, data_y, n_neighbours=9):
    convert_tensor = False
    if isinstance(data_x, torch.Tensor):
        data_x = data_x.clone().cpu().detach().numpy()
        data_y = data_y.clone().cpu().detach().numpy()
        convert_tensor = True
    tree = neighbors.KDTree(data_x, leaf_size=100, metric='euclidean')
    dist, receivers_list = tree.query(data_x, k=n_neighbours)

    num_nodes = len(data_x)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    l_edges = []
    for i in range(len(senders)):
        if (senders[i] != receivers[i]):
            edge_i = [senders[i], receivers[i]]
            l_edges.append(edge_i)

    edge_index_train = tuple(l_edges)
    edge_index_train = torch.tensor(edge_index_train, dtype=torch.long)
    if convert_tensor:
        data_x = torch.FloatTensor(data_x)
        data_y = torch.FloatTensor(data_y)
    graph = Data(x=data_x, y=data_y, edge_index=edge_index_train)

    return graph


def calc_r2(data, model, training, device):
    with torch.no_grad():
        if training:
            output = model.to("cpu")(data.tensors[0].to("cpu"), train_edge_index.T.to("cpu")).cpu().detach().numpy()
        else:
            output = model.to("cpu")(data.tensors[0].to("cpu"), val_edge_index.T.to("cpu")).cpu().detach().numpy()
    model.to(device)
    target = data.tensors[1].cpu().detach().numpy()
    value = r2_score(target, output)
    value = np.where(value < 0.0, 0., value)
    value = np.where(value == np.inf, 0., value)
    return value


if __name__ == "__main__":

    options = {
        'device': "cuda",
        'output': "trial.pt",
        # 'output' : "trial.pt",
        'net': net,
        # 'learning_rate': 1e-4,
        'learning_rate': 1e-3,
        # 'epochs': [40000, 1000],
        # 'epochs': [10000, 1000],
        # 'epochs': [10000, 500],
        'epochs': [2, 2],
        # 'batch_size': 1024,
        'batch_size': 1024,
        'patience': 500,
        'neurons_list': [40, 40, 40, 40, 40, 40, 40, 40],
        # 'neurons_list': [40, 40, 40],
        # 'activation': nn.Tanh(),
        'activation': nn.Tanh(),
        'shuffle_train': True,
        'visualize': True,
        'fixed_w': False,
        'update_freq': 50, #25
        'init_w': [2.0, 2.0, 2.0],
        # 'init_w': [1.0, 1.0, 1.0],
        'max_w': 100.0,
        'min_w': 1.0,
        'show_points': True
    }

    data = pd.read_csv('/home/iagkilam/DeepCFD/applications/cylinderFlow2DCellData.csv')
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
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    n_samples = 25000

    samples = lhs(2, samples=n_samples)
    x_coord = xmin + (xmax - xmin) * samples[:, 0]
    y_coord = ymin + (ymax - ymin) * samples[:, 1]
    coordinates = np.column_stack((x_coord, y_coord))

    center = np.array([0.0, 0.0])
    radius = 0.202845
    theta = np.linspace(0, 2 * np.pi, n_samples)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    points_cylinder = np.column_stack((circle_x, circle_y))

    cylinder_path = Path(points_cylinder)
    mask = ~cylinder_path.contains_points(coordinates)
    coordinates = coordinates[mask]

    if options["show_points"]:
        fig = plt.figure()
        ax = plt.subplot(111)
        train_plot = plt.scatter(coordinates[:, 0], coordinates[:, 1], color="red"),
        test_plot = plt.scatter(points_f[:, 0], points_f[:, 1], color="blue"),
        ax.legend((train_plot, test_plot),
                  ('Train points', 'Test points',),
                  loc='lower left',
                  ncol=2,
                  bbox_to_anchor=(0.0, 1.0),
                  fontsize=16)
        plt.show()

    device = options["device"]
    neurons_list = options["neurons_list"]
    model = options["net"](
        2,
        3,
        neurons_list=neurons_list,
        activation=options["activation"]
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

    train_graph = create_graph(train_data[0], train_data[1])
    train_edge_index = train_graph.edge_index

    val_graph = create_graph(test_data[0], test_data[1])
    val_edge_index = val_graph.edge_index

    torch.manual_seed(0)

    # Define optimizers
    optimizerAdam = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=0.05
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
                if isinstance(submodule, torch_geometric.nn.Linear):
                    weight_grads = torch.autograd.grad(loss, submodule.weight, retain_graph=True, allow_unused=True)[0]
                    # bias_grads = torch.autograd.grad(loss, submodule.bias, retain_graph=True, allow_unused=True)[0]
                    # grads.extend([g.view(-1) for g in [weight_grads, bias_grads] if g is not None])
                    grads.extend([g.view(-1) for g in [weight_grads] if g is not None])
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

        x = xInside[:, 0:1]
        y = xInside[:, 1:2]
        graph = create_graph(x, y)
        x = graph.x.to(device)
        edge_index_batch = graph.edge_index.to(device)
        y = graph.y.to(device)
        
        x.requires_grad = True
        y.requires_grad = True
        
        out = model(torch.cat([x, y], dim=1), edge_index_batch.T)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        xS = model.points_surface[:, 0:1]
        yS = model.points_surface[:, 1:2]
        graphS = create_graph(xS, yS)
        edgeS_index_batch = graphS.edge_index.to(device)
        outS = model(torch.cat([xS, yS], dim=1), edgeS_index_batch.T)
        uS = outS[:, 0:1]
        vS = outS[:, 1:2]
        pS = outS[:, 2:3]

        xL = model.points_inlet[:, 0:1]
        yL = model.points_inlet[:, 1:2]
        graphL = create_graph(xL, yL)
        edgeL_index_batch = graphL.edge_index.to(device)
        outL = model(torch.cat([xL, yL], dim=1), edgeL_index_batch.T)
        uL = outL[:, 0:1]
        vL = outL[:, 1:2]
        pL = outL[:, 2:3]

        xR = model.points_outlet[:, 0:1]
        yR = model.points_outlet[:, 1:2]
        graphR = create_graph(xR, yR)
        edgeR_index_batch = graphR.edge_index.to(device)
        outR = model(torch.cat([xR, yR], dim=1), edgeR_index_batch.T)
        uR = outR[:, 0:1]
        vR = outR[:, 1:2]
        pR = outR[:, 2:3]

        xT = model.points_top[:, 0:1]
        yT = model.points_top[:, 1:2]
        graphT = create_graph(xT, yT)
        edgeT_index_batch = graphT.edge_index.to(device)
        outT = model(torch.cat([xT, yT], dim=1), edgeT_index_batch.T)
        uT = outT[:, 0:1]
        vT = outT[:, 1:2]
        pT = outT[:, 2:3]

        xB = model.points_bottom[:, 0:1]
        yB = model.points_bottom[:, 1:2]
        graphB = create_graph(xB, yB)
        edgeB_index_batch = graphB.edge_index.to(device)
        outB = model(torch.cat([xB, yB], dim=1), edgeB_index_batch.T)
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
            float(calc_r2(scope["dataset"], scope["model"], scope["training"], scope["device"])),
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
    
    tbPath = "/home/iagkilam/DeepCFD/tensorboard_gnn"
    # tbPath = "../tensorboard_gnn/"
    # if not os.path.exists(tbPath):
    #     os.makedirs(tbPath)

    writerAdam = tb.SummaryWriter(log_dir=tbPath + options["output"] + "Adam", comment="Adam_training_gnn")
    # # Training model with Adam
    gnnAdam, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
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
        writer=writerAdam,
        **validation_metrics
    )
    writerAdam.close()
    state_dict = gnnAdam.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]
    
    modelPath = "/home/iagkilam/DeepCFD/models_gnn"
    # modelPath = "../models_gnn/"
    
    # if not os.path.exists(modelPath):
    #     os.makedirs(modelPath)
        
    torch.save(state_dict, modelPath + "/Adam_" + options["output"])

    # torch.save(state_dict, options["output"])
    
    writerFullbatch = tb.SummaryWriter(log_dir=tbPath + options["output"] + "Fullbatch", comment="Fullbatch_training")
    # Training model with L-FBGS
    gnnModel, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
        gnnAdam,
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
    state_dict = gnnModel.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, modelPath + "/Fullbatch_"  + options["output"])


    if (options["visualize"]):
        # Evaluate on the mesh cell points (NOT USED IN TRAINING)
        test_x = data[["x", "y"]].values
        test_x = torch.FloatTensor(test_x)
        sample_y = data[["U:0", "U:1", "p"]].values
        sample_y = torch.FloatTensor(sample_y)
        test_graph = create_graph(test_x, sample_y, n_neighbours=5)
        # test_graph = create_graph(test_x, sample_y, n_neighbours=5)
          
        out_ad = gnnAdam(test_x.to(options["device"]), edge_index=(test_graph.edge_index.T).to(options["device"]))
        out_fb = gnnModel(test_x.to(options["device"]), edge_index=(test_graph.edge_index.T).to(options["device"]))
        sample_x = test_x.cpu().detach().numpy()
        sample_y = sample_y.cpu().detach().numpy()
        out_y_adam = out_ad.cpu().detach().numpy()
        out_y_fb = out_fb.cpu().detach().numpy()
        resultPath = "/home/iagkilam/DeepCFD/results"
        # resultPath = "../results_gnn/"
        # if not os.path.exists(resultPath):
        #     os.makedirs(resultPath)
        
        visualize2DNavierStokes(sample_y, out_y_adam, sample_x, savePath=resultPath + "/Adam_" + options["output"] + ".png")
        visualize2DNavierStokes(sample_y, out_y_fb, sample_x, savePath=resultPath + "/Fullbatch_" + options["output"] + ".png")
