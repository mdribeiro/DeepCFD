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
import torch.utils.tensorboard as tb
from sklearn import neighbors
from deepcfd.models.ExperimentalModels import FeedForwardNN, GNNRegression
net = GNNRegression


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
        'device': "cpu",  # "cuda",
        'output': "mymodel.pt",
        'net': net,
        # 'learning_rate': 1e-4,
        'learning_rate': 1e-3,
         # 'epochs': [40000, 1000],
         # 'epochs': [10000, 1000],
         # 'epochs': [1000, 100],
         'epochs': [100, 10],
        # 'batch_size': 1024,
        'batch_size': 1024,
        'patience': 500,
        # 'neurons_list': [40, 40, 40, 40, 40, 40, 40, 40],
        'neurons_list': [40, 40, 40],
        # 'activation': nn.Tanh(),
        'activation': nn.ReLU(),
        'shuffle_train': True,
        'visualize': True,
        'show_points': True,
    }

    data = pd.read_csv('/home/dias_ma/Github/cylinderFlow2DCellData.csv')
    cylinderFlow2D = data[["x", "y", "U:0", "U:1", "p", "patchIDs"]].values

    points_f, output_f = cylinderFlow2D[:, 0:2], cylinderFlow2D[:, 2:5]
    patch_ids = cylinderFlow2D[:, 5:6]

    n_samples = len(points_f)

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

    def loss_func(model, batch):
        x_batch, y_batch = batch
        graph = create_graph(x_batch, y_batch)
        x_batch = graph.x.to(device)
        edge_index_batch = graph.edge_index.to(device)
        y_batch = graph.y.to(device)

        x = x_batch[:, 0:1]
        y = x_batch[:, 1:2]
        out = model(torch.cat([x, y], dim=1), edge_index_batch.T)

        return torch.sum((y_batch - out)**2), out

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

    writerAdam = tb.SummaryWriter(comment="gnn_training")
    # # Training model with Adam
    gnnModel, train_metrics, train_loss, test_metrics, test_loss, last_epoch = train_model(
        model,
        loss_func,
        train_dataset,
        test_dataset,
        optimizerAdam,
        epochs=options["epochs"][0],
        batch_size=options["batch_size"],
        device=options["device"],
        shuffle_train=options["shuffle_train"],
        writer=writerAdam,
        **validation_metrics
    )
    writerAdam.close()
    state_dict = gnnModel.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, options["output"])

    if (options["visualize"]):
        # Evaluate on the mesh cell points (NOT USED IN TRAINING)
        test_x = data[["x", "y"]].values
        test_x = torch.FloatTensor(test_x)
        sample_y = data[["U:0", "U:1", "p"]].values
        sample_y = torch.FloatTensor(sample_y)
        test_graph = create_graph(test_x, sample_y, n_neighbours=5)
        # test_graph = create_graph(test_x, sample_y, n_neighbours=5)
        out = gnnModel(test_x.to(options["device"]), edge_index=test_graph.edge_index.T)
        sample_x = test_x.cpu().detach().numpy()
        sample_y = sample_y.cpu().detach().numpy()
        out_y = out.cpu().detach().numpy()
        visualize2DNavierStokes(sample_y, out_y, sample_x, savePath="./runGNN.png")
