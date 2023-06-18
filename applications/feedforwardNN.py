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
    model_input = "xMLP.pkl"
    model_output = "yMLP.pkl"
    output = "mymodel.pt"
    learning_rate = 0.001
    epochs = 2000
    batch_size = 32
    patience = 500
    visualize = False

    try:
        opts, args = getopt.getopt(
            argv, "hd:n:mi:mo:o:k:f:l:e:b:p:v",
            [
                "device=",
                "model-input=",
                "model-output=",
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
        elif opt in ("-mi", "--model-input"):
            model_input = arg
        elif opt in ("-mo", "--model-output"):
            model_output = arg
        elif opt in ("-k", "--kernel-size"):
            kernel_size = int(arg)
        elif opt in ("-f", "--filters"):
            filters = [int(x) for x in arg.split(',')]
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
        'model_input': model_input,
        'model_output': model_output,
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

    x = pickle.load(open(options["model_input"], "rb"))
    y = pickle.load(open(options["model_output"], "rb"))
    neurons_list = [80, 80, 80, 80]
    model = options["net"](
        2,
        3,
        neurons_list=neurons_list,
        activation=nn.ReLU(),
        # activation=nn.Tanh(),
        normalize_weights=False,
        normalize_batch=False
    )

    # Pre-process data
    batch = len(x)  # len(y.tensors)
    channels_weights = [np.sum(sample, axis=0)/sample.shape[0]
                        for idx, sample in enumerate(y)]
    channels_weights = sum([np.sqrt(sample ** 2)
                           for sample in channels_weights]) / batch
    channels_weights = channels_weights.reshape((1, 3, 1))
    print(channels_weights)
    print(channels_weights.shape)

    channels_weights = torch.FloatTensor(
        channels_weights).to(options["device"])

    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    x = [torch.from_numpy(array) for array in x]
    y = [torch.from_numpy(array) for array in y]

    # Spliting dataset into 70% train and 30% test
    train_data, test_data = split_tensors(x, y, ratio=0.7)

    train_dataset = ModifiedTensorDataset(train_data[0], train_data[1])
    test_dataset = ModifiedTensorDataset(test_data[0], test_data[1])
    test_x, test_y = test_dataset.tensors

    torch.manual_seed(0)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=0.005
    )

    config = {}
    train_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    test_mse_curve = []
    train_ux_curve = []
    test_ux_curve = []
    train_uy_curve = []
    test_uy_curve = []
    train_p_curve = []
    test_p_curve = []

    def after_epoch(scope):
        train_loss_curve.append(scope["train_loss"])
        test_loss_curve.append(scope["val_loss"])
        train_mse_curve.append(scope["train_metrics"]["mse"])
        test_mse_curve.append(scope["val_metrics"]["mse"])
        train_ux_curve.append(scope["train_metrics"]["ux"])
        test_ux_curve.append(scope["val_metrics"]["ux"])
        train_uy_curve.append(scope["train_metrics"]["uy"])
        test_uy_curve.append(scope["val_metrics"]["uy"])
        train_p_curve.append(scope["train_metrics"]["p"])
        test_p_curve.append(scope["val_metrics"]["p"])

    def loss_func(model, batch):
        x, y = batch
        output = model(x)

        lossu = ((output[:, 0] - y[:, 0]) ** 2).reshape((output.shape[0], 1))
        lossv = ((output[:, 1] - y[:, 1]) ** 2).reshape((output.shape[0], 1))
        lossp = torch.abs((output[:, 2] - y[:, 2])).reshape((output.shape[0], 1))
        loss = (torch.cat((lossu, lossv, lossp), 1)) / channels_weights.reshape((1, 3))

        return torch.sum(loss), output

    # Training model
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(
        model.double(),
        loss_func,
        train_dataset,
        test_dataset,
        optimizer,
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        device=options["device"],
        m_mse_name="Total MSE",
        m_mse_on_batch=lambda scope:
            float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
        m_mse_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        m_ux_name="Ux MSE",
        m_ux_on_batch=lambda scope:
            float(torch.sum((scope["output"][:, 0] -
                             scope["batch"][1][:, 0]) ** 2)),
        m_ux_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
            m_uy_name="Uy MSE",
        m_uy_on_batch=lambda scope:
            float(torch.sum((scope["output"][:, 1] -
                             scope["batch"][1][:, 1]) ** 2)),
        m_uy_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
            m_p_name="p MSE",
        m_p_on_batch=lambda scope:
            float(torch.sum((scope["output"][:, 2] -
                             scope["batch"][1][:, 2]) ** 2)),
        m_p_on_epoch=lambda scope:
            sum(scope["list"]) /
            len(scope["dataset"]), patience=options["patience"], after_epoch=after_epoch
    )

    state_dict = DeepCFD.state_dict()
    state_dict["neurons_list"] = neurons_list
    state_dict["architecture"] = options["net"]

    torch.save(state_dict, options["output"])

    if (options["visualize"]):
        idx = 0
        out = DeepCFD(test_x[idx].to(options["device"]))
        sample_x = test_x[idx].cpu().detach().numpy()
        out_y = out.cpu().detach().numpy()
        sample_y = test_y[idx].cpu().detach().numpy()
        visualizeScatter(sample_y, out_y, sample_x, savePath="./run.png")

# %run applications/feedforwardNN.py  --model-input ./xMLP.pkl --model-output ./yMLP.pkl  --epochs 100 --batch-size 1024 --visualize True
