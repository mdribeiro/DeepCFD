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
import torch.utils.tensorboard as tb

from deepcfd.models.ExperimentalModels import FeedForwardNN
net = FeedForwardNN


def parseOpts(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = "Re_100_1000_wd_rs_model_RBA.pt"
    # output = "trial.pt"
    learning_rate = 0.001
    epochs = 500
    batch_size = 32
    patience = 500
    visualize = True

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
    numRe = 1000
    xBounds = (-1., 1.)
    tBounds = (0., 1.)
    ReBounds = (100., 1000.) #np.pi / 0.01 #decrease Re space, single number to check- don't use scaling
    scaleRe = 1000.
    x, xBdr, xInitial = CreateCollocationPoints(xBounds, tBounds, numSamples, numRe, ReBounds)

    neurons_list = [10, 10, 10, 10, 10, 10]
    model = options["net"](
        3,
        1,
        neurons_list=neurons_list,
        activation=nn.Tanh(),
        normalize_weights=False,
        normalize_batch=False
    )

    # model.nu = 0.01 / np.pi

    y = Burgers(x[:, 0], x[:, 1], x[:, 2]).reshape((numSamples, 1))

    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    idx_residual = np.linspace(0, len(x) - 1, len(x), dtype=int)
    x = np.hstack((x, idx_residual[:, np.newaxis]))
    
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    model.lamb = torch.FloatTensor(torch.ones((len(x), 1)))
    model.lamb.requires_grad = True
    model.lamb.to('cuda')
    model.lamb_factor = 0.999
    model.lamb_lr = 1e-2

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
        weight_decay=0.005 #remove, set = 0.0
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
        num = x/td

        t0 = torch.exp(Re/8)
        # td = td.detach().clone().to(torch.device('cpu'))
        den = 1 + torch.sqrt(td/t0) * torch.exp(Re * (x**2/(4*td)))

        u_xt = torch.nan_to_num(num/den)
        # u_xt = num/den

        return u_xt
    
    def scaleReNum(Re):
        return Re/scaleRe

    def loss_func(model, batch): #debug line by line
        xInside, _, xBdr, xInitial = batch
        xInside.requires_grad = True
        xBdr.requires_grad = True
        xInitial.requires_grad = True

        x = xInside[:, 0:1]
        t = xInside[:, 1:2]
        us_Re = xInside[:, 2:3]
        ids = xInside[:, 2:3].long()
        s_Re = scaleReNum(xInside[:, 2:3])
        u = model(torch.cat([x, t, s_Re], dim=1))
        xB = xBdr[:, 0:1]
        tB = xBdr[:, 1:2]
        outBdr = model(torch.cat([xB, tB, s_Re], dim=1))
        xI = xInitial[:, 0:1]
        tI = xInitial[:, 1:2]
        outInitial = model(torch.cat([xI, tI, s_Re], dim=1))

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

        loss_f = u_t + u * u_x - (1/us_Re) * u_xx
        
        lamb_new = model.lamb.detach().clone()
        lamb_new = lamb_new.cuda()

        # with torch.no_grad():
        lamb_new[ids[:, 0]] = model.lamb_factor * lamb_new[ids[:, 0]] + model.lamb_lr * torch.abs(loss_f) / torch.max(torch.abs(loss_f))
        loss_f = lamb_new[ids[:, 0]] * loss_f
        model.lamb = lamb_new

        # Re = torch.tensor(1/model.nu, requires_grad=True)

        yBdr = BurgersExact(xBdr[:, 0:1], xBdr[:, 1:2], us_Re)
        yInitial = BurgersExact(xInitial[:, 0:1], xInitial[:, 1:2], us_Re)

        loss_Bdr = yBdr - outBdr
        loss_Initial = yInitial - outInitial

        loss = loss_f**2 + loss_Bdr**2 + loss_Initial**2
        # loss = torch.mean(loss_f**2) + torch.mean(loss_Bdr**2) + torch.mean(loss_Initial**2)

        return torch.sum(loss), u
    
    tbPath = "/home/iagkilam/DeepCFD/burgers_tensorboard/"

    writerAdam = tb.SummaryWriter(log_dir=tbPath + options["output"] + "/Adam", comment="Adam_training_gnn")

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
        writer=writerAdam,
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
    writerAdam.close()
    state_dict_Adam = pinnAdam.state_dict()
    state_dict_Adam["neurons_list"] = neurons_list
    state_dict_Adam["architecture"] = options["net"]
    
    modelPath = "/home/iagkilam/DeepCFD/burgers_models"
    
    torch.save(state_dict_Adam, modelPath + "/Adam_" + options["output"])
    
    resultsPath = "/home/iagkilam/DeepCFD/burgers_results/"
    
    if (options["visualize"]):
        test_Renum = 400
        time_label = [0.25, 0.5, 0.75]
        test_x = torch.linspace(-1, 1, 100).reshape((100, 1))
        test_re = (torch.ones_like(test_x) * test_Renum) #.to(options["device"])
        visualize1DBurgers(time_label, test_x, test_re, options, pinnAdam, BurgersExact, xBounds, tBounds, test_Renum,
                            savePath= resultsPath + options["output"] +"_Adam.png")
    
    writerFullbatch = tb.SummaryWriter(log_dir=tbPath + options["output"] + "/Fullbatch", comment="Adam_training_gnn")
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
        writer=writerFullbatch,
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
    # writerAdam.close()
    writerFullbatch.close()
    state_dict_FB = pinnModel.state_dict()
    # state_dict = pinnAdam.state_dict()
    state_dict_FB["neurons_list"] = neurons_list
    state_dict_FB["architecture"] = options["net"]

    torch.save(state_dict_FB, modelPath + "/Fullbatch_" + options["output"])

    if (options["visualize"]):
        test_Renum = 200
        time_label = [0.25, 0.5, 0.75]
        test_x = torch.linspace(-1, 1, 100).reshape((100, 1))
        test_re = (torch.ones_like(test_x) * test_Renum) #.to(options["device"])
        visualize1DBurgers(time_label, test_x, test_re, options, pinnModel, BurgersExact, xBounds, tBounds, test_Renum, savePath= resultsPath + options["output"] +"_Fullbatch.png")

# %run applications/physicsinformedNN.py --device "gpu" --epochs 75 --batch-size 32 --visualize True
# %run applications/physicsinformedNN.py --device "cpu" --epochs 75 --batch-size 32 --visualize True
