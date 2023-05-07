import copy
import torch
from .pytorchtools import EarlyStopping


def generate_metrics_list(metrics_def):
    list = {}
    for name in metrics_def.keys():
        list[name] = []
    return list


def epoch(scope, loader, on_batch=None, training=False):
    model = scope["model"]
    optimizer = scope["optimizer"]
    loss_func = scope["loss_func"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)
    scope["loader"] = loader

    metrics_list = generate_metrics_list(metrics_def)
    total_loss = 0
    if training:
        model.train()
    else:
        model.eval()
    for tensors in loader:
        if "process_batch" in scope and scope["process_batch"] is not None:
            tensors = scope["process_batch"](tensors)
        if "device" in scope and scope["device"] is not None:
            tensors = [tensor.to(scope["device"]) for tensor in tensors]
        loss, output = loss_func(model, tensors)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        scope["batch"] = tensors
        scope["loss"] = loss
        scope["output"] = output
        scope["batch_metrics"] = {}
        for name, metric in metrics_def.items():
            value = metric["on_batch"](scope)
            scope["batch_metrics"][name] = value
            metrics_list[name].append(value)
        if on_batch is not None:
            on_batch(scope)
    scope["metrics_list"] = metrics_list
    metrics = {}
    for name in metrics_def.keys():
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)
    return total_loss, metrics


def train(scope, train_dataset, val_dataset, patience=10, batch_size=256, print_function=print, eval_model=None,
          on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None):

    early_stopping = EarlyStopping(patience, verbose=True)

    epochs = scope["epochs"]
    model = scope["model"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)

    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_val_metrics"] = None
    scope["best_val_loss"] = float("inf")
    scope["best_model"] = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    skips = 0
    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        print_function("Epoch #" + str(epoch_id), flush=True)
        # Training
        scope["dataset"] = train_dataset
        train_loss, train_metrics = epoch(scope, train_loader, on_train_batch, training=True)
        scope["train_loss"] = train_loss
        scope["train_metrics"] = train_metrics
        print_function("\tTrain Loss = " + str(train_loss), flush=True)
        for name in metrics_def.keys():
            print_function("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]), flush=True)
        if on_train_epoch is not None:
            on_train_epoch(scope)
        del scope["dataset"]
        # Validation
        scope["dataset"] = val_dataset
        with torch.no_grad():
            val_loss, val_metrics = epoch(scope, val_loader, on_val_batch, training=False)
        scope["val_loss"] = val_loss
        scope["val_metrics"] = val_metrics
        print_function("\tValidation Loss = " + str(val_loss), flush=True)
        for name in metrics_def.keys():
            print_function("\tValidation " + metrics_def[name]["name"] + " = " + str(val_metrics[name]), flush=True)
        if on_val_epoch is not None:
            on_val_epoch(scope)
        del scope["dataset"]
        # Selection
        is_best = None
        if eval_model is not None:
            is_best = eval_model(scope)
        if is_best is None:
            is_best = val_loss < scope["best_val_loss"]
        if is_best:
            scope["best_train_metric"] = train_metrics
            scope["best_train_loss"] = train_loss
            scope["best_val_metrics"] = val_metrics
            scope["best_val_loss"] = val_loss
            scope["best_model"] = copy.deepcopy(model)
            print_function("Model saved!", flush=True)
            skips = 0
        else:
            skips += 1
        if after_epoch is not None:
            after_epoch(scope)
        early_stopping(val_loss, scope["best_model"])
        if early_stopping.early_stop:
            print_function("Early stopping", flush=True)
            break

    return scope["best_model"], scope["best_train_metric"], scope["best_train_loss"],\
           scope["best_val_metrics"], scope["best_val_loss"]


def train_model(model, loss_func, train_dataset, val_dataset, optimizer, process_batch=None, eval_model=None,
                on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None,
                epochs=100, batch_size=256, patience=10, device=0, **kwargs):
    model = model.to(device)
    scope = {}
    scope["model"] = model
    scope["loss_func"] = loss_func
    scope["train_dataset"] = train_dataset
    scope["val_dataset"] = val_dataset
    scope["optimizer"] = optimizer
    scope["process_batch"] = process_batch
    scope["epochs"] = epochs
    scope["batch_size"] = batch_size
    scope["device"] = device
    metrics_def = {}
    names = []
    for key in kwargs.keys():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])
    for name in names:
        if "m_" + name + "_name" in kwargs and "m_" + name + "_on_batch" in kwargs and "m_" + name + "_on_epoch" in kwargs:
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                "on_batch": kwargs["m_" + name + "_on_batch"],
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
            }
        else:
            print("Warning: " + name + " metric is incomplete!")
    scope["metrics_def"] = metrics_def
    return train(scope, train_dataset, val_dataset, eval_model=eval_model, on_train_batch=on_train_batch,
           on_val_batch=on_val_batch, on_train_epoch=on_train_epoch, on_val_epoch=on_val_epoch, after_epoch=after_epoch,
           batch_size=batch_size, patience=patience)
