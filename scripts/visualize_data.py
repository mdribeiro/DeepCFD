import os
import re
import time
import deepcfd
import torch
import pickle
import matplotlib
import numpy as np
from deepcfd.models.UNetEx import UNetEx
from matplotlib import pyplot as plt

matplotlib.use('tkagg')

model_filename="checkpoint.pt"
data_x_filename="dataX.pkl"
data_y_filename="dataY.pkl"

index = 200

kernel_size = 5
filters = [8, 16, 32, 32]
bn = False
wn = False

last_mtime = 0
last_saved = 1
current_epoch = 1
extent=[0,256,0,128]

min_u_x = 0
max_u_x = 0.15
min_u_y = -0.045
max_u_y = 0.045
min_u_x_error = 0
max_u_x_error = 0.0185
min_u_y_error = 0
max_u_y_error = 0.0085
min_p = 0
max_p = 0.015
min_p_error = 0
max_p_error = 0.0075

model = UNetEx(
    3,
    3,
    filters=filters,
    kernel_size=kernel_size,
    batch_norm=bn,
    weight_norm=wn
)

x = pickle.load(open(data_x_filename, "rb"))
y = pickle.load(open(data_y_filename, "rb"))

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

truth = y[index:(index+1)].cpu().detach().numpy()
inputs = x[index:(index+1)].cpu().detach().numpy()

plt.figure()
fig = plt.gcf()
fig.set_size_inches(15, 10)
fig.canvas.manager.window.wm_attributes('-topmost', 0)

plt.ion()
plt.show()

def update_plot():

    # Wait until the file is completely written
    while True:
        try:
            with open(model_filename, "r") as f:
                break
        except IOError:
            time.sleep(1)

    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict)
    out = model(x[index:(index+1)]).detach().numpy()
    error = abs(out - truth)

    fig.suptitle('Best at epoch ' + str(last_saved) + ' (current: ' + str(current_epoch) + ')')

    plt.subplot(3, 3, 1)
    plt.ylabel('Ux [m/s]', fontsize=18)
    plt.title('CFD', fontsize=18)
    plt.imshow(truth[0, 0, :, :], cmap='jet', vmin = min_u_x, vmax = max_u_x, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 2)
    plt.title('CNN', fontsize=18)
    plt.imshow(out[0, 0, :, :], cmap='jet', vmin = min_u_x, vmax = max_u_x, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 3)
    plt.title('error', fontsize=18)
    plt.imshow(error[0, 0, :, :], cmap='jet', vmin = min_u_x_error, vmax = max_u_x_error, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 4)
    plt.ylabel('Uy [m/s]', fontsize=18)
    plt.imshow(truth[0, 1, :, :], cmap='jet', vmin = min_u_y, vmax = max_u_y, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 5)
    plt.imshow(out[0, 1, :, :], cmap='jet', vmin = min_u_y, vmax = max_u_y, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 6)
    plt.imshow(error[0, 1, :, :], cmap='jet', vmin = min_u_y_error, vmax = max_u_y_error, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 7)
    plt.ylabel('p [m2/s2]', fontsize=18)
    plt.imshow(truth[0, 2, :, :], cmap='jet', vmin = min_p, vmax = max_p, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 8)
    plt.imshow(out[0, 2, :, :], cmap='jet', vmin = min_p, vmax = max_p, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 9)
    plt.imshow(error[0, 2, :, :], cmap='jet', vmin = min_p_error, vmax = max_p_error, origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')

    plt.draw()
    plt.pause(5)
    #plt.show()

last_saved = current_epoch

while True:
    saved_epochs = []
    with open('log.deepcfd2', 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if line.startswith('Epoch #'):
                current_epoch = int(line.split('Epoch #')[1])
                break
        for line in reversed(lines):
            if line.startswith('Epoch #'):
                last_saved = int(line.split('Epoch #')[1])
            elif line.startswith('Model saved!'):
                last_saved = last_saved - 1
                break

    update_plot()

    time.sleep(5)
