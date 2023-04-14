import matplotlib.pyplot as plt

log_file = "log.deepcfd"
initial_epoch = 20

train_loss = []
train_total_mse = []
train_ux_mse = []
train_uy_mse = []
train_p_mse = []
val_loss = []
val_total_mse = []
val_ux_mse = []
val_uy_mse = []
val_p_mse = []

with open(log_file, 'r') as f:
    for line in f:
        if line.strip().startswith('Train Loss'):
            train_loss.append(float(line.split()[-1]))
        elif line.strip().startswith('Train Total MSE'):
            train_total_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Train Ux MSE'):
            train_ux_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Train Uy MSE'):
            train_uy_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Train p MSE'):
            train_p_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Validation Loss'):
            val_loss.append(float(line.split()[-1]))
        elif line.strip().startswith('Validation Total MSE'):
            val_total_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Validation Ux MSE'):
            val_ux_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Validation Uy MSE'):
            val_uy_mse.append(float(line.split()[-1]))
        elif line.strip().startswith('Validation p MSE'):
            val_p_mse.append(float(line.split()[-1]))

epochs = list(range(1, len(train_loss) + 1))

fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(14, 8))

axs[0].plot(epochs[initial_epoch:], train_loss[initial_epoch:], label='Train Loss')
axs[0].plot(epochs[initial_epoch:], val_loss[initial_epoch:], label='Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(epochs[initial_epoch:], train_total_mse[initial_epoch:], label='Train Total MSE')
axs[1].plot(epochs[initial_epoch:], train_ux_mse[initial_epoch:], label='Train Ux MSE')
axs[1].plot(epochs[initial_epoch:], train_uy_mse[initial_epoch:], label='Train Uy MSE')
axs[1].plot(epochs[initial_epoch:], train_p_mse[initial_epoch:], label='Train p MSE')
axs[1].plot(epochs[initial_epoch:], val_total_mse[initial_epoch:], label='Validation Total MSE')
axs[1].plot(epochs[initial_epoch:], val_ux_mse[initial_epoch:], label='Validation Ux MSE')
axs[1].plot(epochs[initial_epoch:], val_uy_mse[initial_epoch:], label='Validation Uy MSE')
axs[1].plot(epochs[initial_epoch:], val_p_mse[initial_epoch:], label='Validation p MSE')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('MSE')
axs[1].legend()

plt.show()

