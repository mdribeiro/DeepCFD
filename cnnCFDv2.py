#!/usr/bin/env python37
"""
Created on Fri Sep  6 16:20:46 2019

@author: mdias
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
import pickle
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

get_ipython().magic('matplotlib inline')

# Please, download new dataset at  https://drive.google.com/drive/folders/1v-7DQJWBp5QCm10HWZb5v7ua5_00Kt0V?usp=sharing
Xs2 = pickle.load( open( "./XsNew.pkl", "rb" ) )
Ys = pickle.load( open( "./YsNew.pkl", "rb"  ) )

Xs2[np.isnan(Xs2)] = 0
Ys[np.isnan(Ys)] = 0

Xs2 = np.swapaxes(Xs2,1,3)
Xs2 = np.swapaxes(Xs2,2,3)

# !!!!!! In the next four lines (after comment) I'm rejecting channel 1 of input 
# with the SDF from the upper and lower walls. I don't think it helps the model. 
# So the only channels left are 0 (SDF from obstacle), 1 (x), 2 (y)
Xs = np.zeros(( Xs2.shape[0] , Xs2.shape[1] - 1, Xs2.shape[2], Xs2.shape[3]  ))
Xs[:,0,:,:] = Xs2[:,0,:,:]
Xs[:,1,:,:] = Xs2[:,2,:,:]
Xs[:,2,:,:] = Xs2[:,3,:,:]

Ys = np.swapaxes(Ys,1,3)
Ys = np.swapaxes(Ys,2,3)

Xs = Xs[0:900]
Ys = Ys[0:900]

def transf(input):
    mean = np.mean(input)
    std = np.std(input)
    output = (input-mean)/std
    return output, mean, std

def inv_transf(input, mean, std):
    output = input*std+mean
    return output

Xtrain = Xs[0:800]
Ytrain = Ys[0:800]

Xtest = Xs[800:]
Ytest = Ys[800:]

Xtrain[:,0,:,:], m0, s0 = transf(Xtrain[:,0,:,:])
Xtrain[:,1,:,:], m1, s1 = transf(Xtrain[:,1,:,:])
Xtrain[:,2,:,:], m2, s2 = transf(Xtrain[:,2,:,:])

Ytrain[:,0,:,:], m3, s3 = transf(Ytrain[:,0,:,:])
Ytrain[:,1,:,:], m4, s4 = transf(Ytrain[:,1,:,:])
Ytrain[:,2,:,:], m5, s5 = transf(Ytrain[:,2,:,:])


Xtest[:,0,:,:], m6, s6 = transf(Xtest[:,0,:,:])
Xtest[:,1,:,:], m7, s7 = transf(Xtest[:,1,:,:])
Xtest[:,2,:,:], m8, s8 = transf(Xtest[:,2,:,:])

Ytest[:,0,:,:], m9, s9 = transf(Ytest[:,0,:,:])
Ytest[:,1,:,:], m10, s10 = transf(Ytest[:,1,:,:])
Ytest[:,2,:,:], m11, s11 = transf(Ytest[:,2,:,:])



# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##       
        self.conv1 = weight_norm(nn.Conv2d(3, 16, 3, 1, 1))
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = weight_norm(nn.Conv2d(16, 32, 5, 1, 1 ) )
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = weight_norm(nn.Conv2d(32, 64, 9, 1, 1 ) )
        self.conv3_bn = nn.BatchNorm2d(64)
#        self.conv4 = weight_norm(nn.Conv2d(64, 128, 3, 1, 1 ) )
#        self.conv4_bn = nn.BatchNorm2d(128)
        
#        self.t_conv1_1 = weight_norm(nn.ConvTranspose2d(128, 64, 3, 1, 1 ))
#        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.t_conv2_1 = weight_norm(nn.ConvTranspose2d(64, 32, 9, 1, 1 ) )
        self.conv2_1_bn = nn.BatchNorm2d(32)
        self.t_conv3_1 = weight_norm(nn.ConvTranspose2d(32, 13, 5, 1, 1 )  )
        self.conv3_1_bn = nn.BatchNorm2d(13)
        self.t_conv4_1 = weight_norm(nn.ConvTranspose2d(16, 1, 3, 1, 1 ))
        
#        self.t_conv1_2 = weight_norm(nn.ConvTranspose2d(128, 64, 3, 1, 1 ))
#        self.conv1_2_bn = nn.BatchNorm2d(64)
        self.t_conv2_2 = weight_norm(nn.ConvTranspose2d(64, 32, 9, 1, 1 ) )
        self.conv2_2_bn = nn.BatchNorm2d(32)
        self.t_conv3_2 = weight_norm(nn.ConvTranspose2d(32, 13, 5, 1, 1 )  )
        self.conv3_2_bn = nn.BatchNorm2d(13)
        self.t_conv4_2 = weight_norm(nn.ConvTranspose2d(16, 1, 3, 1, 1 ))
        
#        self.t_conv1_3 = weight_norm(nn.ConvTranspose2d(128, 64, 3, 1, 1 ))
#        self.conv1_3_bn = nn.BatchNorm2d(64)
        self.t_conv2_3 = weight_norm(nn.ConvTranspose2d(64, 32, 9, 1, 1 ) )
        self.conv2_3_bn = nn.BatchNorm2d(32)
        self.t_conv3_3 = weight_norm(nn.ConvTranspose2d(32, 13, 5, 1, 1 )  )
        self.conv3_3_bn = nn.BatchNorm2d(13)
        self.t_conv4_3 = weight_norm(nn.ConvTranspose2d(16, 1, 3, 1, 1 ))
        
    def forward(self, x):
        
        x1 = x
        
        # Create spatial latent representation
        x = self.conv1_bn(F.relu(self.conv1(x1)))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.conv3_bn(F.relu(self.conv3(x)))
#        x = self.conv4_bn(F.relu(self.conv4(x)))
        
        # X-component velocity prediction        
        
#        ux = self.conv1_1_bn(F.relu(self.t_conv1_1(x)))
        ux = self.conv2_1_bn(F.relu(self.t_conv2_1(x)))
        ux = self.conv3_1_bn(F.relu(self.t_conv3_1(ux)))
        ux = torch.cat((ux,x1),1)
        ux = self.t_conv4_1(ux)
        
        # Y-component velocity prediction        
        
#        uy = self.conv1_2_bn(F.relu(self.t_conv1_2(x)))
        uy = self.conv2_2_bn(F.relu(self.t_conv2_2(x)))
        uy = self.conv3_2_bn(F.relu(self.t_conv3_2(uy)))
        uy = torch.cat((uy,x1),1)
        uy = self.t_conv4_2(uy)
        
        # Pressure prediction
        
        #p = F.relu(self.t_conv1_3(x))
#        p = self.conv1_3_bn(F.relu(self.t_conv1_3(x)))
        p = self.conv2_3_bn(F.relu(self.t_conv2_3(x)))
        p = self.conv3_3_bn(F.relu(self.t_conv3_3(p)))
        p = torch.cat((p,x1),1)
        p = self.t_conv4_3(p)
                
        return ux, uy, p        
    
  
model = ConvAutoencoder().to(device)    
    
print(model)
print('Number of parameters in model: ' + str(sum(p.numel() for p in model.parameters())))

learning_rate = 1e-5
batch_size =  10  #5  #10 #15. 20
#criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

Xs = torch.from_numpy(Xtrain)
Xs = Variable(Xs).float()

Ys = torch.from_numpy(Ytrain)
Ys = Variable(Ys).float()

Xst = torch.from_numpy(Xtest)
Xst = Variable(Xst).float()

Yst = torch.from_numpy(Ytest)
Yst = Variable(Yst).float()

#torch_dataset = torch.utils.data.TensorDataset(Xs,(Ys[:,0,:,:].reshape((102,1,172,79)))) 
torch_dataset = torch.utils.data.TensorDataset(Xs,Ys) 
dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

torch_dataset_test = torch.utils.data.TensorDataset(Xst,Yst) 
dataloader_test = DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=True)

num_epochs = 1000

#weight = 'checkpoint.pt'
#PATH = '/home/mdias/CFDML/obstacle/cnnCFD/' +  weight
#model.load_state_dict(torch.load(PATH))

# If you want to try with early stopping, uncomment the next two lines
# and the lines at the end of training loop

from pytorchtools import EarlyStopping
early_stopping = EarlyStopping(patience=30, verbose=True)

fac = 1.0
save_loss = []
save_lossv = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_lossv = 0.0
    for step, (batch_x, batch_y) in enumerate(dataloader): 

        b_x = Variable(batch_x).to(device)        
        b_y = Variable(batch_y).to(device)
        
        b_lim = int(0.8*batch_size)
        x_tr = b_x[:b_lim]
        x_val = b_x[b_lim:]
        
        y_tr = b_y[:b_lim]
        y_val = b_y[b_lim:]
        
        ux_tr, uy_tr, p_tr = model(x_tr)     # input x and predict based on x
        y1_tr = y_tr[:,0,:,:].reshape(( y_tr.shape[0], 1 , y_tr.shape[2], y_tr.shape[3] ))
        y2_tr = y_tr[:,1,:,:].reshape(( y_tr.shape[0], 1 , y_tr.shape[2], y_tr.shape[3] ))
        y3_tr = y_tr[:,2,:,:].reshape(( y_tr.shape[0], 1 , y_tr.shape[2], y_tr.shape[3] ))
        
        ux_val, uy_val, p_val = model(x_val)     # input x and predict based on x
        y1_val = y_val[:,0,:,:].reshape(( y_val.shape[0], 1 , y_val.shape[2], y_val.shape[3] ))
        y2_val = y_val[:,1,:,:].reshape(( y_val.shape[0], 1 , y_val.shape[2], y_val.shape[3] ))
        y3_val = y_val[:,2,:,:].reshape(( y_val.shape[0], 1 , y_val.shape[2], y_val.shape[3] ))

        # Train loss
        loss = fac*(criterion(ux_tr, y1_tr) + criterion(uy_tr, y2_tr)) + criterion(p_tr, y3_tr)
        
        # Validation loss
        lossv = fac*(criterion(ux_val, y1_val) + criterion(uy_val, y2_val)) + criterion(p_val, y3_val)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # print statistics
        running_loss += loss.item()
        running_lossv += lossv.item()
        #if step % 10 == 9:    # print every 10 mini-batches
        if (step+1) % 10 == 0:
           print('[%d, %5d] loss: %.8f  lr: %.8f' % (epoch + 1, step + 1, running_loss / b_lim, current_lr))
           save_loss.append(running_loss / b_lim)
           save_lossv.append(running_lossv / ( batch_size - b_lim))
           running_loss = 0.0
           running_lossv = 0.0
           
    plt.plot(save_loss,'-b',linewidth=2.0, label='Train')
    plt.plot(save_lossv,'-k',linewidth=2.0, label='Validation')
    plt.show()               
    scheduler.step(metrics=lossv.item())    
    early_stopping(loss.item(), model)        
    if early_stopping.early_stop:
        print("Early stopping")
        break
          
torch.save(model.state_dict(), './weightsCFD12.pth')
save_loss = np.asarray(save_loss)
pickle.dump( save_loss, open( "save_lossCFD12.pkl", "wb" ) )
save_lossv = np.asarray(save_lossv)
pickle.dump( save_lossv, open( "save_loss_valCFD12.pkl", "wb" ) )

#Xtest = torch.from_numpy(Xtrain)
Xtest = torch.from_numpy(Xtest)
Xtest = Variable(Xtest).float()
Xtest = Xtest.to(device)

prediction = model(Xtest[0:50])
#pred = prediction.cpu().detach().numpy()
pred1 = prediction[0].cpu().detach().numpy()
pred2 = prediction[1].cpu().detach().numpy()
pred3 = prediction[2].cpu().detach().numpy()

pred1 = inv_transf(pred1,m9,s9)
pred2 = inv_transf(pred2,m10,s10)
pred3 = inv_transf(pred3,m11,s11)

pickle.dump( pred1, open( "pred1CFD12.pkl", "wb" ) )
pickle.dump( pred2, open( "pred2CFD12.pkl", "wb" ) )
pickle.dump( pred3, open( "pred3CFD12.pkl", "wb" ) )

Xs = Xtest
Ys = Ytest

Xs = Xs.cpu().detach().numpy()
Xs[:,0,:,:] = inv_transf(Xs[:,0,:,:],m6,s6)
Xs[:,1,:,:] = inv_transf(Xs[:,1,:,:],m7,s7)
Xs[:,2,:,:] = inv_transf(Xs[:,2,:,:],m8,s8)

Ys[:,0,:,:] = inv_transf(Ys[:,0,:,:],m9,s9)
Ys[:,1,:,:] = inv_transf(Ys[:,1,:,:],m10,s10)
Ys[:,2,:,:] = inv_transf(Ys[:,2,:,:],m11,s11)

sample = 3
index = np.argwhere(  Xs[sample,0,:,:] < -40   )
error1 =  np.abs(  pred1[sample,0,:,:] - Ys[sample,0,:,:] )
error2 =  np.abs(  pred2[sample,0,:,:] - Ys[sample,1,:,:] )
error3 =  np.abs(  pred3[sample,0,:,:] - Ys[sample,2,:,:] )

pred1 = pred1[sample,0,:,:]
pred2 = pred2[sample,0,:,:]
pred3 = pred3[sample,0,:,:]

pred1[index[:,0],index[:,1]] = 0
pred2[index[:,0],index[:,1]] = 0
pred3[index[:,0],index[:,1]] = 0

error1[index[:,0],index[:,1]] = 0
error2[index[:,0],index[:,1]] = 0
error3[index[:,0],index[:,1]] = 0

plt.figure()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.subplot(3, 3, 1)
minp = np.min(Ys[sample,0,:,:])
maxp = np.max(Ys[sample,0,:,:])
plt.title('CFD', fontsize=18) 
plt.imshow( np.transpose(  Ys[sample,0,:,:]) , cmap='jet', vmin = minp, vmax = maxp)
plt.colorbar(orientation='horizontal')
plt.ylabel('Ux', fontsize=18)
plt.subplot(3, 3, 2)
plt.title('CNN', fontsize=18) 
plt.imshow( np.transpose(  pred1), cmap='jet', vmin = minp, vmax = maxp)
plt.colorbar(orientation='horizontal')
plt.subplot(3, 3, 3)
plt.title('Error', fontsize=18) 
plt.imshow( np.transpose( error1  ), cmap='jet', vmin = 0, vmax = 0.015)
plt.colorbar(orientation='horizontal')

plt.subplot(3, 3, 4)
minp = np.min(Ys[sample,1,:,:])
maxp = np.max(Ys[sample,1,:,:])
plt.imshow( np.transpose(  Ys[sample,1,:,:]), cmap='jet', vmin = minp, vmax = maxp)
plt.colorbar(orientation='horizontal')
plt.ylabel('Uy', fontsize=18)
plt.subplot(3, 3, 5)
plt.imshow( np.transpose(  pred2), cmap='jet', vmin = minp, vmax = maxp)
plt.colorbar(orientation='horizontal')
plt.subplot(3, 3, 6)
plt.imshow( np.transpose( error2  ), cmap='jet', vmin = 0, vmax = 0.015)
plt.colorbar(orientation='horizontal')

plt.subplot(3, 3, 7)
minp = np.min(Ys[sample,2,:,:])
maxp = np.max(Ys[sample,2,:,:])
plt.imshow( np.transpose(  Ys[sample,2,:,:]), cmap='jet', vmin = minp, vmax = maxp)
plt.colorbar(orientation='horizontal')
plt.ylabel('p', fontsize=18)
plt.subplot(3, 3, 8)
plt.imshow( np.transpose(  pred3), cmap='jet', vmin = minp, vmax = maxp)
plt.colorbar(orientation='horizontal')
plt.subplot(3, 3, 9)
plt.imshow( np.transpose( error3  ), cmap='jet', vmin = 0, vmax = 0.015)
plt.colorbar(orientation='horizontal')
plt.tight_layout()
plt.show()