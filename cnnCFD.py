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

Xs2 = pickle.load( open( "./Xs.pkl", "rb" ) )
Ys = pickle.load( open( "./Ys.pkl", "rb"  ) )

Xs2 =  np.nan_to_num(Xs2)
Ys =  np.nan_to_num(Ys)

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

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##       
        self.conv1 = weight_norm(nn.Conv2d(3, 16, 3, 1, 1)  )
        #self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = weight_norm(nn.Conv2d(16, 32, 3, 1, 1 ) )
        #self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = weight_norm(nn.Conv2d(32, 64, 3, 1, 1 ) )
        #self.conv3_bn = nn.BatchNorm2d(64)
        #self.conv4 = weight_norm(nn.Conv2d(64, 128, 3, 1, 1 ) )
        #self.conv4_bn = nn.BatchNorm2d(128)
        
        #self.fc = nn.Linear(64*128*3,1)
        
        #self.t_conv1_1 = weight_norm(nn.ConvTranspose2d(128, 64, 3, 1, 1 ))
        #self.conv1_1_bn = nn.BatchNorm2d(64)
        self.t_conv2_1 = weight_norm(nn.ConvTranspose2d(64, 32, 3, 1, 1 ) )
        #self.conv2_1_bn = nn.BatchNorm2d(32)
        self.t_conv3_1 = weight_norm(nn.ConvTranspose2d(32, 13, 3, 1, 1 )  )
        #self.conv3_1_bn = nn.BatchNorm2d(13)
        self.t_conv4_1 = weight_norm(nn.ConvTranspose2d(16, 1, 3, 1, 1 )  )
        
        #self.t_conv1_2 = weight_norm(nn.ConvTranspose2d(128, 64, 3, 1, 1 ) )
        #self.conv1_2_bn = nn.BatchNorm2d(64)
        self.t_conv2_2 = weight_norm(nn.ConvTranspose2d(64, 32, 3, 1, 1 ) )
        #self.conv2_2_bn = nn.BatchNorm2d(32)
        self.t_conv3_2 = weight_norm(nn.ConvTranspose2d(32, 13, 3, 1, 1 ) ) 
        #self.conv3_2_bn = nn.BatchNorm2d(13)
        self.t_conv4_2 = weight_norm(nn.ConvTranspose2d(16, 1, 3, 1, 1 )  )
        
        #self.t_conv1_3 = weight_norm(nn.ConvTranspose2d(128, 64, 3,1, 1 )  )
        #self.conv1_3_bn = nn.BatchNorm2d(64)
        self.t_conv2_3 = weight_norm(nn.ConvTranspose2d(64, 32, 3, 1, 1 ) )
        #self.conv2_3_bn = nn.BatchNorm2d(32)
        self.t_conv3_3 = weight_norm(nn.ConvTranspose2d(32, 13, 3, 1, 1 ) ) 
        #self.conv3_3_bn = nn.BatchNorm2d(13)
        self.t_conv4_3 = weight_norm(nn.ConvTranspose2d(16, 1, 3, 1, 1 )  )
        
    def forward(self, x):
        
        x1 = x
        
        # Create spatial latent representation
        #x = self.conv1_bn(F.relu(self.conv1(x1)))
        #x = self.conv2_bn(F.relu(self.conv2(x)))
        #x = self.conv3_bn(F.relu(self.conv3(x)))
        #x = self.conv4_bn(F.relu(self.conv4(x)))
        x = F.relu(self.conv1(x1))        
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x))    
        #x = F.relu(self.conv4(x))        
        
        #x = F.relu(self.fc(x))
        
        # X-component velocity prediction        
        
        #ux = F.relu(self.t_conv1_1(x))
        #ux = self.conv1_1_bn(F.relu(self.t_conv1_1(x)))
        #ux = self.conv2_1_bn(F.relu(self.t_conv2_1(ux)))
        #ux = self.conv3_1_bn(F.relu(self.t_conv3_1(ux)))
        ux = F.relu(self.t_conv2_1(x))        
        ux = F.relu(self.t_conv3_1(ux))        
        ux = torch.cat((ux,x1),1)
        ux = self.t_conv4_1(ux)
        
        # Y-component velocity prediction        
        
        #uy = F.relu(self.t_conv1_2(x))
        #uy = self.conv1_2_bn(F.relu(self.t_conv1_2(x)))
        #uy = self.conv2_2_bn(F.relu(self.t_conv2_2(uy)))
        #uy = self.conv3_2_bn(F.relu(self.t_conv3_2(uy)))
        uy = F.relu(self.t_conv2_2(x))        
        uy = F.relu(self.t_conv3_2(uy))        
        uy = torch.cat((uy,x1),1)
        uy = self.t_conv4_2(uy)
        
        # Pressure prediction
        
        #p = F.relu(self.t_conv1_3(x))
        #p = self.conv1_3_bn(F.relu(self.t_conv1_3(x)))
        #p = self.conv2_3_bn(F.relu(self.t_conv2_3(p)))
        #p = self.conv3_3_bn(F.relu(self.t_conv3_3(p)))
        p = F.relu(self.t_conv2_3(x))        
        p = F.relu(self.t_conv3_3(p))        
        p = torch.cat((p,x1),1)
        p = self.t_conv4_3(p)
                
        return ux, uy, p        
    
  
model = ConvAutoencoder().to(device)    
    
print(model)
print('Number of parameters in model: ' + str(sum(p.numel() for p in model.parameters())))

learning_rate = 1e-5
batch_size =  10  #5  #10 #15.....
#learning_rate = 1e-6
#batch_size = 20
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

Xs = torch.from_numpy(Xs)
Xs = Variable(Xs).float()

Ys = torch.from_numpy(Ys)
Ys = Variable(Ys).float()

#torch_dataset = torch.utils.data.TensorDataset(Xs,(Ys[:,0,:,:].reshape((102,1,172,79)))) 
torch_dataset = torch.utils.data.TensorDataset(Xs,Ys) 
dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)


num_epochs = 10000

#weight = 'weights.pth'
#PATH = '/home/mdias/CFDML/obstacle/cnnModel/' +  weight
#model.load_state_dict(torch.load(PATH))

# If you want to try with early stopping, uncomment the next two lines
# and the lines at the end of training loop

#from pytorchtools import EarlyStopping
#early_stopping = EarlyStopping(patience=10, verbose=True)

save_loss = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for step, (batch_x, batch_y) in enumerate(dataloader): 

        b_x = Variable(batch_x).to(device)
        
        b_y = Variable(batch_y).to(device)
        
        ux, uy, p = model(b_x)     # input x and predict based on x
        y1 = b_y[:,0,:,:].reshape(( b_y.shape[0], 1 , b_y.shape[2], b_y.shape[3] ))
        y2 = b_y[:,1,:,:].reshape(( b_y.shape[0], 1 , b_y.shape[2], b_y.shape[3] ))
        y3 = b_y[:,2,:,:].reshape(( b_y.shape[0], 1 , b_y.shape[2], b_y.shape[3] ))

        loss = criterion(ux, y1) + criterion(uy, y2) + criterion(p, y3)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # print statistics
        running_loss += loss.item()
        #if step % 10 == 9:    # print every 10 mini-batches
        if step % 10 == 0:
           print('[%d, %5d] loss: %.8f  lr: %.8f' % (epoch + 1, step + 1, running_loss / 10, current_lr))
           save_loss.append(running_loss / 10)
           running_loss = 0.0
           
    #scheduler.step(metrics=loss.item())    
    #early_stopping(loss.item(), model)        
    #if early_stopping.early_stop:
    #    print("Early stopping")
    #    break
          
torch.save(model.state_dict(), './weightsCFD.pth')
save_loss = np.asarray(save_loss)
pickle.dump( save_loss, open( "save_lossCFD.pkl", "wb" ) )

x3 = Xs[0:100].to(device)
prediction = model(x3)
#pred = prediction.cpu().detach().numpy()
pred1 = prediction[0].cpu().detach().numpy()
pred2 = prediction[1].cpu().detach().numpy()
pred3 = prediction[2].cpu().detach().numpy()

pickle.dump( pred1, open( "pred1CFD.pkl", "wb" ) )
pickle.dump( pred2, open( "pred2CFD.pkl", "wb" ) )
pickle.dump( pred3, open( "pred3CFD.pkl", "wb" ) )

Xs = Xs.cpu().detach().numpy()
Ys = Ys.cpu().detach().numpy()

sample = 1
error1 =  np.abs(  pred1[sample,0,:,:] - Ys[sample,0,:,:] )
error2 =  np.abs(  pred2[sample,0,:,:] - Ys[sample,1,:,:] )
error3 =  np.abs(  pred3[sample,0,:,:] - Ys[sample,2,:,:] )

plt.figure()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.subplot(3, 3, 1)
plt.title('CFD', fontsize=18) 
plt.imshow( np.transpose(  Ys[sample,0,:,:]) , cmap='jet')
plt.colorbar(orientation='horizontal')
plt.ylabel('Ux', fontsize=18)
plt.subplot(3, 3, 2)
plt.title('CNN', fontsize=18) 
plt.imshow( np.transpose(  pred1[sample,0,:,:]), cmap='jet')
plt.colorbar(orientation='horizontal')
plt.subplot(3, 3, 3)
plt.title('Error', fontsize=18) 
plt.imshow( np.transpose( error1  ), cmap='jet')
plt.colorbar(orientation='horizontal')

plt.subplot(3, 3, 4)
plt.imshow( np.transpose(  Ys[sample,1,:,:]), cmap='jet')
plt.colorbar(orientation='horizontal')
plt.ylabel('Uy', fontsize=18)
plt.subplot(3, 3, 5)
plt.imshow( np.transpose(  pred2[sample,0,:,:]), cmap='jet')
plt.colorbar(orientation='horizontal')
plt.subplot(3, 3, 6)
plt.imshow( np.transpose( error2  ), cmap='jet')
plt.colorbar(orientation='horizontal')

plt.subplot(3, 3, 7)
plt.imshow( np.transpose(  Ys[sample,2,:,:]), cmap='jet')
plt.colorbar(orientation='horizontal')
plt.ylabel('p', fontsize=18)
plt.subplot(3, 3, 8)
plt.imshow( np.transpose(  pred3[sample,0,:,:]), cmap='jet')
plt.colorbar(orientation='horizontal')
plt.subplot(3, 3, 9)
plt.imshow( np.transpose( error3  ), cmap='jet')
plt.colorbar(orientation='horizontal')
plt.tight_layout()
plt.show()