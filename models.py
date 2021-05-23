#!/usr/bin/env python
# coding: utf-8




import torch
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue




#########################################################################
#                     NO WEIGHT SHARING AND NO AUXILIARY LOSS
#########################################################################
class noWeightsharingnoAuxloss(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1a = nn.Dropout2d(p = 0.5)
        self.bn = nn.BatchNorm2d(16)
        self.conv2a = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2a = nn.Dropout2d(p = 0.5)


        self.conv1b = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1b = nn.Dropout2d(p = 0.5)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2b = nn.Dropout2d(p = 0.5)

        self.transfa = nn.Sequential(nn.Dropout(),nn.Linear(1600, 20),nn.ReLU(),nn.Dropout(),nn.Linear(20, 10),nn.ReLU())
        self.transfb = nn.Sequential(nn.Dropout(),nn.Linear(1600, 20),nn.ReLU(),nn.Dropout(),nn.Linear(20, 10),nn.ReLU())
        
        self.transf = nn.Linear(20,2)

    def forward(self, x):
        xa = x[:,0,:,:].unsqueeze(1)
        xb = x[:,1,:,:].unsqueeze(1)

        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xa)))), padding = 1,kernel_size = 2,stride =2)
        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xa)))), padding = 1,kernel_size = 2,stride =1)
        xa = self.transfa(xa.view(-1,1600))

        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop1b(self.conv1b(xb)))), padding = 1,kernel_size = 2,stride =2)
        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop2b(self.conv2b(xb)))), padding = 1,kernel_size = 2,stride =1)
        xb = self.transfb(xb.view(-1,1600))

        return self.transf(torch.cat((xa,xb),dim = 1))
    


# In[17]:


#########################################################################
#                     WEIGHT SHARING AND NO AUXILIARY LOSS
#########################################################################

class weightsharingnoAuxloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1a = nn.Dropout2d(p = 0.5)
        self.conv2a = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2a = nn.Dropout2d(p = 0.5)

        self.bn = nn.BatchNorm2d(16)
        self.transfa = nn.Sequential(nn.Dropout(),nn.Linear(1600, 20),nn.ReLU(),nn.Dropout(),nn.Linear(20, 10),nn.ReLU())
        self.transf = nn.Linear(20,2)

    def forward(self, x):
        xa = x[:,0,:,:].unsqueeze(1)
        xb = x[:,1,:,:].unsqueeze(1)
        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xa)))), padding = 1,kernel_size = 2,stride =2)
        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xa)))), padding = 1,kernel_size = 2,stride =1)
        xa = self.transfa(xa.view(-1,1600))

        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xb)))), padding = 1,kernel_size = 2,stride =2)
        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xb)))), padding = 1,kernel_size = 2,stride =1)
        xb = self.transfa(xb.view(-1,1600))

        return self.transf(torch.cat((xa,xb),dim = 1))


# In[18]:


#########################################################################
#                     NO WEIGHT SHARING AND AUXILIARY LOSS
#########################################################################

class noWeightsharingAuxloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1a = nn.Dropout2d(p = 0.5)
        self.conv2a = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2a = nn.Dropout2d(p = 0.5)

        self.bn = nn.BatchNorm2d(16)

        self.conv1b = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1b = nn.Dropout2d(p = 0.5)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2b = nn.Dropout2d(p = 0.5)

        self.transfa = nn.Sequential(nn.Dropout(),nn.Linear(1600, 20),nn.ReLU(),nn.Dropout(),nn.Linear(20, 10))
        self.transfb = nn.Sequential(nn.Dropout(),nn.Linear(1600, 20),nn.ReLU(),nn.Dropout(),nn.Linear(20, 10))

        self.transf = nn.Sequential(nn.ReLU(),nn.Linear(20,2))
    
    def forward(self, x):
        xa = x[:,0,:,:].unsqueeze(1)
        xb = x[:,1,:,:].unsqueeze(1)

        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xa)))), padding = 1,kernel_size = 2,stride =2)
        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xa)))), padding = 1,kernel_size = 2,stride =1)
        xa = self.transfa(xa.view(-1,1600))

        digitResa = xa
        xa = F.relu(xa)

        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop1b(self.conv1b(xb)))), padding = 1,kernel_size = 2,stride =2)
        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop2b(self.conv2b(xb)))), padding = 1,kernel_size = 2,stride =1)
        xb = self.transfb(xb.view(-1,1600))

        digitResb = xb

        return digitResa,digitResb,self.transf(torch.cat((xa,xb),dim = 1))


# In[19]:


#########################################################################
#                      WEIGHT SHARING AND AUXILIARY LOSS
#########################################################################

class weightsharingAuxloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1a = nn.Dropout2d(p = 0.5)
        self.bn = nn.BatchNorm2d(16)
        self.conv2a = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2a = nn.Dropout2d(p = 0.5)

        self.transfa = nn.Sequential(nn.Dropout(),nn.Linear(1600, 20),nn.ReLU(),nn.Dropout(),nn.Linear(20,10))
        self.transf = nn.Sequential(nn.ReLU(),nn.Linear(20,2))

    def forward(self, x):
        xa = x[:,0,:,:].unsqueeze(1)
        xb = x[:,1,:,:].unsqueeze(1)

        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xa)))), padding = 1,kernel_size = 2,stride =2)
        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xa)))), padding = 1,kernel_size = 2,stride =1)
        xa = self.transfa(xa.view(-1,1600))

        digitResa = xa
        xa = F.relu(xa)

        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xb)))), padding = 1,kernel_size = 2,stride =2)
        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xb)))), padding = 1,kernel_size = 2,stride =1)
        xb = self.transfa(xb.view(-1,1600))
        digitResb = xb
        xb = F.relu(xb)

        return digitResa,digitResb,self.transf(torch.cat((xa,xb),dim = 1))



class final_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 16, kernel_size=3,padding = 1,dilation = 1,stride =1)
        self.conv2_drop1a = nn.Dropout2d(p = 0.5)
        self.bn = nn.BatchNorm2d(16)
        self.conv2a = nn.Conv2d(16, 16, kernel_size=2,padding = 1,dilation = 1,stride =1)
        self.conv2_drop2a = nn.Dropout2d(p = 0.5)

        self.transfa = nn.Sequential(nn.Dropout(),nn.Linear(1600, 40),nn.ReLU(),nn.Linear(40, 20),nn.Dropout(),nn.Linear(20,10))
        self.transf = nn.Sequential(nn.ReLU(),nn.Linear(20,2))

    def forward(self, x):
        xa = x[:,0,:,:].unsqueeze(1)
        xb = x[:,1,:,:].unsqueeze(1)

        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xa)))), padding = 1,kernel_size = 2,stride =2)
        xa = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xa)))), padding = 1,kernel_size = 2,stride =1)
        xa = self.transfa(xa.view(-1,1600))

        digitResa = xa
        xa = F.relu(xa)

        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop1a(self.conv1a(xb)))), padding = 1,kernel_size = 2,stride =2)
        xb = F.max_pool2d(F.relu(self.bn(self.conv2_drop2a(self.conv2a(xb)))), padding = 1,kernel_size = 2,stride =1)
        xb = self.transfa(xb.view(-1,1600))
        digitResb = xb
        xb = F.relu(xb)
        return digitResa,digitResb,self.transf(torch.cat((xa,xb),dim = 1))







