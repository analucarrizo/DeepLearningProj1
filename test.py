#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.optim import SGD
import dlc_practical_prologue as prologue
from plot import getMetrics, stats, plot_losses
from train import train_model_auxloss, train_model 
from models import noWeightsharingnoAuxloss, weightsharingnoAuxloss, noWeightsharingAuxloss, weightsharingAuxloss, final_model, normalize


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


train_input, train_target, test_input, test_target =     prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)


# In[4]:


###noWeightsharingnoAuxloss(nn.Module) training##

train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
train_input_norm = normalize(train_input).to(device)
train_target = train_target.to(device)
test_input = normalize(test_input).to(device)
test_target = test_target.to(device)
model_noWeight_noAux = noWeightsharingnoAuxloss().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_noWeight_noAux.parameters(), lr=0.01,momentum=0.8,nesterov=True)
train_loss,validation_loss,train_acc,validation_acc = train_model(model_noWeight_noAux ,train_input_norm,train_target,optimizer,criterion,test_input,test_target,mini_batch_size = 5,nb_epochs = 25,print_progress= True)
model_noWeight_noAux.eval()
noWeightsharingnoAuxloss

#### metric evaluation model_noWeight_noAux
print("test data")
stats(model_noWeight_noAux,test_input.to(device),test_target.to(device))
print("train data")
stats(model_noWeight_noAux,train_input.to(device),train_target.to(device))
plot_losses(train_loss,validation_loss,train_acc,validation_acc)


# In[5]:


###WeightsharingnoAuxloss(nn.Module) training##F

train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
train_input_norm = normalize(train_input).to(device)
train_target = train_target.to(device)
test_input = normalize(test_input).to(device)
test_target = test_target.to(device)
model_yesWeight_noAux = weightsharingnoAuxloss().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_yesWeight_noAux.parameters(), lr=0.01,momentum=0.8,nesterov=True)
train_loss,validation_loss,train_acc,validation_acc = train_model(model_yesWeight_noAux,train_input_norm,train_target,optimizer,criterion,test_input,test_target,mini_batch_size = 16,nb_epochs = 60,print_progress= True)
model_yesWeight_noAux.eval()


#### metric evaluation model_yesWeight_noAux
print("test data")
stats(model_yesWeight_noAux,test_input.to(device),test_target.to(device))
print("train data")
stats(model_yesWeight_noAux,train_input.to(device),train_target.to(device))
plot_losses(train_loss,validation_loss,train_acc,validation_acc)


# In[6]:


###noWeightsharingAuxloss(nn.Module) training##

train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
train_input_norm = normalize(train_input).to(device)
train_target = train_target.to(device)
test_input = normalize(test_input).to(device)
test_target = test_target.to(device)
train_classes = train_classes.to(device)
model_noWeight_yesAux = noWeightsharingAuxloss().to(device)
criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.SGD(model_noWeight_yesAux.parameters(), lr=0.01,momentum=0.7,nesterov=True)
train_loss,validation_loss,train_acc,validation_acc =  train_model_auxloss(model_noWeight_yesAux,train_input_norm,train_target,train_classes,optimizer,criterion,test_input,test_target,mini_batch_size = 16,nb_epochs = 60,print_progress= True)
model_noWeight_yesAux.eval()


#### metric evaluation model_noWeight_yesAux
print("test data")
stats(model_noWeight_yesAux,test_input.to(device),test_target.to(device),aux_loss = True)
print("train data")
stats(model_noWeight_yesAux,train_input.to(device),train_target.to(device),aux_loss = True)
plot_losses(train_loss,validation_loss,train_acc,validation_acc)


# In[7]:


#######weightsharingAuxloss training

train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
train_input_norm = normalize(train_input).to(device)
train_target = train_target.to(device)
test_input = normalize(test_input).to(device)
test_target = test_target.to(device)
train_classes = train_classes.to(device)
model_yesWeight_yesAux = weightsharingAuxloss().to(device)
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(model_yesWeight_yesAux.parameters(), lr=0.01,momentum=0.8,nesterov=True)
train_loss,validation_loss,train_acc,validation_acc =  train_model_auxloss(model_yesWeight_yesAux,train_input_norm,train_target,train_classes,optimizer,criterion,test_input,test_target,mini_batch_size = 16,nb_epochs = 25,print_progress= True)
model_yesWeight_yesAux.eval()


#### metric evaluation weightsharingAuxloss
print("test data")
stats(model_yesWeight_yesAux,test_input.to(device),test_target.to(device),aux_loss = True)
print("train data")
stats(model_yesWeight_yesAux,train_input.to(device),train_target.to(device),aux_loss = True)
plot_losses(train_loss,validation_loss,train_acc,validation_acc)

