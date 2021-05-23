#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torch.optim import SGD
from torch import nn

# from models import noWeightsharingnoAuxloss, 


# In[4]:


def train_model_auxloss(model,train_input,train_target,train_classes,optimizer,criterion,test_input,test_target,mini_batch_size = 100,nb_epochs = 100,print_progress= True):
    validation_input = test_input
    validation_target = test_target
    validation_loss = []
    train_loss = []
    validation_acc = []
    train_acc = []
    batch_size = mini_batch_size
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            if b + batch_size > train_input.size(0):
                batch_size = train_input.size(0) - b
            else:
                batch_size = mini_batch_size
            digitResa,digitResb,output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(output, train_target.narrow(0, b, batch_size))+ criterion(digitResa,train_classes[:,0].narrow(0, b, batch_size)) + criterion(digitResb,train_classes[:,1].narrow(0, b, batch_size))
            loss_acc = criterion(output, train_target.narrow(0, b, batch_size))
            acc_loss = acc_loss + loss_acc.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if(print_progress):
          ##loss
            with torch.no_grad():
                model.eval()
                _,_,val_out = model(validation_input)
                val_loss = criterion((val_out),validation_target).item()
                validation_loss.append(val_loss)
                batches = int(train_input.size(0)/ mini_batch_size) if train_input.size(0)% mini_batch_size == 0 else int(train_input.size(0)/ mini_batch_size)+1
                _,_,tl = model(train_input) 
                train_loss.append(criterion(tl,train_target).item())
            
                ##acc
                _,_,out_train = model(train_input)
                pred_train = torch.argmax((out_train),dim = 1)
                pred_test = torch.argmax((val_out),dim = 1)
                train_acc.append(pred_train[pred_train == train_target].shape[0]/pred_train.shape[0])
                validation_acc.append(pred_test[pred_test == test_target].shape[0]/pred_test.shape[0])
                model.train()
                model.zero_grad()
                print(e, acc_loss/batches)

    return train_loss,validation_loss,train_acc,validation_acc


# In[2]:


def train_model(model,train_input,train_target,optimizer,criterion,test_input,test_target,mini_batch_size = 100,nb_epochs = 100,print_progress= True):
    validation_input = test_input
    validation_target = test_target
    validation_loss = []
    train_loss = []
    validation_acc = []
    train_acc = []
    batch_size = mini_batch_size
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            if b + batch_size > train_input.size(0):
                batch_size = train_input.size(0) - b
            else:
                batch_size = mini_batch_size
            output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if(print_progress):
          ##loss
            with torch.no_grad():
                model.eval()
                val_loss = criterion((model(validation_input)),validation_target)
                validation_loss.append(val_loss)
                batches = int(train_input.size(0)/ mini_batch_size) if train_input.size(0)% mini_batch_size == 0 else int(train_input.size(0)/ mini_batch_size)+1
                train_loss.append(criterion((model(train_input)),train_target).item())
                ##acc
                pred_train = torch.argmax(model(train_input),dim = 1)
                pred_test = torch.argmax((model(test_input)),dim = 1)
                train_acc.append(pred_train[pred_train == train_target].shape[0]/pred_train.shape[0])
                validation_acc.append(pred_test[pred_test == test_target].shape[0]/pred_test.shape[0])
                model.train()
                model.zero_grad()
                print(e, acc_loss/batches)
    return train_loss,validation_loss,train_acc,validation_acc



