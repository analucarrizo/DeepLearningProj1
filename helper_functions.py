#!/usr/bin/env python
# coding: utf-8

# In[2]:

#function used to normalize data
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std

#computes accuracy of a model
#input: model: model we are evaluating on
    #  inp: input data
    #  target: expected target
    #  aux_loss: Boolean specifying if model uses aux loss or not
#returns accuracy
def computeAccuracy(model,inp,target,aux_loss):
    model.eval()
    if aux_loss:
        _,_,pred = model(inp)
    else:
        pred = model(inp)
    pred = pred.argmax(dim = 1)
    actual = target
    model.train()
    return (pred[pred == actual].shape[0]/pred.shape[0])







