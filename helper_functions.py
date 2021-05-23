#!/usr/bin/env python
# coding: utf-8

# In[2]:


def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std

def computeAccuracy(model,inp,target,aux_loss):
    model.eval()
    #pred = model(normalize(input)).argmax(dim = 1)
    if aux_loss:
        _,_,pred = model(inp)
    else:
        pred = model(inp)
    pred = pred.argmax(dim = 1)
    actual = target
    model.train()
    return (pred[pred == actual].shape[0]/pred.shape[0])


# In[ ]:




