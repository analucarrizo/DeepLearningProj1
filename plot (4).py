#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame
from models import normalize


# In[1]:


def getMetrics(pred,labels,pos,neg):
    right_preds = pred[pred == labels]
    wrong_preds = pred[pred != labels]
    TP = right_preds[right_preds == pos].shape[0]
    TN = right_preds[right_preds == neg].shape[0]
    FP = wrong_preds[wrong_preds == pos].shape[0]
    FN = wrong_preds[wrong_preds == neg].shape[0]
    print(right_preds.shape[0]/pred.shape[0])
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1Score = 2*(Recall * Precision) / (Recall + Precision)
    return accuracy,Precision,Recall,F1Score


# In[3]:


def stats(model,data,labels,aux_loss = False):
    if aux_loss:
        _,_,output = model(normalize(data))
        pred = torch.argmax(output,dim =1)
    else:
        pred = torch.argmax(model(normalize(data)),dim = 1)
    accuracy_leq,Precision_leq,Recall_leq,F1Score_leq = getMetrics(pred,labels,1,0)
    accuracy_greater,Precision_greater,Recall_greater,F1Score_greater = getMetrics(pred,labels,0,1)
    types = ['accuracy','Precision','Recall','F1Score']
    classes = ['lesser or equal class ', 'greater class']
    data = [[accuracy_leq,Precision_leq,Recall_leq,F1Score_leq],[accuracy_greater,Precision_greater,Recall_greater,F1Score_greater]]
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=types,
        y = classes,
        colorscale='blues'))
    #fig.data[0].update(zmin= 0.75, zmax=1)
    annotations = go.Annotations()
    for n, row in enumerate(data):
        for m, val in enumerate(row):
            annotations.append(go.Annotation(text=str(data[n][m])[:5], x=types[m], y=classes[n],
                                          xref='x1', yref='y1', showarrow=False))
    fig['layout'].update(title="Metrics", annotations=annotations)
    fig.show()


# In[4]:


def plot_losses(train_loss,validation_loss,train_acc,validation_acc):
    train_curve = list(zip(train_loss,range(1,len(train_loss)+1)))
    validation_curve = list(zip(validation_loss,range(1,len(validation_loss)+1)))
    train = list(map(lambda x: ['train_loss',x[0],x[1]],train_curve))
    validation = list(map(lambda x: ['validation_loss',x[0],x[1]],validation_curve))
    train.extend(validation)
    df = DataFrame(train,columns = ['type','y','x'])
    fig = px.line(df, x="x", y="y", title='loss_plot', color='type')
    fig.show()

    train_curve = list(zip(train_acc,range(1,len(train_acc)+1)))
    validation_curve = list(zip(validation_acc,range(1,len(validation_acc)+1)))
    train = list(map(lambda x: ['train_loss',x[0],x[1]],train_curve))
    validation = list(map(lambda x: ['validation_loss',x[0],x[1]],validation_curve))
    train.extend(validation)
    df2 = DataFrame(train,columns = ['type','y','x'])
    fig2 = px.line(df2, x="x", y="y", title='accuracy_plot', color='type')
    fig2.show()

