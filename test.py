#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.optim import SGD
import dlc_practical_prologue as prologue
from train import train_model_auxloss, train_model 
from models import noWeightsharingnoAuxloss, weightsharingnoAuxloss, noWeightsharingAuxloss, weightsharingAuxloss, final_model
from helper_functions import normalize,computeAccuracy

# In[2]:

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ###noWeightsharingnoAuxloss(nn.Module) training##
    print("training model 1: model with no weight sharing and no auxiliary loss" )
    train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
    train_input = normalize(train_input).to(device)
    train_target = train_target.to(device)
    test_input = normalize(test_input).to(device)
    test_target = test_target.to(device)
    model_noWeight_noAux = noWeightsharingnoAuxloss().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_noWeight_noAux.parameters(), lr=0.1,momentum=0.2,nesterov=False)
    train_loss,validation_loss,train_acc,validation_acc = train_model(model_noWeight_noAux ,train_input,train_target,optimizer,criterion,test_input,test_target,mini_batch_size = 50,nb_epochs = 25,print_progress= True)
    model_noWeight_noAux.eval()


    #### metric evaluation model_noWeight_noAux
    print(f'train data accuracy:{computeAccuracy(model_noWeight_noAux,train_input,train_target,aux_loss = False)}')
    print(f'test data accuracy:{computeAccuracy(model_noWeight_noAux,test_input,test_target,aux_loss = False)}')
   




    # In[5]:

    print("*****************************************")
    print("*****************************************")
    ###WeightsharingnoAuxloss(nn.Module) training##F
    print("training model 2: model with weight sharing and no auxiliary loss ")
    train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
    train_input = normalize(train_input).to(device)
    train_target = train_target.to(device)
    test_input = normalize(test_input).to(device)
    test_target = test_target.to(device)
    model_yesWeight_noAux = weightsharingnoAuxloss().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_yesWeight_noAux.parameters(), lr=0.1,momentum=0.0,nesterov=False)
    train_loss,validation_loss,train_acc,validation_acc = train_model(model_yesWeight_noAux,train_input,train_target,optimizer,criterion,test_input,test_target,mini_batch_size = 32,nb_epochs = 25,print_progress= True)
    model_yesWeight_noAux.eval()

    
    #### metric evaluation model_yesWeight_noAux
    print(f'train data accuracy:{computeAccuracy(model_yesWeight_noAux,train_input,train_target,aux_loss = False)}')
    print(f'test data accuracy:{computeAccuracy(model_yesWeight_noAux,test_input,test_target,aux_loss = False)}')
    


    # In[6]:

    print("*****************************************")
    print("*****************************************")
    ###noWeightsharingAuxloss(nn.Module) training##
    print("training model 3: model with no weight sharing and with auxiliary loss ")
    train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
    train_input = normalize(train_input).to(device)
    train_target = train_target.to(device)
    test_input = normalize(test_input).to(device)
    test_target = test_target.to(device)
    train_classes = train_classes.to(device)
    model_noWeight_yesAux = noWeightsharingAuxloss().to(device)
    criterion = nn.CrossEntropyLoss()



    optimizer = torch.optim.SGD(model_noWeight_yesAux.parameters(), lr=0.01,momentum=0.8,nesterov=False)
    train_loss,validation_loss,train_acc,validation_acc =  train_model_auxloss(model_noWeight_yesAux,train_input,train_target,train_classes,optimizer,criterion,test_input,test_target,mini_batch_size = 50,nb_epochs = 25,print_progress= True)
    model_noWeight_yesAux.eval()


    #### metric evaluation model_noWeight_yesAux
    print(f'train data accuracy:{computeAccuracy(model_noWeight_yesAux,train_input,train_target,aux_loss = True)}')
    print(f'test data accuracy:{computeAccuracy(model_noWeight_yesAux,test_input,test_target,aux_loss = True)}')
    

    # In[7]:

    print("*****************************************")
    print("*****************************************")
    #######weightsharingAuxloss training
    print("training model 4: model with weight sharing and with auxiliary loss ")
    train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
    train_input = normalize(train_input).to(device)
    train_target = train_target.to(device)
    test_input = normalize(test_input).to(device)
    test_target = test_target.to(device)
    train_classes = train_classes.to(device)
    model_yesWeight_yesAux = weightsharingAuxloss().to(device)
    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model_yesWeight_yesAux.parameters(), lr=0.01,momentum=0.7,nesterov=False)
    train_loss,validation_loss,train_acc,validation_acc =  train_model_auxloss(model_yesWeight_yesAux,train_input,train_target,train_classes,optimizer,criterion,test_input,test_target,mini_batch_size = 16,nb_epochs = 25,print_progress= True)
    model_yesWeight_yesAux.eval()


    #### metric evaluation weightsharingAuxloss
    print(f'train data accuracy:{computeAccuracy(model_yesWeight_yesAux,train_input,train_target,aux_loss = True)}')
    print(f'test data accuracy:{computeAccuracy(model_yesWeight_yesAux,test_input,test_target,aux_loss = True)}')


    
    print("*****************************************")
    print("*****************************************")
    #######final model
    print("training model 5: model with weight sharing and with auxiliary loss with extra hidden layer")
    train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(1000)
    train_input = normalize(train_input).to(device)
    train_target = train_target.to(device)
    test_input = normalize(test_input).to(device)
    test_target = test_target.to(device)
    train_classes = train_classes.to(device)
    finalModel = final_model().to(device)
    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(finalModel.parameters(), lr=0.01,momentum=0.5,nesterov=False)
    train_loss,validation_loss,train_acc,validation_acc =  train_model_auxloss(finalModel,train_input,train_target,train_classes,optimizer,criterion,test_input,test_target,mini_batch_size = 16,nb_epochs = 25,print_progress= True)
    finalModel.eval()


    #### metric evaluation weightsharingAuxloss
    print(f'train data accuracy:{computeAccuracy(finalModel,train_input,train_target,aux_loss = True)}')
    print(f'test data accuracy:{computeAccuracy(finalModel,test_input,test_target,aux_loss = True)}')
    