#imports
import torch
import torchvision
import torch.nn as nn
import math
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, Compose
import shutil
from tqdm import tqdm
import torch.optim as optim
import sys 
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


# how to check if cuda is available
cuda = torch.cuda.is_available()
if cuda == True:
    device = torch.device("cuda")
if cuda != True:
    device = torch.device("cpu")
    
print(device)

# loading the data
def loader(t1data, valdata, t2data, batch):
    bs=batch
    bol1=True
    bol2=False
    train = torch.utils.data.DataLoader(t1data, batch_size=bs,num_workers =4, shuffle=bol1)
    valid = torch.utils.data.DataLoader(valdata, batch_size=bs, num_workers =4,shuffle=bol1)
    test = torch.utils.data.DataLoader(t2data, batch_size=bs, num_workers =4,shuffle=bol2)
    allLoaders = {
        'train' : train,
        'valid' : valid,
        'test'  : test
    }
    return allLoaders

# get model function is defined here
def model_get(modelName):
    bol =modelName.lower() == 'resnet50'
    model = None
    import torchvision as tv 
    if bol:
        model = tv.models.resnet50(pretrained=True)
    return model

# performinig the transformation to match model input dimensions
def transform():
    string ='Normalize'
    valResize = 256 #134 #36
    sizeChange = 224 #128#32
    valCenterCrop = sizeChange
    
    
    t1_t = Compose([RandomResizedCrop(sizeChange),
                       RandomHorizontalFlip(),
                       ToTensor(),
                       Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    val_t = Compose([Resize(valResize),
                       CenterCrop(valCenterCrop),
                       ToTensor(),
                       Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    t2_t = Compose([Resize((sizeChange,sizeChange)), 
                      ToTensor(), 
                      Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
    transforms = {
        'training':   t1_t,
        'validation': val_t,
        'test': t2_t
    }
    
    return transforms

#Loading dataset fn
def data_load():
    transforms=transform()
    t1set  = torchvision.datasets.ImageFolder('/kaggle/input/dl-assignment-2-data/inaturalist_12K/train', transforms['training'])
    train, val = random_split(t1set, [8000, 1999])
    t2set   = torchvision.datasets.ImageFolder('/kaggle/input/dl-assignment-2-data/inaturalist_12K/val', transforms['test'])
    return train, val, t2set


#########################################################################

def model_change_classifier(model):
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features,500),
                         nn.ReLU(),
                         nn.Dropout(),
                         nn.Linear(500,10))

#training the model
def train(totalEpoch, allLoaders, model, opt, criterion, cuda):
    

    for epoch in range(1, totalEpoch+1):
        
        train_loss ,valid_loss= 0.0,0.0
        
        optimizer=opt
        model.train()
        tnum_correct,tnum_examples=0,0
        for data, target in tqdm(allLoaders['train']):
            # move to GPU
            bol=cuda
            if bol:
                data, target = data.cuda(), target.cuda()
                
            opt.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            tnum_examples += target.size(0)
            tnum_correct += (predicted == target).sum().item()
            
        train_acc = (tnum_correct / tnum_examples) * 100
        train_loss = train_loss / len(allLoaders['train'])

        
  
        # validating the Model 

        model.eval()
        num_correct ,num_examples= 0,0
        
        
        
        for data, target in tqdm(allLoaders['valid']):
            bol=cuda
            if bol:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = criterion(output, target)
            
            
            
            valid_loss += loss.item()
            
            _, val_predicted = torch.max(output.data, 1)
            num_examples += target.size(0)
            num_correct += (val_predicted == target).sum().item()
           

        valid_acc = (num_correct / num_examples) * 100
        valid_loss = valid_loss / len(allLoaders['valid'])
        
        
        print('Epoch: {}\tTraining Loss: {:.6f}\tTrain Accuracy: {:.2f}\tValidation Loss: {:.6f}\tvalidation Accuracy: {:.2f}'.format(
            epoch, 
            train_loss,
            train_acc,
            valid_loss,
            valid_acc
            ))
        
        
    return model

def freeze(model, strategy, k):
    if k == 0:
        return
    
    if strategy == "first":
        layer_num = 0
        for _, layer in model.named_children():
            layer_num += 1
            if layer_num <= k:
                for _, param in layer.named_parameters():
                    param.requires_grad = False

    if strategy == "middle":
        layer_num = 0
        for _, layer in model.named_children():
            layer_num += 1
            if (len(list(model.named_children())) // 3  ) <= layer_num < ((len(list(model.named_children())) // 3) * 2):
                for _, param in layer.named_parameters():
                    param.requires_grad = False

    if strategy == "last":
        layer_num = 0
        for _, layer in model.named_children():
            layer_num += 1
            if (len(list(model.named_children())) - k + 1) <= layer_num <= (len(list(model.named_children()))):
                for _, param in layer.named_parameters():
                    param.requires_grad = False


########################################################################

def sp_train(l,strategy):
    config = {
        'model_name':'ResNet50',
        'totalEpoch': 1,
        'learning_rate_1': 1e-4,
        'learning_rate_2': 1e-4,
        'batchnorm_pretrain':'YES',
        'opt': 'sgd'
    }
    
    modelName = config['model_name']
    model = model_get(modelName)
   
    
    datasetTrain, datasetVal, datasetTest = data_load()
    batch_size = 64
    allLoaders = loader(datasetTrain, datasetVal, datasetTest, batch_size)
    
    freeze(model,strategy,l)
    model = model.to(device)
    string = 'opt'
    if config[string]=='sgd':
        opt = optim.SGD(model.parameters(), lr=config['learning_rate_1'], momentum = 0.9)
    if config[string]=='adam':
        opt = optim.Adam(model.parameters(), lr=config['learning_rate_1'], betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    
    _ = train(totalEpoch=config['totalEpoch'],
                      allLoaders = allLoaders,
                      model = model,
                      opt = opt,
                      criterion = criterion,
                      cuda = cuda
                     )
    


# sp_train(3,'first')

sp_train(3,'last')

# sp_train(3,'middle')

# sp_train(0,'all')