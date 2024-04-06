import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import torchvision
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import wandb
from torchvision.transforms import RandomCrop, RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, Compose


# configuration cell
epochs = 1

activationFunctions = dict()
activationFunctions["conv1"] = "ReLU"
activationFunctions["conv2"] = "ReLU"
activationFunctions["conv3"] = "ReLU"
activationFunctions["conv4"] = "ReLU"
activationFunctions["conv5"] = "ReLU"
activationFunctions["fc1"] = "ReLU"
list_kernelSize= [3]*5
listDropout = [0,0,0.5]
kernelNumber = [32,32]+[64,64]+[128]
classes = 10
learningRate = 1e-4
nodesfc1 = 1024
lr_schedule = 1 # per 10 epochs half the learningRate
modelName = 'Best_CNN_5Layers_iNaturalist'


# check if CUDA is available
cuda = torch.cuda.is_available()
if cuda == True:
    device = torch.device("cuda")
if cuda != True:
    device = torch.device("cpu")
    
print(device)

################################## util.py##############################################################


## dataloader
def loader(t1data,valdata,t2data,batch):
    bs=batch
    bool=True
    bool2=False
    allLoaders = {
        'train' : torch.utils.data.DataLoader(t1data, batch_size=bs,num_workers = 4, shuffle=bool) ,
        'valid' : torch.utils.data.DataLoader(valdata, batch_size=bs,num_workers = 4,  shuffle=bool) ,
        'test'  : torch.utils.data.DataLoader(t2data, batch_size=bs,num_workers = 4,  shuffle=bool2)
    }
    return allLoaders


## transforms to match realModel input dims
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

def activationFun(activation):
    act=activation
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'elu':
        return nn.ELU()
    elif activation.lower() == 'silu':
        return nn.SiLU()
    elif activation.lower() == 'mish':
        return nn.Mish()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    
## Load dataset fn
def data_load():
    transforms=transform()
    t1set  = torchvision.datasets.ImageFolder('/kaggle/input/dl-assignment-2-data/inaturalist_12K/train', transforms['training'])
    train, val = random_split(t1set, [8000, 1999])
    t2set   = torchvision.datasets.ImageFolder('/kaggle/input/dl-assignment-2-data/inaturalist_12K/val', transforms['test'])
    lables = t1set.classes
    return train, val, t2set, lables

class blockConv(nn.Module):
    def __init__(self,channelsIn,channelsOut,kernel= 3 , BN=True , NL="ReLU",stride = 1, padding = 0):
        KL=channelsOut
        super(blockConv, self).__init__()
        self.BN,self.NL=BN,NL
        bol=False
        self.conv = nn.Conv2d(channelsIn, channelsOut, kernel_size= kernel , stride = stride, padding = padding, bias=bol)
        bol=self.BN==True
        if bol:
            val=0.001
            self.bn = nn.BatchNorm2d(channelsOut, eps=val)
        self.act = activationFun(NL)
        
    def forward(self, x):
        x = self.conv(x)
        bol=self.BN==True
        if bol:
            x = self.bn(x)
        
        return self.act(x)


class fc_block(nn.Module):
    runs=1
    def __init__( self , channelsIn ,channelsOut , BN=False , NL="relu"):
        x=channelsOut
        super(fc_block, self).__init__()
        self.fc = nn.Linear(channelsIn, channelsOut)
        self.BN,self.NL=BN,NL
       
        bol=self.BN==True
        if bol:
            value=0.001
            self.bn = nn.BatchNorm2d(channelsOut, eps=value)    
        self.act = activationFun(NL)
        
    def forward(self, x):
        x = self.fc(x)
        bol = self.BN==True
        if bol:
            value=0.001
            x = self.bn(x)
        x = self.act(x)
        
        return x

def get_fc_in(dim, list_kernelSize, kernelNumber):
    fc_in = dim - list_kernelSize[0] + 1 # conv1
    val=(fc_in - 2) //2  + 1
    fc_in = val # max pool 1

    s=1
    while s < 5:
        fc_in = fc_in - list_kernelSize[s] + 1 # conv2
        val1=(fc_in - 2) //2  + 1 
        fc_in =val1# max pool 
        s=s+1 
    
    val2=fc_in * fc_in
    ans= val2 * kernelNumber[4]
    return ans

def config_str_list_int(s):
    l=1
    return list(map(int, s[3:].split('-')))

def config_str_list_float(s):
    l=1
    return list(map(float, s.split('-')))

# test_data_path = 'give/path'
# class_images = [[] for _ in range(len(os.listdir(test_data_path)))]
# def gen_random_images():
#     for i, folder in enumerate(os.listdir(test_data_path)):
#         folder_path = os.path.join(test_data_path, folder)
#         for image_name in random.sample(os.listdir(folder_path),3):
#             image_path = os.path.join(folder_path, image_name)
#             img = Image.open(image_path).resize((256, 256))
#             class_images[i].append((img, folder))

# def get_prediction(image):
#     trained_model.eval()
#     transform = Compose([Resize((224, 224)), ToTensor()])
#     input_tensor = transform(image).to(device)
#     input_batch = input_tensor.unsqueeze(0)
#     with torch.no_grad():
#         output = trained_model(input_batch)
#     _, predicted_class = torch.max(output, 1)
#     predicted_class_idx = predicted_class.item()
#     return predicted_class_idx

class CNN_5layer(nn.Module):
    def __init__( self,list_kernelSize , kernelNumber , activationFunctions , listDropout, nodesfc1, classes):
        list1=list_kernelSize
        super(CNN_5layer, self).__init__()
        self.listDropout = listDropout
        bol2=False
        self.dim = 224
        self.conv1 = nn.Sequential(blockConv(3 , kernelNumber[0], kernel=list_kernelSize[0], BN=bol2, NL=activationFunctions['conv1']),nn.MaxPool2d((2, 2)))
        bol1=True
        

        self.conv2 = nn.Sequential(blockConv(kernelNumber[0], kernelNumber[1], kernel=list_kernelSize[1], BN=bol1, NL=activationFunctions['conv2']),nn.MaxPool2d((2, 2)))
        bol=self.listDropout[0]!=0
        if bol:
            self.dropout1 = nn.Dropout(listDropout[0])

        self.conv3 = nn.Sequential(blockConv(kernelNumber[1], kernelNumber[2], kernel=list_kernelSize[2], BN=bol1, NL=activationFunctions['conv3']),nn.MaxPool2d((2, 2)))
        self.conv4 = nn.Sequential(blockConv(kernelNumber[2], kernelNumber[3], kernel=list_kernelSize[3], BN=bol1, NL=activationFunctions['conv4']),nn.MaxPool2d((2, 2)))
        bol=self.listDropout[1]!=0
        if bol:
            self.dropout2 = nn.Dropout(listDropout[1])

        self.conv5 = nn.Sequential(blockConv(kernelNumber[3], kernelNumber[4], kernel=list_kernelSize[4], BN=bol1, NL=activationFunctions['conv5']),nn.MaxPool2d((2, 2)))
        kernel_size=kernelNumber[3]
        self.fc1_in_features = get_fc_in(self.dim, list_kernelSize, kernelNumber)
        kernel_size=list_kernelSize
        self.fc1 = fc_block(self.fc1_in_features, nodesfc1 , NL=activationFunctions['fc1'])
        bol=self.listDropout[2]!=0
        if bol:
            self.dropout3 = nn.Dropout(listDropout[2])
        
        self.fc2 = nn.Linear(nodesfc1, classes)
    
    
    def forward(self, x):
        bol=x.shape[2]!=self.dim
        if bol:
            print("input dim not matched")
            return
        x = self.conv1(x)
        x = self.conv2(x)
        
        bol=self.listDropout[0]!=0
        if bol:
            x = self.dropout1(x)

        x = self.conv3(x)
        
        x = self.conv4(x)
        
        bol=self.listDropout[1]!=0
        if bol:
            x = self.dropout2(x)
        y=x
        x = self.conv5(x)
        y=x.shape[0]
        x = x.view(y, -1)
        
        x = self.fc1(x)
        bol=self.listDropout[2]!=0
        if bol:
            x = self.dropout3(x)
        
        x = self.fc2(x)
        
        return x

def sp_train():
    runs=1
    configdefaults = {
        'epochs': 2,
        'kernel_size_config':'1) 5-5-3-3-3' ,
        'no_kernel_config':'1) 16-16-16-16-16',
        'dropout_config':'0-0-0.4',
        'fc1_nodes': 32,
        'batch_size': 64
    }
    # starting a wandb run

    wandb.init(config=configdefaults)
    start =0
    config = wandb.config
    epachs=2
    sizes= config.kernel_size_config
    numbers=config.no_kernel_config
    dropout=config.dropout_config
    str1=str(config.fc1_nodes)
    batchsize=str(config.batch_size)
    run_name="kSizes:["+sizes+"] kNumbers:["+numbers+"] dp:["+dropout+"] fc1:["+str1+"] bs:["+batchsize+"]"
    wandb.run.name=run_name
    
   
    model = CNN_5layer(config_str_list_int(config.kernel_size_config),
                       config_str_list_int(config.no_kernel_config),
                       {
        'conv1':'relu',
        'conv2':'relu',
        'conv3':'relu',
        'conv4':'relu',
        'conv5':'relu',
        'fc1':'relu'
        },
                       config_str_list_float(config.dropout_config),
                       config.fc1_nodes, 10).to(device)

    opti = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    trained_model,val_accuracy = train(
                      totalEpoch = config.epochs,
                      allLoaders = allLoaders,
                      realModel = model,
                      opt = opti,
                      criterion = nn.CrossEntropyLoss(),
                      scheduler = StepLR(opti, step_size=10, gamma=0.5),
                      cuda = cuda
                     )
    wandb.log({'val_accuracy':val_accuracy})
    

def train(totalEpoch, allLoaders, realModel, opt, criterion,scheduler, cuda):
    
    for epoch in range(totalEpoch):
        
        train_loss ,valid_loss= 0.0,0.0
        
        
        ###################
        # train the realModel #
        ###################
        realModel.train()
        tnum_correct,tnum_examples=0,0
        for data, target in allLoaders['train']:
            # move to GPU
            bol=cuda
            if bol:
                data, target = data.cuda(), target.cuda()
                
            opt.zero_grad()
            
            output = realModel(data)
            loss = criterion(output, target)
            
            
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            tnum_examples += target.size(0)
            tnum_correct += (predicted == target).sum().item()
            
        train_acc = (tnum_correct / tnum_examples) * 100
        train_loss = train_loss / len(allLoaders['train'])

        
        ######################    
        # validate the realModel #
        ######################
        realModel.eval()
        num_correct ,num_examples= 0,0
        
        
        
        for data, target in allLoaders['valid']:
            bol=cuda
            if bol:
                data, target = data.cuda(), target.cuda()
            
            output = realModel(data)
            loss = criterion(output, target)
            
            
            
            valid_loss += loss.item()
            
            _, val_predicted = torch.max(output.data, 1)
            num_examples += target.size(0)
            num_correct += (val_predicted == target).sum().item()
           

        valid_acc = (num_correct / num_examples) * 100
        valid_loss = valid_loss / len(allLoaders['valid'])
        
        scheduler.step()
        
        print('Epoch: {}\tTraining Loss: {:.6f}\tTrain Accuracy: {:.2f}\tValidation Loss: {:.6f}\tvalidation Accuracy: {:.2f}'.format(
            epoch, 
            train_loss,
            train_acc,
            valid_loss,
            valid_acc
            ))
        
        # wandb.log({'epoch': epoch,'train loss': train_loss,'train accuracy': train_acc,
        #            'val loss': valid_loss, 'val accuracy': valid_acc})
        
   
    return realModel, valid_acc

batch = 64
dataT1, dataVal, dataT2, lables = data_load()
allLoaders = loader(dataT1, dataVal, dataT2,  batch)

realModel = CNN_5layer(list_kernelSize, kernelNumber, activationFunctions, listDropout, nodesfc1, classes)
realModel = realModel.to(device)

bol=False
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(realModel.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=bol)
scheduler = StepLR(opt, step_size=10, gamma=lr_schedule)

trained_model,_ = train(
                      totalEpoch = epochs,
                      allLoaders = allLoaders,
                      realModel = realModel,
                      opt = opt,
                      criterion = criterion,
                      scheduler = scheduler,
                      cuda = cuda
                     )

trained_model.eval()
test_acc,num_correct,num_examples, test_loss = 0,0,0,0
loader=allLoaders['test']
for data, target in loader:
    bol=cuda
    if bol:
        data, target = data.cuda(), target.cuda()

    output = trained_model(data)
    loss = criterion(output, target)



    test_loss += loss.item()

    _, test_predicted = torch.max(output.data, 1)
    num_examples += target.size(0)
    num_correct += (test_predicted == target).sum().item()


    test_acc = (num_correct / num_examples) * 100
    test_loss = test_loss / len(loader)

print('Test Accuracy of the realModel is : {}%'.format(test_acc, 2))



kernel_size_config_options = ['1) 5-5-3-3-3', 
                       '2) 3-3-3-3-3', 
                       '3) 3-3-3-5-5'
                       ]
no_kernel_config_options = [ 
                      '1) 64-64-32-32-16', 
                      '2) 32-32-64-64-128']
fc1_nodes_options = [512,1024]

required_sweep_params = dict({
        'kernel_size_config':{
            'values': kernel_size_config_options
        },
        'no_kernel_config': {
            'values': no_kernel_config_options
        },
        'dropout_config':{
            'values':['0-0-0.5','0-0-0.3']
        },
        'fc1_nodes':{
            'values': fc1_nodes_options
        },
        'batch_size': {
            'values':[32]
        },
        'epochs': {
            'values':[25]
        },        
    })
sweep_config = {
    'name' : 'DL Assigment 2 Part-A',
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': required_sweep_params
}

''''
sweep_id = wandb.sweep(sweep_config, project='DL-Assignment-2')
wandb.agent(sweep_id, sp_train,count=20)
wandb.finish()
'''

'''
# Question-4
fig, axs = plt.subplots(10, 3, figsize=(10, 30))
for i in range(10):
    for j in range(3):
        image, true_label = class_images[i][j]
        axs[i, j].imshow(image)
        axs[i, j].axis('off')
        axs[i, j].set_title(f"True Label: {true_label} | Predicted Label: {lables[get_prediction(image)]}", fontsize=10, ha='center', pad=10, loc='center', color='green')
wandb.init(project='DL-Assignment-2', name = 'Grid')
wandb.log({'Grid': wandb.Image(plt)})
wandb.finish()
plt.show()
'''