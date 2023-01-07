import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
from torch.autograd import Variable
#default root directory


#loading data set function and pre processing them
def test_transform():
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return test_transform
    

def train_transform():
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return train_transforms

def valid_transform():
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return valid_transforms

def load_train_data(data_dir="./flowers"):
    train_dir = data_dir + '/train'
    train_data = datasets.ImageFolder(train_dir, transform=train_transform())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    return trainloader,train_data

def load_test_data(data_dir="./flowers"):
    test_dir = data_dir + '/test'
    test_data = datasets.ImageFolder(test_dir, transform=test_transform())
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = True)
    return testloader

def load_valid_data(data_dir="./flowers"):
    valid_dir = data_dir + '/valid'
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform())
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle = True)
    return validloader
    

#reference
#https://github.com/vishalnarnaware/Create-your-own-Image-Classifier/blob/master/futility.py