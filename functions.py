import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import data_prep


#setuping up network
def setup_nn(structure='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, power ='gpu'):
    device = torch.device("cuda" if torch.cuda.is_available() and power =="gpu" else "cpu")
    model,input_size = check_structure(structure)
    
    for para in model.parameters():
        para.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_size , hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    print(model)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
 
    return model, criterion
                                     
                                     
                                     
#saving checkpoint function
def save_checkpoint(train_data, model = 0, path = 'checkpoint.pth', structure = 'vgg16', hidden_units = 4096, dropout = 0.1, lr = 0.001, epochs = 1):
    print("Saving checkpoint...")
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    print("Checkpoint Saved!")
                                                                      
#Loading model from checkpoint function   
def load_checkpoint(path = 'checkpoint.pth'):
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']

    model, _ = setup_nn(structure, dropout, hidden_units, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

#prediction function                                     
def predict(image_path, model, topk=5,power='gpu'):
    device=torch.device("cuda" if torch.cuda.is_available and  power == "gpu" else "cpu")
    model.to(device)
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.to(device))
        
    probs = torch.exp(output).data
    
    return probs.topk(topk)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img)

    return image


def check_structure(structure):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features  
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = alexnet.classifier[1].in_features
    else:
        print("Please choose from vgg16,alexnet or densenet121")
        
    return model,input_size



 #refrence
 #https://github.com/vishalnarnaware/Create-your-own-Image-Classifier/blob/master/fmodel.py