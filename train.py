import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import json
from PIL import Image
import functions 
import data_prep as dp
from workspace_utils import active_session


# arguments and their default values
parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--checkpoint', action="store", default="./checkpoint.pth")
parser.add_argument('--dropout', action="store", default=0.1,type=float)
parser.add_argument('--hidden_units', action="store", type=int, default=256)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--structure', action="store", default="vgg16")
parser.add_argument('--power', action="store", default="gpu")
parser.add_argument('data_dir', action="store", default="./flowers")


args = parser.parse_args()
lr = args.learning_rate
path = args.checkpoint
data_dir = args.data_dir
dropout = args.dropout
hidden_units = args.hidden_units
epochs = args.epochs
structure = args.structure
power = args.power



def setup():
    model, criterion = functions.setup_nn(structure,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    return model,optimizer,criterion

def train_model():
    device=torch.device("cuda" if torch.cuda.is_available and  power == "gpu" else "cpu")
    trainloader,train_data = dp.load_train_data(data_dir)
    validloader=  dp.load_valid_data(data_dir)
    testloader = dp.load_test_data(data_dir)
   
    model,optimizer,criterion =setup()
# Training Model classifier
    steps = 0
    running_loss = 0
    print_every = 5
    print("--Training starting--")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
           
            inputs, labels = inputs.to(device), labels.to(device)
            model = model.to(device)
                

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader)*100:.3f}%")
                running_loss = 0
                model.train()
         
    return train_data,model

def main():
   
    model,optimizer,criterion = setup()
    train_data,model = train_model()
    
    functions.save_checkpoint(train_data,model, path, structure, hidden_units, dropout, lr, epochs)
    
    
if __name__ == "__main__":
    main()
#reference
#https://github.com/vishalnarnaware/Create-your-own-Image-Classifier/blob/master/train.py