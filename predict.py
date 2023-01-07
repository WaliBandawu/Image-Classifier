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
import matplotlib.pyplot as plt
import json
from PIL import Image
import functions

#Arguments and their default values
parser = argparse.ArgumentParser(description = 'Parser for predict.py')

parser.add_argument('image', default='./flowers/test/1/image_06743.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
#arguments assignment
args = parser.parse_args()
image = args.image
possible_outputs = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

def main():
    model=functions.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as f:
         cat_to_name = json.load(f)
        
    probabilities = functions.predict(image, model, possible_outputs, device)
    probability = np.array(probabilities[0][0])
    labels = [cat_to_name [str(index + 1)] for index in np.array(probabilities[1][0])]
    
    i = 0
    while i < possible_outputs:
        print("{} with a probability of {}%".format(labels[i], probability[i]*100 ))
        i += 1
    print("Completed!")

    
if __name__== "__main__":
    main()
#reference
#https://github.com/vishalnarnaware/Create-your-own-Image-Classifier/blob/master/predict.py
