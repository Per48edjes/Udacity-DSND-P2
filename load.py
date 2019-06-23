'''
This file contains functions that load the image data and for training and contain
the class definitions for the model to be trained.
'''


## DEPENDENCIES
import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import os


## HELPER FUNCTIONS
def data_paths(data_dir):
    '''
    This function sets the training, validation, and testing directories.
    '''
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    return train_dir, valid_dir, test_dir


def data_loader(train_dir, valid_dir, test_dir):
    '''
    This function does the necessary transformations and returns loaders.
    '''
    # Normalization parameters of ImageNet dataset
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(), 
                                           transforms.Normalize(norm_mean, norm_std)])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize(norm_mean, norm_std)])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    
    # Class mappings to be stored in saved model checkpoint
    class_mapping = test_dataset.class_to_idx

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    return  class_mapping, train_loader, valid_loader, test_loader


def checkpoint_loader(PATH, gpu):
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(PATH, map_location=device)

    # Import pretrained model
    num_classes = checkpoint['num_classes']
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = Model(num_classes, arch, hidden_units).model
    model.class_to_idx = checkpoint['class_mapping']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


## PROCESS IMAGES FOR INFERENCE
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    im = Image.open(image)
    
    ## Resize & crop

    # Resizing
    size = 256, 256
    im.thumbnail(size, Image.ANTIALIAS)    
    
    width, height = im.size
    new_width, new_height = 224, 224
    
    # Cropping
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = im.crop((left, top, right, bottom))
    
    ## Convert to NumPy array
    np_im = np.array(im)/255
    
    # Normalize array
    np_im = (np_im - norm_mean) / norm_std
    np_im = np_im.transpose((2,0,1))
    
    return np_im


## MODEL CLASS
class Model:
    
    def __init__(self, num_classes, arch, hidden_layers):        
        
        # Changing classifier based on model selection
        if arch == "vgg16":
            self.model = models.vgg16(pretrained=True)
            input_layers = 25088
            for param in self.model.parameters():
                param.requires_grad = False    
            self.model.classifier = nn.Sequential(nn.Linear(input_layers, hidden_layers), 
                                                  nn.ReLU(), 
                                                  nn.Dropout(p=0.2), 
                                                  nn.Linear(hidden_layers, num_classes),
                                                  nn.LogSoftmax(dim=1))
            self.model.last_layer = self.model.classifier
            
        if arch == "resnet18":
            self.model = models.resnet18(pretrained=True)
            input_layers = self.model.fc.in_features
            for param in self.model.parameters():
                param.requires_grad = False    
            self.model.fc = nn.Sequential(nn.Linear(input_layers, hidden_layers), 
                                     nn.ReLU(), 
                                     nn.Dropout(p=0.2), 
                                     nn.Linear(hidden_layers, num_classes), 
                                     nn.LogSoftmax(dim=1))
            self.model.last_layer = self.model.fc
        
    
