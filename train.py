'''
This script sets up and trains model, preprocessing images with help from load.py.
'''


## IMPORT DEPENDENCIES
import load
import torch
from torch import nn, optim
import argparse
import os


## COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description="Training a transfer learning neural net")
parser.add_argument('data_directory', action='store', help='Set root directory for images')
parser.add_argument('--save_dir', action='store', help='Set save location of trained model checkpoint', default='.')
parser.add_argument('--arch', action='store', help='Choose pre-trained neural net architecture', default="vgg16", choices=['vgg16', 'resnet18'])
parser.add_argument('--learning_rate', action='store', help='Set learning rate', default=0.001, type=int)
parser.add_argument('--hidden_units', action='store', help='Set units in hidden layer', default=4096//2, type=int)
parser.add_argument('--epochs', action='store', help='Set number of epochs', type=int, default=5)
parser.add_argument('--gpu', action='store_true', default=False, help='Toggle GPU')
results = parser.parse_args()


## SET VARIABLES AND DEFAULTS
data_dir = results.data_directory
save_dir = results.save_dir
epochs = results.epochs
gpu = results.gpu
lr = results.learning_rate
arch = results.arch
hidden_units = results.hidden_units


## LOAD DATA
loaders = load.data_paths(data_dir)
class_mapping, train_loader, valid_loader, test_loader = load.data_loader(*loaders)
num_classes = 0
for entry in os.scandir(loaders[0]):
   if entry.is_dir():
        num_classes += 1

## CHECK DEVICE
device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")


## LOAD MODEL, DEFINE OBJECTIVE FUNCTION, OPTIMIZER
model = load.Model(num_classes, arch=arch, hidden_layers=hidden_units).model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.last_layer.parameters(), lr=lr)
model.to(device)


## TRAINING
steps = 0
running_loss = 0
print_every = 100
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Validation to see model training progress
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:

                    # Calculate loss for validation batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy for validation batch
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}, Step {steps}.. ", 
                          f"Train loss: {running_loss/print_every:.3f}.. ", 
                          f"Validation loss: {valid_loss/len(valid_loader):.3f}.. ", 
                          f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                    running_loss = 0
                    model.train()


## TESTING TRAINED MODEL
batch_counter = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        batch_counter += 1

        # Calculate logits for test batch
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)

        # Calculate accuracy for test batch
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(test_loader):.3f}.. ", f"Test batches: {batch_counter}")


## SAVING MODEL TO CHECKPOINT                            
model_checkpoint_dict = {}
model_checkpoint_dict['state_dict'] = model.state_dict()
model_checkpoint_dict['num_classes'] = num_classes
model_checkpoint_dict['arch'] = arch
model_checkpoint_dict['hidden_units'] = hidden_units
model_checkpoint_dict['epochs'] = epochs
model_checkpoint_dict['class_mapping'] = class_mapping
save_location = os.path.join(save_dir, 'checkpoint.pth')
torch.save(model_checkpoint_dict, save_location)
