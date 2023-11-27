import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from einops.layers.torch import Rearrange
from einops import rearrange

import mup
from mup import MuReadout, MuSharedReadout, MuAdam, get_shapes, make_base_shapes, set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data

import pickle
import numpy as np

import torch
import numpy as np
import random

# Prepare the CIFAR-10 dataset
def get_data_loaders(batch_size=128, augment=True):
    """
    Prepares the data loaders for the CIFAR-10 dataset.

    Args:
        batch_size (int): The size of each batch of data.
        augment (bool): Flag to determine whether to apply data augmentation.

    Returns:
        Tuple[DataLoader, DataLoader]: Returns training and validation data loaders.
    """
    # Define transformations for training and validation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load datasets
    print("Loading CIFAR-10 dataset...")
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Training loop
def train_model(model_ensemble, train_loader, optimizer, criterion, epoch):
    """
    Trains the model ensemble.

    Args:
        model_ensemble (ModelEnsemble): The model ensemble to train.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer to use for training.
        criterion (Loss): Loss function for training.
        epoch (int): The current training epoch.
    """
    print("Starting training...")
    model_ensemble.train()
    loop = tqdm(train_loader, leave=True)
    for data, target in loop:
        data, target = data.to('cuda'), target.to('cuda')
        loop.set_description(f'Epoch [{epoch+1}/{80}]')
        optimizer.zero_grad()
        output = model_ensemble(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == target).float().mean()
        loop.set_postfix(loss=loss.item(), accuracy=round(acc.item()*100, 4))

# Validation loop
def validate_model(model_ensemble, val_loader, criterion):
    """
    Validates the model ensemble.

    Args:
        model_ensemble (ModelEnsemble): The model ensemble to validate.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function for validation.
    """
    model_ensemble.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating', leave=True):            
            data, target = data.to('cuda'), target.to('cuda')
            output = model_ensemble(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)')

class ModelClass(nn.Module):
    def __init__(self, width=120, num_models=1):
        super().__init__()
        self.width = width
        self.num_models = num_models
        self.models = nn.ModuleList([self._create_single_model() for _ in range(num_models)])
        self.softmax = nn.Softmax(dim=1)
        self.classifier = MuReadout(int(self.width*0.5), 10, readout_zero_init=True)

    def _create_single_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, self.width),
            nn.ReLU(),
            nn.Linear(self.width, int(self.width*0.5)),
            nn.ReLU(),
        )
        return model

    def forward(self, x):
        parallel_outputs = [self.classifier(model(x)) for model in self.models]
        # avg_output = torch.mean(torch.stack(parallel_outputs), dim=0)
        # avg_output = self.softmax(avg_output)
        softmax_outputs = [self.softmax(output) for output in parallel_outputs]

        # Average the softmax probabilities
        avg_output = torch.mean(torch.stack(softmax_outputs), dim=0)
        return avg_output

        return avg_output

def save_base_shapes(model, save_path):
    base_shapes = get_shapes(model)
    delta_shapes = get_shapes(ModelClass(width=130))  # Create another instance for delta shapes
    make_base_shapes(base_shapes, delta_shapes, savefile=save_path)


# Main function
def main(num_models=3, do_coord_check=False):
    train_loader, val_loader = get_data_loaders()

    model = ModelClass(width=1200)
    save_base_shapes(model, 'single_base.bsh')

    def gen(w):
        def f():
            assert w*0.5 % 1 == 0 # check that this is an integer
            model = ModelClass(width=w).to('cuda')
            set_base_shapes(model, 'single_base.bsh')
            return model
        return f


    set_base_shapes(model, 'single_base.bsh')
    model=model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = MuAdam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.2)

    if do_coord_check: 

        widths = 2**np.arange(7, 14)
        models = {w: gen(w) for w in widths}

        df = get_coord_data(models, train_loader, mup=mup, lr=0.1, optimizer='adam', flatten_input=False, nseeds=20, nsteps=3, lossfn='xent')

        prm = 'μP'
        return plot_coord_data(
            df, 
            legend=False,
            save_to=f'{prm.lower()}_convnet_adam_coord_{num_models}_model_fusion.png',
            suptitle=f'{prm} {num_models} ConvNet fusion ADAM lr={0.1} nseeds={20}',
            face_color=None
        )

    for epoch in range(80):
        train_model(model, train_loader, optimizer, criterion, epoch)
        validate_model(model, val_loader, criterion)
        if epoch in [29, 49, 69]:
            scheduler.step()

if __name__ == "__main__":
    main(num_models=5, do_coord_check=False) # Change this value to set the number of models in the ensemble