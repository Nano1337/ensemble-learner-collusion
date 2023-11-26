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
from mup import MuReadout, MuAdam, get_shapes, make_base_shapes, set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data

import pickle
import numpy as np

import torch
import numpy as np
import random

def seed_everything(seed=42):
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # If you are using CUDA (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


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
def train_model(model, train_loader, optimizer, criterion, epoch):
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
    model.train()
    loop = tqdm(train_loader, leave=True)
    for data, target in loop:
        data, target = data.to('cuda'), target.to('cuda')
        loop.set_description(f'Epoch [{epoch+1}/{80}]')
        optimizer.zero_grad() 
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # average outputs for inference
        # output = torch.stack(outputs).mean(0)
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
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating', leave=True):            
            data, target = data.to('cuda'), target.to('cuda')
            output = model_ensemble(data)
            # output = torch.stack(output).mean(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)')

class ModelClass(nn.Module):
    def __init__(self, width=120): # try 130 as alternative
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, width)
        self.fc2 = nn.Linear(width, int(width*0.5))
        self.fc3 = MuReadout(int(width*0.5), 10, readout_zero_init=True)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))



def save_base_shapes(model, save_path):
    base_shapes = get_shapes(model)
    delta_shapes = get_shapes(ModelClass(width=130))  # Create another instance for delta shapes
    make_base_shapes(base_shapes, delta_shapes, savefile=save_path)

# Main function
def main(num_models=3, do_coord_check=False):
    train_loader, val_loader = get_data_loaders()

    model = ModelClass()
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
            save_to=f'{prm.lower()}_convnet_adam_coord_singlemodel.png',
            suptitle=f'{prm} ConvNet ADAM lr={0.1} nseeds={20}',
            face_color=None
        )

    for epoch in range(80):
        train_model(model, train_loader, optimizer, criterion, epoch)
        validate_model(model, val_loader, criterion)
        if epoch in [29, 49, 69]:
            scheduler.step()

if __name__ == "__main__":
    main(num_models=1, do_coord_check=False)  # Change this value to set the number of models in the ensemble
