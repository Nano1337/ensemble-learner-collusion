import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from einops.layers.torch import Rearrange
from einops import rearrange

import mup
from mup import MuReadout, MuAdam, get_shapes, make_base_shapes, set_base_shapes

import pickle

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
    def __init__(self, width=120): # try 130 as alternative
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, width)
        self.fc2 = nn.Linear(width, int(width*0.7))
        self.fc3 = MuReadout(int(width*0.7), 10)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class ModelEnsemble(nn.Module): 
    def __init__(self, num_models): 
        super().__init__() 
        self.num_models = num_models 

        models = []

        for i in range(num_models): 
            model = ModelClass()
            set_base_shapes(model, 'single_base.bsh')
            models.append(model)

        self.models = nn.ModuleList(models) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        parallel_outputs = [model(x) for model in self.models]
        softmax_outputs = [self.softmax(output) for output in parallel_outputs]

        # Average the softmax probabilities
        avg_output = torch.mean(torch.stack(softmax_outputs), dim=0)
        return avg_output


def save_base_shapes(model, save_path):
    base_shapes = get_shapes(model)
    delta_shapes = get_shapes(ModelClass(width=130))  # Create another instance for delta shapes
    make_base_shapes(base_shapes, delta_shapes, savefile=save_path)

# Main function
def main(num_models=3):
    train_loader, val_loader = get_data_loaders()

    model_ensemble = ModelEnsemble(num_models).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = MuAdam(model_ensemble.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.2)

    for epoch in range(80):
        train_model(model_ensemble, train_loader, optimizer, criterion, epoch)
        validate_model(model_ensemble, val_loader, criterion)
        if epoch in [29, 49, 69]:
            scheduler.step()

if __name__ == "__main__":
    main(num_models=3)  # Change this value to set the number of models in the ensemble
