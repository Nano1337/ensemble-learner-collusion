import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from einops.layers.torch import Rearrange, Reduce

from torch.func import stack_module_state
import copy

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
        print(data.shape)
        exit()
        data, target = data.to('cuda'), target.to('cuda')
        loop.set_description(f'Epoch [{epoch+1}/{epoch}]')
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
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Rearrange('b c h w -> b (c h w)'), # Flatten
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        ).to('cuda')

    def forward(self, x):
        return self.model(x)

# Main function
def main(num_models=3):
    """
    Main function to execute the training and validation process.

    Args:
        num_models (int): Number of models in the ensemble.
    """
    train_loader, val_loader = get_data_loaders()


    # input shape is [128, 3, 32, 32]
    # output shape is [128, 10]

    # Assuming your ModelClassWrapper accepts (128, 3, 32, 32) shaped input
    data = torch.randn(128, 3, 32, 32).to('cuda')

    # Initialize your models
    models = [ModelClass().to('cuda') for _ in range(num_models)]

    # Copy a single model and put it on the meta device
    base_model = copy.deepcopy(models[0]).to('meta')

    # Stack the parameters and buffers of all models
    params, buffers = torch.func.stack_module_state(models)

    # Function to call a single model
    def call_single_model(params, buffers, data):
        return torch.func.functional_call(base_model, (params, buffers), (data,))

    # Apply the vmap with same minibatch for all models
    model_ensemble = torch.vmap(call_single_model, (0, 0, None))(params, buffers, data)
    print("Model ensemble shape", model_ensemble.shape) # [num_models, 128, 10]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ensemble.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.2)

    for epoch in range(80):
        train_model(model_ensemble, train_loader, optimizer, criterion, epoch)
        validate_model(model_ensemble, val_loader, criterion)
        if epoch in [29, 49, 69]:
            scheduler.step()

if __name__ == "__main__":
    main(num_models=5)  # Change this value to set the number of models in the ensemble
