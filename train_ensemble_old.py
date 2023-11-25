import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

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

# Define the ResNet-18 model
def get_model():
    """
    Creates and returns a ResNet-18 model.

    Returns:
        torch.nn.Module: ResNet-18 model.
    """
    return resnet18(weights=None, num_classes=10)

# Define the model ensemble and run forward pass in parallel
class ModelEnsemble(nn.Module):
    def __init__(self, num_models):
        """
        Initialize the model ensemble.

        Args:
            num_models (int): Number of models to include in the ensemble.
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList([get_model() for _ in range(num_models)])

    def forward(self, x):
        """
        Forward pass through the ensemble. Averages the softmax probabilities from all models.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Averaged softmax probabilities.
        """
        # Wrap models in nn.DataParallel
        parallel_models = [nn.DataParallel(model) for model in self.models]

        # Perform parallel inference
        softmax_outputs = [nn.Softmax(dim=1)(model(x)) for model in parallel_models]

        # Average the softmax probabilities
        avg_output = torch.mean(torch.stack(softmax_outputs), dim=0)
        return avg_output

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
        target = target.to('cuda')
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
            target = target.to('cuda')
            output = model_ensemble(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)')

# Main function
def main(num_models=3):
    """
    Main function to execute the training and validation process.

    Args:
        num_models (int): Number of models in the ensemble.
    """
    train_loader, val_loader = get_data_loaders()
    model_ensemble = ModelEnsemble(num_models)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ensemble.parameters(), lr=0.1, weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.2)

    for epoch in range(80):
        train_model(model_ensemble, train_loader, optimizer, criterion, epoch)
        validate_model(model_ensemble, val_loader, criterion)
        if epoch in [29, 49, 69]:
            scheduler.step()

if __name__ == "__main__":
    main(num_models=5)  # Change this value to set the number of models in the ensemble
