import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Define the dimensions and number of epochs
n0 = 3072  # Input dimension (CIFAR-10 images are 32x32x3)
d = 512  # Width of the hidden layers
num_epochs = 20  # Number of epochs for training

# Step 1: Data Preparation
# Define transformations and load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Filter out classes 'airplane' and 'automobile' from training and test sets
train_targets = cifar10_train.targets
test_targets = cifar10_test.targets

airplane_automobile_indices = [0, 1]  # Assuming 'airplane': 0, 'automobile': 1
train_indices = [i for i, target in enumerate(train_targets) if target in airplane_automobile_indices]
test_indices = [i for i, target in enumerate(test_targets) if target in airplane_automobile_indices]

subset_train = Subset(cifar10_train, train_indices)
subset_test = Subset(cifar10_test, test_indices)

train_loader = DataLoader(subset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(subset_test, batch_size=64, shuffle=False)

# Step 2: Model Definition
# Define the MLP model with 3 layers and ReLU activation
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model with given σ values
model = MLP(n0, d, 1)
model.fc1.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0 / n0)))
model.fc2.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0 / d)))
model.fc3.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0))/torch.tensor(d))

# Step 3: Training Loop
# Define different learning rates for each layer
optimizer = optim.SGD([
    {'params': model.fc1.parameters(), 'lr': 0.1 * d / n0},
    {'params': model.fc2.parameters(), 'lr': 0.1},
    {'params': model.fc3.parameters(), 'lr': 0.1 / d}
], momentum=0.9)

# Define a binary cross-entropy loss for two classes
criterion = nn.BCEWithLogitsLoss()

# Training loop with specified learning rates
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        targets = targets.float().view(-1, 1)  # Reshape for BCE loss
        binary_targets = torch.where(targets == 0, torch.tensor([0.0]), torch.tensor([1.0]))  # Convert to binary
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, binary_targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.0).float()
        total += binary_targets.size(0)
        correct += (predicted == binary_targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

# Step 4: Evaluation
# Evaluate the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc='Evaluating', unit='batch'):
        targets = targets.float().view(-1, 1)  # Reshape for BCE loss
        binary_targets = torch.where(targets == 0, torch.tensor([0.0]), torch.tensor([1.0]))  # Convert to binary
        outputs = model(inputs)
        predicted = (outputs > 0.0).float()
        total += binary_targets.size(0)
        correct += (predicted == binary_targets).sum().item()

test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.4f}')
