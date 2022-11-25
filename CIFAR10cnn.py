import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            #  TODO 1 instead of 3??
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25)
        )
        self.block2 = nn.Sequential(
            nn.Flatten(),
            #  TODO 9216 instead of 12544??
            nn.Linear(in_features=12544, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=10),
            nn.BatchNorm1d(num_features=10)
        )

    def forward(self, x):
        x = self.block1(x)
        return self.block2(x)


train_dataset = CIFAR10('.', train=True,
                        transform=ToTensor(), download=True)
test_dataset = CIFAR10('.', train=False,
                       transform=ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

lr = 1e-4
num_epochs = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                       eps=1e-8, weight_decay=0, amsgrad=False)

for epochs in range(num_epochs):
    running_loss = 0.0
    num_correct = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        torch.autograd.backward(loss)
        running_loss += loss
        optimizer.step()
        _, idx = outputs.max(dim=1)
        num_correct += (idx == labels).sum()

    print("Epoch: {} Loss: {}, Accuracy: {}".format(
        epochs + 1, running_loss / len(train_loader),
        num_correct / len(train_loader)))

transform = transforms.Compose([
    transforms.RandomCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0, contrast=0,
                           saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])
