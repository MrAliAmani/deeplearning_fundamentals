import torch.autograd
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.fc1(x)
        return out


train_dataset = MNIST('.', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST('.', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

lr = 1e-4
num_epochs = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTConvNet().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epochs in range(num_epochs):
    running_loss = 0.0
    num_correct = 0
    for inputs, labels in train_loader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        torch.autograd.backward(loss)
        running_loss += loss
        optimizer.step()
        _, idx = outputs.max(dim=1)
        num_correct += (idx == labels).sum()

    print("Epoch: {} Loss: {}, Accuracy: {}".format(
        epochs, running_loss / len(train_loader),
        num_correct / len(train_loader)))


transform = transforms.Compose([
    transforms.RandomCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0, contrast=0,
                           saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])
