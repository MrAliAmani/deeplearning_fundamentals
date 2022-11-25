# TODO ImageNet download and run


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet, CIFAR10
from torchvision.models import resnet34
from torchvision.transforms import ToTensor


class ResidualBlock(nn.Module):
    def __init__(self, in_layers, out_layers, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_layers, out_layers, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_layers)
        self.conv2 = nn.Conv2d(out_layers, out_layers, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_layers)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(inp)
        out = self.bn2(out)
        if self.downsample:
            inp = self.downsample(inp)
        out += inp
        return out


class ResidualNet34(nn.Module):
    def __init__(self):
        super(ResidualNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.comp1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.comp2 = nn.Sequential(
            ResidualBlock(64, 128, downsample=downsample1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.comp3 = nn.Sequential(
            ResidualBlock(128, 256, downsample=downsample2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        downsample3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.comp4 = nn.Sequential(
            ResidualBlock(256, 512, downsample=downsample3),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, inp):
        out = self.conv1(inp)

        out = self.comp1(inp)
        out = self.comp2(inp)
        out = self.comp3(inp)
        out = self.comp4(inp)

        out = self.avgpool(out)
        out = torch.flatten(out, dims=1)
        out = self.fc(out)

        return out


train_dataset = CIFAR10('.', train=True,
                        transform=ToTensor(), download=True)
test_dataset = CIFAR10('.', train=False,
                       transform=ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

lr = 1e-4
num_epochs = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet34().to(device=device)
# model = ResidualNet34().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                       eps=1e-8, weight_decay=0, amsgrad=False)

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
