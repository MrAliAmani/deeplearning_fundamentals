# TODO run on cpu type error pytorch 1.11 anaconda update
#  tensorboard visualization and compare to pca 2d
#  200 epochs better performance


import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class Encoder(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_hidden3, n_out) -> None:
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, n_hidden1, bias=True),
            nn.BatchNorm1d(n_hidden1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden2, bias=True),
            nn.BatchNorm1d(n_hidden2),
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden2, n_hidden3, bias=True),
            nn.BatchNorm1d(n_hidden3),
            nn.Sigmoid()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden3, n_out, bias=True),
            nn.BatchNorm1d(n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class Decoder(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_hidden3, n_out) -> None:
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_in, n_hidden1, bias=True),
            nn.BatchNorm1d(n_hidden1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden2, bias=True),
            nn.BatchNorm1d(n_hidden2),
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden2, n_hidden3, bias=True),
            nn.BatchNorm1d(n_hidden3),
            nn.Sigmoid()
        )
        n_size = math.floor(math.sqrt(n_out))
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden3, n_out, bias=True),
            nn.BatchNorm1d(n_out),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=torch.Size([1, n_size, n_size]))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def corrupt_input(x):
    corrupting_matrix = 2.0 * torch.rand_like(x)
    return x * corrupting_matrix


encoder = Encoder(n_in=784, n_hidden1=1000, n_hidden2=500, n_hidden3=250, n_out=2)
decoder = Decoder(n_in=2, n_hidden1=250, n_hidden2=500, n_hidden3=1000, n_out=784)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder, decoder = encoder.to(device), decoder.to(device)

loss_fn = nn.MSELoss()
lr = 1e-3
optimizer = optim.Adam(decoder.parameters(), lr=lr,
                       betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

trainset = datasets.MNIST('.', train=True, transform=transforms.
                          ToTensor(), download=True)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
print("trainset[0][0].shape: ", trainset[0][0].shape)

corrupt = 1.0
trainset_corrupt = []
for i in range(trainset_corrupt.shape[0]):
    for j in range(trainset_corrupt.shape[1]):
        trainset_corrupt[i][j] = corrupt_input(trainset[i][j]) * \
                                 corrupt + trainset[i][j] * (1 - corrupt)
trainloader_corrupt = DataLoader(trainset_corrupt, batch_size=32, shuffle=True)

NUM_EPOCHS = 5
loss_lt = []
for epochs in range(NUM_EPOCHS):
    for input, label in trainloader_corrupt:
        if torch.cuda.is_available():
            input, label = input.cuda(), label.cuda()
        optimizer.zero_grad()
        code = encoder(input)
        output = decoder(code)
        # print("input.shape: {}, output.shape: {}".format(input.shape, output.shape))
        loss = loss_fn(output, input)
        loss_lt.append(loss)
        optimizer.step()
    print(f"Epoch: {epochs}, Loss: {loss}")

i = 0
with torch.no_grad():
    for images, labels in trainloader_corrupt:
        if torch.cuda.is_available():
            images, labels = input.cuda(), label.cuda()
        if i == 3:
            break
        grid = utils.make_grid(images).cpu()
        plt.figure()
        plt.imshow(grid.permute(1, 2, 0))

        code = encoder(images)
        outout = decoder(code)

        grid = utils.make_grid(output).cpu()
        plt.figure()
        plt.imshow(grid.permute(1, 2, 0))
        i += 1
# $ tensorboard --logdir ~/path/to/mnist_autoencoder_hidden=2_logs
# http://localhost:6006/
