# TODO
#  CIFAR_10,
#  other architectures,
#  hidden layer num,
#  optimizers AdaDelta, RMSProp with Momentum, adamax


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


class BaseClassifier(Dataset):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)


train_dataset = MNIST(".", train=True, download=True,
                      transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True,
                     transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=64,
                         shuffle=False)


in_dim, feature_dim, out_dim = 784, 256, 10
lr = 1e-3
loss_fn = nn.CrossEntropyLoss()
epochs = 40
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = classifier.model.to(device=device)

'''optimizer = optim.SGD(params=classifier.model.parameters(),
                      lr=lr, momentum=0.9)'''

'''optimizer = optim.SGD(params=classifier.model.parameters(),
                      lr=lr, momentum=0.9, nesterov=True)'''

'''optimizer = optim.Adagrad(model.parameters(), lr=lr,
                    weight_decay=0, initial_accumulator_value=0)'''

'''optimizer = optim.RMSprop(model.parameters(), lr=lr,
                          alpha=0.99, eps=1e-8,
                          weight_decay=0, momentum=0)'''

optimizer = optim.Adam(model.parameters(), lr=lr,
                       betas=(0.9, 0.999),
                       eps=1e-8, weight_decay=0,
                       amsgrad=False)


def train(classifier=model, optimizer=optimizer,
          epochs=epochs, loss_fn=loss_fn):
    classifier.train()
    loss_lt = []
    for epoch in range(epochs):
        running_loss = 0
        for minibatch in train_loader:
            data, target = minibatch
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data = data.flatten(start_dim=1)
            out = classifier.forward(data)
            computed_loss = loss_fn(out, target)
            torch.autograd.backward(computed_loss)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += computed_loss.item()
        loss_lt.append(running_loss/len(train_loader))
        print("Epoch {}, train loss: {}"
              .format(epoch+1, running_loss/len(train_loader)))

    plt.plot([i for i in range(1, epochs+1)], loss_lt)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MNIST Training Loss: Optimizer {}, lr {}"
              .format("SGD", lr))

    torch.save(classifier.state_dict(), 'mnist.pt')


def test(classifier=model, loss_fn=loss_fn):
    classifier.eval()
    accuracy = 0
    computed_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier.forward(data)
            _, preds = out.max(dim=1)
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)

        print("Test Loss: {}, total accuracy: {}".format(
            computed_loss/(len(test_loader)*64),
            accuracy*100.0/(len(test_loader)*64)
        ))


train()
test()
