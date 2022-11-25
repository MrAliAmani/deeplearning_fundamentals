import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class Net(Dataset):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.classifier(x)


IN_DIM, FEATURE_DIM, OUT_DIM = 784, 256, 10
classifier = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = classifier.model.to(device=device)
loss_fn = nn.CrossEntropyLoss()
epochs = 40
lr = 1e-4
optimizer = optim.SGD(params=model.parameters(), lr=lr)


train_dataset = MNIST(".", train=True, download=True,
                      transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True,
                     transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=64,
                         shuffle=False)


for param_tensor in model.state_dict():
    print(param_tensor, "\t",
          model.state_dict()[param_tensor].size())


loss_lt = []
for epoch in range(epochs):
    running_loss = 0
    for minibatch in train_loader:
        data, target = minibatch
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data = data.flatten(start_dim=1)
        out = model.forward(data)
        computed_loss = loss_fn(out, target)
        torch.autograd.backward(computed_loss)
        optimizer.step()
        optimizer.zero_grad()
        running_loss += computed_loss.item()
    loss_lt.append(running_loss/len(train_loader))
    print("Epoch {}, train loss: {}"
          .format(epoch+1, running_loss/len(train_loader)))

torch.save(model.state_dict(), 'mnist.pt')


model.load_state_dict(torch.load('mnist.pt'))
opt_state_dict = copy.deepcopy(model.state_dict())

for param_tensor in opt_state_dict:
    print(param_tensor, "\t",
          opt_state_dict[param_tensor].size())


print("model_rand")
model_rand = Net(IN_DIM, FEATURE_DIM, OUT_DIM).model
rand_state_dict = copy.deepcopy(model_rand.state_dict())
#print("rand_state_dict: ", rand_state_dict)

print("model_test")
model_test = Net(IN_DIM, FEATURE_DIM, OUT_DIM).model
test_state_dict = copy.deepcopy(model_test.state_dict())
#print("test_state_dict: ", test_state_dict)


#print("inference")
def inference(test_loader, model, loss_fn):
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss
        running_loss /= len(test_loader)
    return running_loss


print("alpha")
results = []
for alpha in torch.arange(-2, 2, 0.05):
    beta = 1 - alpha

    for p in opt_state_dict:
        test_state_dict[p] = (opt_state_dict[p] * alpha +
                              rand_state_dict[p] * beta)

    model.load_state_dict(test_state_dict)

    loss = inference(train_loader, model, loss_fn=loss_fn)
    results.append(loss)

print("results: ", results)

print("plot")
plt.plot(np.arange(-2, 2, 0.05), results, 'ro')
plt.xlabel("Alpha")
plt.ylabel("Incurred error")
plt.show()

