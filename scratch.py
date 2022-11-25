import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

import dataset
import fnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

arr = np.array([1, 2])
tensor = torch.from_numpy(arr)
print('tensor.dtype: ', tensor.dtype)

arr = [1, 2]
tensor1 = torch.tensor(arr, dtype=torch.float32, device=device)
print('tensor1.shape: ', tensor1.shape)

tensor2 = tensor1.to(device=device, dtype=torch.int64)

x1_t = torch.zeros([2, 2])
x2_t = torch.ones([2, 2])
add_t = x1_t + x2_t

x3_t = torch.tensor([[1, 2], [3, 4]])
x4_t = torch.tensor([[1, 2, 3], [3, 4, 5]])
mul_t = torch.matmul(x3_t, x4_t)

i, j, k = 0, 1, 1
x5_t = torch.tensor([[[1, 2, 3], [2, 8, 7]], [[1, 9, 3], [5, 8, 4]]])
print('x5_t[i, j , k]: ', x5_t[i, j, k])
print('x5_t[:, 1, :]: ', x5_t[:, 1, :])
print('x5_t[0, 1:3, :]: ', x5_t[0, 1:3, :])

x5_t[0, 0, 0] = 14

x6_t = torch.randn([2, 3, 4])
x7_t = torch.randn(2, 4)
x6_t[0, 1:3, :] = x7_t
x6_t[0, 1:3, :] = 1
x6_t[0, 1:3, :] = torch.randn([1, 4])

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)
f = x ** 2 + y ** 2 + z ** 2
f.backward()
print('x.grad, y.grad, z.grad: ', x.grad, y.grad, z.grad)

in_dim, out_dim = 256, 10
vec = torch.randn(256)
layer = nn.Linear(in_dim, out_dim, bias=True)
out = layer(vec)
# print('out: ', out)

W = torch.rand([10, 256])
b = torch.zeros([10, 1])
out = torch.matmul(W, vec) + b
# print('out: ', out)

in_dim, feature_dim, out_dim = 784, 256, 10
vec = torch.randn(784)
layer1 = nn.Linear(in_dim, feature_dim, bias=True)
layer2 = nn.Linear(feature_dim, out_dim, bias=True)
out = layer2(layer1(vec))

relu = nn.ReLU()
out = layer2(relu(layer1(vec)))
# print('out: ', out)
# __________________________________________________

no_examples = 10
in_dim, feature_dim, out_dim = 784, 256, 10
x = torch.randn((no_examples, in_dim))
classifier = fnn.BaseClassifier(in_dim, feature_dim, out_dim)
out = classifier.forward(x)
print('out: ', out)

loss = nn.CrossEntropyLoss()
target = torch.tensor([0, 4, 2, 8, 5, 4, 5, 1, 2, 3], dtype=torch.long)
computed_loss = loss(out, target)
# torch.autograd.backward(computed_loss)
computed_loss.backward()

for p in classifier.parameters():
    print('parameter shape: ', p.shape)

lr = 1e-3
optimizer = optim.SGD(classifier.parameters(), lr=lr)

# optimizer.step()
# optimizer.zero_grad()


train_dataset = dataset.ImageDataset(img_dir="./data/train/",
                                     label_file="./data/train/labels.npy")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for minibatch in train_loader:
    data, label = minibatch['data'], minibatch['label']
    print('data: ', data)
    print('label: ', label)
    out = classifier(data)
    print('out: ', out)

# momentum random walk
rand_walk = [torch.randint(-10, 10, (1, 1)) for x in range(100)]
momentum_rand_walk = [torch.randint(-10, 10, (1, 1)) for x in range(100)]

momentum_lt = [0.1, 0.5, 0.9, 0.99]
momentum_rand_walk_lt = []
for momentum in momentum_lt:
    for i in range(1, len(rand_walk) - 1):
        prev = momentum_rand_walk[i - 1]
        rand_choice = torch.randint(-10, 10, (1, 1))
        new_step = momentum * prev + (1 - momentum) * rand_choice
        momentum_rand_walk[i] = new_step

    momentum_rand_walk_lt.append(momentum_rand_walk[:-1])

for i in range(len(momentum_lt)):
    plt.figure()
    plt.subplot(5, 2, 1)
    plt.plot(momentum_rand_walk_lt[i])

plt.show()

layer = nn.Conv2d(in_channels=3, out_channels=64,
                  kernel_size=(5, 5),
                  stride=2, padding=1)

layer = nn.BatchNorm2d(num_features=32, eps=1e-5,
                       momentum=0.1, affine=True,
                       track_running_stats=True)

layer = nn.BatchNorm1d(num_features=32)

layer = nn.GroupNorm(num_groups=1, num_channels=32)
