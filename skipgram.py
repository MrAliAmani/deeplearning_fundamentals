# TODO run nce project on github
#  t_sne visualization
#  pytorch 1.11, run
#  error on colab: ValueError: <query>, <positive_key. must have 2 dimensions.


import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn, optim
from info_nce import info_nce
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

outputs = torch.randint(low=0, high=100, size=(100, 100))
targets = torch.randint(low=0, high=100, size=(100, 100))

trainset = CIFAR10('.', train=True,
                   transform=ToTensor(), download=True)
testset = CIFAR10('.', train=False,
                  transform=ToTensor(), download=True)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
# TODO validation set from Imagenet
valloader = testloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 500
emb_vector_len = 128
embedding = nn.Embedding(num_embeddings=vocab_size,
                         embedding_dim=emb_vector_len).to(device)

optimizer = optim.SGD(embedding.parameters(), lr=1e-1)
# loss_fn = info_nce()
batch_size, embedding_size = 32, 128


def train(inputs, targets, embedding=embedding):
    optimizer.zero_grad()
    input_emb = embedding(inputs)
    target_emb = embedding(targets)
    loss = info_nce(input_emb, target_emb)
    torch.autograd.backward(loss)
    optimizer.step()
    return loss


cosine_sim = nn.CosineSimilarity()


def evaluate(inputs, targets, embedding=embedding):
    with torch.no_grad():
        input_emb = embedding(inputs)
        target_emb = embedding(targets)
        norm = torch.sum(input_emb, dim=1)
        normalized = input_emb / norm
        score = cosine_sim(normalized, target_emb)
        return normalized, score


n_epochs = 1
writer = SummaryWriter()
for epoch in range(n_epochs):
    # train
    running_loss = 0.0
    for inputs, targets in trainloader:
        # print('inputs: {}, targets: {}'.format(inputs, targets))
        inputs = inputs.to(torch.int64)
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        loss = train(inputs, targets, embedding)
        running_loss += loss

    writer.add_scalar('Train Loss',
                      running_loss / len(trainloader), epoch)

    # validate
    running_score = 0.0
    for inputs, targets in valloader:
        inputs = inputs.to(torch.int64)
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        _, score = evaluate(inputs, targets, embedding)
        running_score += score

    writer.add_scalar('Val Loss',
                      running_score / len(valloader), epoch)

'''tsne = TSNE(perplexity=30, n_components=2, init='pca',
            n_iter=5000)
plot_embeddings = np.asfarray(final_embeddings[:plot_num, :],
                              dtype=float)
low_dim_embs = tsne.fit_transform(plot_embeddings)
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
data.plot_with_labels(low_dim_embs, labels)
'''
