# TODO validation code for hyperparameters: lr, number of latent variables
#  RNNs, hyperparameter tuning, and longer training times

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal \
    import MultivariateNormal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(VAE, self).__init__()
        self.D_in, self.H, self.D_out = D_in, H, D_out

        self.input_layer = nn.Linear(D_in, H)
        self.hidden_layer_mean = nn.Linear(H, D_out)
        self.hidden_layer_var = nn.Linear(H, D_out)

        self.recon_layer = nn.Linear(D_out, H)
        self.recon_output = nn.Linear(H, D_in)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, inp):
        h_vec = self.input_layer(inp)
        h_vec = self.sigmoid(h_vec)
        means = self.hidden_layer_mean(h_vec)
        log_vars = self.hidden_layer_var(h_vec)
        return means, log_vars

    def decode(self, means, log_vars):
        std_devs = torch.pow(2, log_vars)**0.5
        aux = MultivariateNormal(torch.zeros(self.D_out),
                                 torch.eye(self.D_out)).sample()
        sample = means + aux * std_devs

        h_vec = self.recon_layer(sample)
        h_vec = self.tanh(h_vec)
        output = self.sigmoid(self.recon_output(h_vec))
        return output

    def forward(self, inp):
        means, log_vars = self.encode(inp)
        output = self.decode(means, log_vars)
        return output, means, log_vars

    def reconstruct(self, sample):
        h_vec = self.recon_layer(sample)
        h_vec = self.tanh(h_vec)
        output = self.sigmoid(self.recon_output(h_vec))
        return output


def compute_loss(inp, recon_inp, means, log_vars):
    kl_loss = -0.5 * torch.sum(1 + log_vars -
                               means ** 2 - torch.pow(2, log_vars))
    loss = nn.BCELoss(reduction='sum')
    recon_loss = loss(recon_inp, inp)
    return kl_loss + recon_loss


D_in, H, D_out = 784, 500, 20
vae = VAE(D_in, H, D_out)
vae.to('cpu')


def train():
    vae.train()
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    trainloader = DataLoader(
        datasets.MNIST('.', train=True,
                       transform=transforms.ToTensor(), download=True),
        batch_size=100, shuffle=True
    )

    epochs = 10
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(trainloader):
            optimizer.zero_grad()
            data = data.view((100, 784))
            output, means, log_vars = vae(data)
            loss = compute_loss(data, output, means, log_vars)
            torch.autograd.backward(loss)
            optimizer.step()
            if (batch_idx * len(data)) % 10000 == 0:
                print(
                    'Train epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'
                    .format(
                        epoch, batch_idx * len(data), len(trainloader.dataset),
                        100. * batch_idx * len(trainloader), loss
                    )
                )
        torch.save(vae.state_dict(), 'vae.%d' % epoch)


def test():
    dist = MultivariateNormal(torch.zeros(D_out), torch.eye(D_out))
    vae = VAE(D_in, H, D_out)
    vae.load_state_dict(torch.load("vae.%d" % 9))
    vae.eval()
    outputs = []

    for i in range(100):
        sample = dist.sample()
        outputs.append(vae.reconstruct(sample).view((1, 1, 28, 28)))
    outputs = torch.stack(outputs).view(100, 1, 28, 28)
    save_image(outputs, "prior_reconstruct_100.png", nrow=10)


train()

test()
