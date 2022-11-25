# downloaded googlenews.bin.gz manually
# wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz -O - | gunzip | cut -f1,2 -d" " > pos.train.txt
# wget http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz -O - | gunzip | cut -f1,2 -d " " > pos.test.txt


# TODO leveldb, windows 8.1 SDK, build tools v140, run


import re
import leveldb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

model = KeyedVectors.load_word2vec_format('./googlenews.bin.gz', binary=True)


def create_pos_dataset(filein, fileout):
    dataset = []
    with open(filein) as f:
        dataset_raw = f.readlines()
        dataset_raw = [e.split() for e in dataset_raw if len(e.split()) > 0]

    counter = 0
    while counter < len(dataset_raw):
        pair = dataset_raw[counter]
        if counter < len(dataset_raw) - 1:
            next_pair = dataset_raw[counter + 1]
            if pair[0] + "_" + next_pair[0] in model and (pair[1] == next_pair[1]):
                dataset.append([pair[0] + "_" + next_pair[0], pair[1]])
                counter += 2
                continue

        word = re.sub("\d", "#", pair[0])
        word = re.sub("-", "_", word)

        if word in model:
            dataset.append([word, pair[1]])
            counter += 1
            continue

        if "_" in word:
            subwords = word.split("_")
            for subword in subwords:
                if not (subword.isspace() or len(subword) == 0):
                    dataset.append([subword, pair[1]])
                counter += 1
                continue

        dataset.append([word, pair[1]])
        counter += 1

    with open(fileout, 'w') as processed_file:
        for item in dataset:
            processed_file.write("%s\n" % (item[0] + " " + item[1]))

    return dataset


train_pos_dataset = create_pos_dataset('./pos.train.txt',
                                       './pos.train.processed.txt')
test_pos_dataset = create_pos_dataset('./pos.test.txt',
                                      './pos.test.processed.txt')

print("train_pos_dataset[0]: {}, test_pos_dataset[0]: {}"
      .format(train_pos_dataset[0], test_pos_dataset[0]))

print("len(train_pos_dataset): {}, len(test_pos_dataset): {}"
      .format(len(train_pos_dataset), len(test_pos_dataset)))

db = leveldb.LevelDB("./word2vecdb")

counter = 0
dataset_vocab = {}
tags_to_index = {}
index_to_tags = {}
index = 0
for pair in train_pos_dataset + test_pos_dataset:
    if pair[0] not in dataset_vocab:
        dataset_vocab[pair[0]] = index
        index += 1
    if pair[1] not in tags_to_index:
        tags_to_index[pair[1]] = counter
        index_to_tags[counter] = pair[1]
        counter += 1

    nonmodel_cache = {}

    counter = 1
    total = len(dataset_vocab.keys())
    for word in dataset_vocab:

        if word in model:
            db.Put(bytes(word, 'utf_8'), model[word])
        elif word in nonmodel_cache:
            db.Put(bytes(word, 'utf_8'), nonmodel_cache[word])
        else:
            print(word)
            nonmodel_cache[word] = np.random.uniform(-0.25, 0.25, 300).astype(np.float32)
            db.Put(bytes(word, 'utf_8'), nonmodel_cache[word])
            counter += 1

x = db.Get(bytes('Confidence', 'utf_8'))
print("np.frombuffer(x, dtype=float).shape", np.frombuffer(x, dtype='float32').shape)


class NgramPOSDataset(Dataset):
    def __init__(self, db, dataset, tags_to_index, n_grams):
        self.db = db
        self.dataset = dataset
        self.tags_to_index = tags_to_index
        self.n_grams = n_grams

    def __getitem__(self, index):
        ngram_vector = np.array([])

        for ngram_index in range(index, index + self.n_grams):
            word, _ = self.dataset(ngram_index)
            vector_bytes = self.db.Get(bytes, 'utf_8')
            vector = np.frombuffer(vector_bytes, dtype='float32')
            ngram_vector = np.append(ngram_vector, vector)

            _, tag = self.dataset(index + int(np.floor(self.n_grams / 2)))
            label = self.tags_to_index[tag]
        return torch.tensor(ngram_vector, dtype='float32'), index

    def __len__(self):
        return len(self.dataset) - self.n_grams + 1


trainset = NgramPOSDataset(db, train_pos_dataset, tags_to_index, 3)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

print("len(tags_to_index.keys()", len(tags_to_index.keys()))


class FeedForwardPOS(nn.Module):
    def __init__(self, n_inputs, n_hidden_1, n_hidden_2, n_outputs):
        super(FeedForwardPOS, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_inputs, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.Sigmoid()
        )
        self.layer_out = nn.Sequential(
            nn.Linear(n_hidden_2, n_outputs),
            nn.BatchNorm1d(n_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer_out(x)


n_gram = 3
embedding_size = 300
n_hidden_1 = 512
n_hidden_2 = 256
n_outputs = len(tags_to_index.keys())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardPOS(n_gram * embedding_size,
                       n_hidden_1, n_hidden_2, n_outputs).to(device)

training_epochs = 10
batch_size = 32

loss_fn = nn.CrossEntropyLoss()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                       eps=1e-8, weight_decay=0, amsgrad=False)

trainset = NgramPOSDataset(db, train_pos_dataset, tags_to_index, n_gram)
trainloader = DataLoader(trainset, batch_size, shuffle=True)

writer = SummaryWriter()

loss_lt = []
for epoch in range(training_epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        torch.autograd.backward(loss)
        optimizer.step()
        running_loss += loss
    loss_lt.append(running_loss)

    writer.add_scalar("Loss/train", running_loss / len(trainloader), epoch)
    '''if(epoch % 100 == 0):
        print(f"Epoch: {epoch}, Loss: {running_loss/len(trainloader)}")'''
    print(f"Epoch: {epoch}, Loss: {running_loss / len(trainloader)}")

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_lt
}, 'ngram_pos_model_training.pt')

testset = NgramPOSDataset(db, test_pos_dataset, tags_to_index, 3)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

num_correct = 0
while torch.no_grad():
    for inputs, labels in trainloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        _, index = output.max(1)
        print(f"output: {output}, index: {index}, labels: {labels}")
        num_correct = (index == labels).sum()
print(f'Accuracy: {num_correct*100/len(testloader)*batch_size}')

print(f'num_correct: {num_correct}, len(testloader): {len(testloader)}, labels: {labels}')


# $ tensorboard --logdir ~/path/to/mnist_autoencoder_hidden=2_logs
# http://localhost:6006/
