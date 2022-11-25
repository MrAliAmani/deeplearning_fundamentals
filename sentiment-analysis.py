import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = IMDB(split='train')

tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


text_vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                       specials=['<unk>', '<pad>'])
text_vocab.set_default_index(text_vocab['<unk>'])
print('text_vocab(tokenizer("Hello is it me you\'re looking for?")):')
print(text_vocab(tokenizer("Hello is it me you're looking for?")))


def text_pipeline(x, max_size=512):
    text = tokenizer(x)

    pruned_text = []
    for token in text:
        if text_vocab.get_stoi()[token] > 30000:
            token = '<unk>'
        pruned_text.append(token)

        if len(pruned_text) < max_size:
            pruned_text += ['<pad>'] * (max_size - len(pruned_text))
        else:
            pruned_text = pruned_text[0:max_size]
    return text_vocab(pruned_text)


label_pipeline = lambda x: (0 if (x == 'neg') else 1)

print('text_vocab.get_itos()[29999]: ', text_vocab.get_itos()[29999])
print('text_vocab.get_itos()[30000]: ', text_vocab.get_itos()[30000])
print(text_pipeline('hello, I saw the wanderings waned'))
print(len(text_pipeline('hello, I saw the wanderings waned')))


def collate_batch(batch):
    label_list, text_list = [], []
    for label, review in batch:
        label_list.append(label_pipeline(label))
        text_list.append(text_pipeline(review))
    return (torch.tensor(label_list, dtype=torch.long),
            torch.tensor(text_list, dtype=torch.int32))


train_iter, val_iter = IMDB(split=('train', 'test'))
trainloader = DataLoader(train_iter, batch_size=4,
                         shuffle=False, collate_fn=collate_batch)
valloader = DataLoader(val_iter, batch_size=4,
                       shuffle=False, collate_fn=collate_batch)

embedding = nn.Embedding(num_embeddings=30000, embedding_dim=512,
                         padding_idx=text_vocab.get_stoi()['<pad>'])

for labels, reviews in trainloader:
    print(f'labels.shape: {labels.shape}, reviews.shape: {reviews.shape}')
    break

emb = embedding(torch.randint(high=29999, size=(4, 500)))
print('emb.shape: ', emb.shape)


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.layer_1 = nn.Embedding(num_embeddings=30000, embedding_dim=512,
                                    padding_idx=1)
        self.layer_2 = nn.LSTMCell(input_size=512, hidden_size=512)
        self.layer_3 = nn.Dropout(p=0.5)
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid(),
            nn.BatchNorm1d(2)
        )

    def forward(self, x):
        x = self.layer_1(x)
        print(f'x before permute: {x}')
        x = x.permute(1, 0, 2)
        print(f'x after permute: {x}')
        h = torch.rand(x.shape[1], 512)
        c = torch.rand(x.shape[1], 512)
        for t in range(x.shape[0]):
            h, c = self.layer_2(x[t], (h, c))
            h = self.layer_3(h)
        return self.layer_4(h)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextClassifier().to(device)
N_EPOCHS = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epochs in range(N_EPOCHS):
    running_loss = 0
    for labels, inputs in trainloader:
        if torch.cuda.is_available():
            labels, inputs = labels.cuda(), inputs.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        torch.autograd.backward(loss)
        optimizer.step()
        running_loss += loss
        break
print(f'Epoch: {epochs}, Loss: {loss}')
