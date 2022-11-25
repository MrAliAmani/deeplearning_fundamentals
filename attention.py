# TODO different tokenization
#  errors, run
#  adding layers


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IWSLT2016
from torchtext.vocab import build_vocab_from_iterator

print(f'Pytorch Version: {torch.__version__}')
print(f'torchtext Version: {torchtext.__version__}')

train_iter = IWSLT2016(root='.data', split=('train'),
                       language_pair=('en', 'fr'))

tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_fr = get_tokenizer('spacy', language='fr_core_news_sm')

sent_len_en, sent_len_fr = [], []
iter_len = 0
for en_sent, fr_sent in train_iter:
    sent_len_en.append(len(tokenizer_en(en_sent)))
    sent_len_fr.append(len(tokenizer_fr(fr_sent)))
    iter_len += 1
print(f'Dataset contains {iter_len} sentences.')

bucket_sizes = [(5, 10), (10, 15), (20, 25), (40, 50)]

print('np.array(sent_len_en).min(), np.array(sent_len_en).max()')
print(np.array(sent_len_en).min(), np.array(sent_len_en).max())
plt.hist(sent_len_en, range=(0, 100))
print('np.array(sent_len_fr).min(), np.array(sent_len_fr).max()')
print(np.array(sent_len_fr).min(), np.array(sent_len_fr).max())
plt.hist(sent_len_fr, range=(0, 100))


def yield_tokens(data_iter, language):
    if language == 'en':
        for data_sample in data_iter:
            yield tokenizer_en(data_sample[0])
    else:
        for data_sample in data_iter:
            yield tokenizer_fr(data_sample[1])


UNK_IDX, PAD_IDX, GO_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<go>', '<eos>']

vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, 'en'), min_freq=1,
                                     specials=special_symbols, special_first=True)

# reload
train_iter = IWSLT2016(root='.data', split=('train'),
                       language_pair=('en', 'fr'))

vocab_fr = build_vocab_from_iterator(yield_tokens(train_iter, 'fr'), min_freq=1,
                                     specials=special_symbols, special_first=True)

print('len(vocab_en): {}, len(vocab_fr): {}'.format(len(vocab_en), len(vocab_fr)))


def process_tokens(source, target, bucket_sizes):
    # account for <go> and <eos> tokens
    for i in range(len(bucket_sizes) + 2):
        if i >= len(bucket_sizes):
            bucket = bucket_sizes[i - 1]
            bucket_id = i - 1
            if len(source) > bucket[0]:
                source = source[:bucket[0]]
            if len(target) > (bucket[1] - 2):
                target = target[:bucket[1] - 2]
            break

        bucket = bucket_sizes[i]
        if (len(source) < bucket[0]) and ((len(target) + 1) < bucket[1]):
            bucket_id = i
            break

    source += ((bucket_sizes[bucket_id][0] - len(source)) * ['<pad>'])
    source = list(reversed(source))

    target.insert(0, '<go>')
    target.append('<eos>')
    target += (bucket_sizes[bucket_id][1] - len(target)) * ['<pad>']

    return vocab_en(source), vocab_fr(target), bucket_id


train_iter = IWSLT2016(split=('train'), language_pair=('en', 'fr'))
for sent_en, sent_fr in train_iter:
    source, target, bucket_id = process_tokens(
        tokenizer_en(sent_en), tokenizer_fr(sent_fr), bucket_sizes
    )

    print('bucket_id, bucket_sizes[bucket_id]'
          .format(bucket_id, bucket_sizes[bucket_id]))
    print('source.shape, len(source), source'
          .format(source.shape, len(source), source))
    print('target.shape, len(target), target'
          .format(target.shape, len(target), target))

    break


def create_bucketed_datasets(data_iter, bucket_sizes, max_data_size=None):
    datasets = []
    for i in range(len(bucket_sizes)):
        datasets.append([])

    iter_len = 0
    for sent_en, sent_fr in train_iter:
        source, target, bucket_id = process_tokens(
            tokenizer_en(sent_en), tokenizer_fr(sent_fr), bucket_sizes
        )
        datasets[bucket_id].append((source, target))
        iter_len += 1
        if max_data_size != None and iter_len > max_data_size:
            break
    print(f'Dataset contains {iter_len} sentences.')
    return dataset


train_iter = IWSLT2016(split='train', languages=('en', 'fr'))
datasets = create_bucketed_datasets(train_iter, bucket_sizes)

for dataset in datasets:
    print('len(dataset), len(dataset[0][0]), len(dataset[0][1])'
          .format(len(dataset), len(dataset[0][0]), len(dataset[0][1])))
print(f'dataset[0][0][1]: {dataset[0][0][1]}')


class BucketedDataset(Dataset):
    def __init__(self, bucketed_dataset, bucket_size):
        super(BucketedDataset, self).__init__()
        self.length = len(bucketed_dataset)
        self.input_len = bucket_size[0]
        self.target_len = bucket_size[1]
        self.bucketed_dataset = bucketed_dataset

    def __getitem__(self, index):
        return (torch.tensor(self.bucketed_dataset[index][0], dtype=torch.float32)), \
               (torch.tensor(self.bucketed_dataset[index][1], dtype=torch.float32))

    def __len__(self):
        return self.length


bucketed_datasets = []
for i, dataset in enumerate(datasets):
    bucketed_datasets.append(BucketedDataset(dataset, bucket_sizes[i]))

for dataset in bucketed_datasets:
    print('len(dataset), dataset[0]'.format(len(dataset), dataset[0]))

dataloaders = []
for dataset in bucketed_datasets:
    dataloaders.append(DataLoader(dataset, batch_size=32, shuffle=True))


class TranslateLSTM(nn.Module):
    def __getitem__(self, item):
        super(TranslateLSTM, self).__getitem__()
        self.layer_1 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        return self.layer_1(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TranslateLSTM().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


def train(source, target):
    optimizer.zero_grad()
    outputs = model(source)
    loss = loss_fn(outputs, target)
    loss.backward()
    optimizer.step()

    return loss, outputs


n_epochs = 5
for epoch in range(n_epochs):
    dataloader_sizes = []
    for dataloader in dataloaders:
        dataloader_sizes.append(len(dataloader))

    while np.array(dataloader_sizes).sum() != 0:
        bucket_id = torch.randint(low=0, high=len(bucket_sizes), size=(1, 1))
        if dataloader_sizes[bucket_id] == 0:
            continue
        source, target = next(iter(dataloaders[bucket_id]))
        dataloader_sizes[bucket_id] -= 1
        # loss = train(encoder_inputs, decoder_inputs, target_weights, bucket_id)
        # loss += step_loss/steps_per_checkpoint
        # current_step += 1
