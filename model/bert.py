import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T

from utils import set_seed
from transformers import BertConfig, BertForSequenceClassification
from data_preprocess import read_imdb


def build_dataset(reviews, labels, vocab, max_len=512):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.Truncate(max_seq_len=max_len - 2),
        T.AddToken(token=vocab['<cls>'], begin=True),
        T.AddToken(token=vocab['<sep>'], begin=False),
        T.ToTensor(padding_value=vocab['<pad>']),
        T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']),
    )
    dataset = TensorDataset(text_transform(reviews), torch.tensor(labels))
    return dataset


def load_imdb():
    reviews_train, labels_train = read_imdb(is_train=True)
    reviews_test, labels_test = read_imdb(is_train=False)
    vocab = build_vocab_from_iterator(reviews_train, min_freq=3, specials=['<pad>', '<unk>', '<cls>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])
    train_data = build_dataset(reviews_train, labels_train, vocab)
    test_data = build_dataset(reviews_test, labels_test, vocab)
    return train_data, test_data, vocab


class BERT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.config = BertConfig()
        self.config.vocab_size = len(self.vocab)

        self.bert = BertForSequenceClassification(config=self.config)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.vocab['<pad>']).long().float()
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


set_seed()

BATCH_SIZE = 256
LEARNING_RATE = 0.0001
NUM_EPOCHS = 40

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

devices = [f'cuda:{i}' for i in range(4)]
model = nn.DataParallel(BERT(vocab), device_ids=devices).to(devices[0])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'Epoch {epoch}\n' + '-' * 32)
    avg_train_loss = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(devices[0]), y.to(devices[0])
        pred = model(X)
        loss = criterion(pred, y)
        avg_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss:.4f}")

    print(f"Avg train loss: {avg_train_loss/(batch_idx + 1):.4f}\n")

acc = 0
for X, y in test_loader:
    with torch.no_grad():
        X, y = X.to(devices[0]), y.to(devices[0])
        pred = model(X)
        acc += (pred.argmax(1) == y).sum().item()

print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")
