import torch
import torch.nn as nn

from utils import set_seed
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, get_scheduler
from data_preprocess import read_imdb
from collections import OrderedDict
from torchtext.vocab import vocab
from torchtext import transforms as T


def load_imdb():
    with open('./vocab.txt', 'r') as f:
        freq = []
        for token in f.readlines():
            freq.append((token.strip(), 1))
    v = vocab(OrderedDict(freq))
    v.set_default_index(v['[UNK]'])

    text_transform = T.Sequential(
        T.VocabTransform(vocab=v),
        T.Truncate(max_seq_len=510),
        T.AddToken(token=v['[CLS]'], begin=True),
        T.AddToken(token=v['[SEP]'], begin=False),
        T.ToTensor(padding_value=v['[PAD]']),
        T.PadTransform(max_length=512, pad_value=v['[PAD]']),
    )

    reviews_train, labels_train = read_imdb(is_train=True)
    reviews_test, labels_test = read_imdb(is_train=False)

    train_data = TensorDataset(text_transform(reviews_train), torch.tensor(labels_train))
    test_data = TensorDataset(text_transform(reviews_test), torch.tensor(labels_test))

    return train_data, test_data, v


class BERT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.vocab['[PAD]']).long().float()
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


set_seed(42)

BATCH_SIZE = 256
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
model = nn.DataParallel(BERT(vocab), device_ids=devices).to(devices[0])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS * len(train_loader))

model.train()
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
        scheduler.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss:.4f}")

    print(f"Avg train loss: {avg_train_loss/(batch_idx + 1):.4f}\n")

model.eval()
acc = 0
for X, y in test_loader:
    with torch.no_grad():
        X, y = X.to(devices[0]), y.to(devices[0])
        pred = model(X)
        acc += (pred.argmax(1) == y).sum().item()

print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")
