import torch
import torch.nn as nn

from utils import set_seed
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification
from data_preprocess import load_imdb


class BERT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.vocab_size = len(self.vocab)
        self.bert = BertForSequenceClassification(config=self.config)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.vocab['<pad>']).long().float()
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


set_seed(42)

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

train_data, test_data, vocab = load_imdb(bert_preprocess=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
model = nn.DataParallel(BERT(vocab), device_ids=devices).to(devices[0])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)

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
