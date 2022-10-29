import torch
import torch.nn as nn

from data_preprocess import load_imdb
from torch.utils.data import DataLoader
from utils import set_seed
from torchtext.vocab import GloVe


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size=100, kernel_sizes=[2, 2, 3, 3, 4, 4], num_channels=[100] * 6):
        super().__init__()
        self.glove = GloVe(name="6B", dim=100)
        self.embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab['<pad>'])
        self.constant_embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                               padding_idx=vocab['<pad>'],
                                                               freeze=True)

        self.convs = nn.ModuleList()
        for out_channels, kernel_size in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2 * embed_size, out_channels=out_channels, kernel_size=kernel_size))

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_channels), 2)

        self.apply(self._init_weights)

    def forward(self, x):
        x = torch.cat((self.embedding(x), self.constant_embedding(x)), dim=-1)  # (batch_size, seq_len, 2 * embed_size)
        x = x.transpose(1, 2)  # (batch_size, 2 * embed_size, seq_len)
        x = torch.cat([self.pool(self.relu(conv(x))).squeeze() for conv in self.convs], dim=-1)  # (batch_size, sum(num_channels))
        x = self.fc(self.dropout(x))
        return x

    def _init_weights(self, m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)


set_seed()

BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 35

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TextCNN(vocab).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'Epoch {epoch}\n' + '-' * 32)
    avg_train_loss = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        avg_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 5 == 0:
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss:.4f}")

    print(f"Avg train loss: {avg_train_loss/(batch_idx + 1):.4f}\n")

acc = 0
for X, y in test_loader:
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        pred = model(X)
        acc += (pred.argmax(1) == y).sum().item()

print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")
