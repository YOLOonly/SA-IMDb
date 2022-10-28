import torch
import torch.nn as nn

from data_preprocess import load_imdb
from torch.utils.data import DataLoader
from utils import set_seed


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(4 * hidden_size, 2)

        self._reset_parameters()

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)  # (seq_len, batch_size, embed_size)
        output, (_, _) = self.rnn(x)  # output shape: (seq_len, batch_size, 2 * hidden_size)
        output = self.fc(torch.cat((output[0], output[-1]), dim=-1))  # output shape: (batch_size, 2)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


set_seed()

BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BiLSTM(vocab_size=len(vocab), embed_size=128, hidden_size=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
