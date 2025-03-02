import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset


torch.manual_seed(42)
np.random.seed(42)

dataset = load_dataset("dnagpt/dna_core_promoter")
data = dataset['train']
sequences = [item['sequence'] for item in data]
labels = [item['label'] for item in data]


char2idx = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
max_len = max(len(seq) for seq in sequences)


class DNADataset(Dataset):
    def __init__(self, sequences, labels, char2idx, max_len):
        self.sequences = sequences
        self.labels = labels
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_encoded = [self.char2idx.get(nuc, 0) for nuc in seq]
        seq_encoded += [0] * (self.max_len - len(seq_encoded))
        label = float(self.labels[idx])
        return torch.tensor(seq_encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float)


dataset_obj = DNADataset(sequences, labels, char2idx, max_len)

n_total = len(dataset_obj)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
train_dataset, test_dataset = random_split(dataset_obj, [n_train, n_test])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class DNAClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout):
        super(DNAClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            forward_hidden = h_n[-2, :, :]
            backward_hidden = h_n[-1, :, :]
            hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            hidden = h_n[-1, :, :]
        out = self.fc(hidden)
        out = torch.sigmoid(out)
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 5  # 0+ 4个核苷酸
embed_dim = 64
hidden_dim = 128
num_layers = 2
bidirectional = True
dropout = 0.3

model = DNAClassifierLSTM(vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f"测试集准确率: {accuracy:.4f}")
