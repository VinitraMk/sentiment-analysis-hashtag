import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class RNN(nn.Module):
    embedding_dim = None
    num_layers = None
    dropout = None
    lstm_size = None

    def __init__(self, lstm_size, embedding_dim, num_layers, num_words, num_classes, dropout):
        super(RNN, self).__init__()
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim = embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.lstm_size, num_layers = self.num_layers,
        dropout=dropout, batch_first = True)
        self.fc = nn.Linear(self.lstm_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, state = self.lstm(embedded)
        logits = self.fc(state[-1])
        logits = torch.sigmoid(logits)
        logits = (logits > 0.5).type(torch.int8)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size), torch.zeros(self.num_layers, sequence_length, self.lstm_size))
