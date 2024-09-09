# model.py

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_output):
        # Apply attention mechanism
        attn_weights = torch.softmax(self.attention(rnn_output), dim=1)
        context_vector = torch.sum(attn_weights * rnn_output, dim=1)
        return context_vector, attn_weights

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        context_vector, _ = self.attention(lstm_out)
        output = self.fc(context_vector)
        return output

