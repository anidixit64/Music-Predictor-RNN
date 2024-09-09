# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from model import BiLSTMWithAttention  # Import the new model

# Load and preprocess data
with open('trainfile-2.json', 'r') as train_file:
    train_data = json.load(train_file)
with open('testfile-2.json', 'r') as test_file:
    test_data = json.load(test_file)

all_training_lyrics = [i[0].lower() for i in train_data]
all_training_labels = [i[1].lower() for i in train_data]
all_testing_lyrics = [i[0].lower() for i in test_data]
all_testing_labels = [i[1].lower() for i in test_data]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_training_lyrics)
vocab_size = len(tokenizer.word_index) + 1

training_sequences = tokenizer.texts_to_sequences(all_training_lyrics)
testing_sequences = tokenizer.texts_to_sequences(all_testing_lyrics)

max_seq_length = max(len(seq) for seq in training_sequences + testing_sequences)
X_train = pad_sequences(training_sequences, maxlen=max_seq_length, padding='post')
X_test = pad_sequences(testing_sequences, maxlen=max_seq_length, padding='post')

label_encoder = LabelEncoder()
label_encoder.fit(all_training_labels)
num_artists = len(label_encoder.classes_)

Y_train = label_encoder.transform(all_training_labels)
Y_test = label_encoder.transform(all_testing_labels)

X_train = torch.LongTensor(X_train)
Y_train = torch.LongTensor(Y_train)
X_test = torch.LongTensor(X_test)
Y_test = torch.LongTensor(Y_test)

# Initialize model, criterion, and optimizer
hidden_size = 256
output_size = num_artists
model = BiLSTMWithAttention(vocab_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluate on the test set
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == Y_test).sum().item()
    total = Y_test.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

