# hyperparameter_tuning.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from model import BiLSTMWithAttention  # Import your model

# Load and preprocess data (similar to train.py)
# ...

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    model = BiLSTMWithAttention(vocab_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    num_epochs = 5  # Use fewer epochs for faster tuning
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on the validation set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == Y_test).sum().item()
        total = Y_test.size(0)
        accuracy = correct / total

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print('Best hyperparameters:', study.best_params)
