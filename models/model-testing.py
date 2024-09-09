import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import BiLSTMWithAttention
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open('testfile-2.json', 'r') as test_file:
    test_data = json.load(test_file)

all_testing_lyrics = [i[0].lower() for i in test_data]
all_testing_labels = [i[1].lower() for i in test_data]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_testing_lyrics)
vocab_size = len(tokenizer.word_index) + 1

testing_sequences = tokenizer.texts_to_sequences(all_testing_lyrics)
max_seq_length = max(len(seq) for seq in testing_sequences)
X_test = pad_sequences(testing_sequences, maxlen=max_seq_length, padding='post')

label_encoder = LabelEncoder()
label_encoder.fit(all_testing_labels)
Y_test = label_encoder.transform(all_testing_labels)

X_test = torch.LongTensor(X_test)
Y_test = torch.LongTensor(Y_test)


hidden_size = 256
output_size = len(label_encoder.classes_)
model = BiLSTMWithAttention(vocab_size, hidden_size, output_size)
model.load_state_dict(torch.load('bilstm_with_attention.pth'))
model.eval()

# Predict on test data
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# Generate confusion matrix
conf_matrix = confusion_matrix(Y_test, predicted)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


report = classification_report(Y_test, predicted, target_names=label_encoder.classes_)
print(report)

