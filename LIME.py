# interpret_model.py

import torch
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from model import BiLSTMWithAttention  # Import your model

# Load the model, data, and tokenizer
# ...

class ModelWrapper:
    def __init__(self, model, tokenizer, max_seq_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def predict_proba(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')
        inputs = torch.LongTensor(padded_sequences)
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()

# Wrap the model for LIME
wrapped_model = ModelWrapper(model, tokenizer, max_seq_length)

# Use LIME to interpret
explainer = LimeTextExplainer(class_names=label_encoder.classes_)
explanation = explainer.explain_instance('sample text', wrapped_model.predict_proba)
explanation.show_in_notebook()
