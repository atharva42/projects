# -*- coding: utf-8 -*-
"""skip-gram-classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OU1lBf0H_XeWdVLACjIg5kqDJxh8haa4
"""

# -*- coding: utf-8 -*-
"""skip-gram-classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S4qICfgNn-OT8zVt2LsJjn89Hnrs2ikZ
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Define the RNN LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.lstm.weight_ih_l0.device)

        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Load data
df_train = pd.read_csv("/content/drive/MyDrive/train.csv", nrows=10000)
df_test = pd.read_csv("/content/drive/MyDrive/test.csv")
df_train = df_train
X_train = df_train["Description"].tolist()
X_test = df_test["Description"].tolist()
y_train = df_train["Class Index"].values
y_test = df_test["Class Index"].values
num_epochs = 10
batch_size = 128
input_size = 100
hidden_size = 32
# Load Skip-Gram word vectors from the file
with open('/content/drive/MyDrive/skip-gram-word-vectors.pt', 'rb') as f:
    word_vectors_skip = pickle.load(f)

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer and transform the training data
X_train_transform = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_transform = vectorizer.transform(X_test)

# Get the vocabulary from the vectorizer
vocab = vectorizer.get_feature_names_out()

# Create a dictionary to map words to their indices
word_to_idx = {word: i for i, word in enumerate(vocab)}

# Define a function to convert text sequences to word vectors using Skip-Gram word vectors
def convert_to_word_vectors(text_sequences, word_to_idx, skip_word_vectors, max_length, input_size):
    batch_size = 128
    data_sequences = len(text_sequences)
    count_batches = (data_sequences + batch_size - 1) // batch_size  # Calculate number of batches

    changed_data = np.zeros((data_sequences, max_length, input_size))
    changed_labels = np.zeros((data_sequences,), dtype=np.int64)

    for batch in range(count_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, data_sequences)

        for i in range(start, end):
            sequence = text_sequences[i]
            for j, word in enumerate(sequence.split()):
                if j < max_length:
                    if word in word_to_idx:
                        word_index = word_to_idx[word]
                        if word_index < len(skip_word_vectors):  # Check if the word index is within bounds
                            changed_data[i, j, :] = skip_word_vectors[word_index]
                        else:
                            changed_data[i, j, :] = np.zeros(input_size)  # Handle out-of-bounds index
                    else:
                        # Handle out-of-vocabulary words
                        changed_data[i, j, :] = np.zeros(input_size)  # Replace with zeros or another strategy

            # Assign a special index or label for out-of-vocabulary words
            if not any(changed_data[i].flatten()):
                changed_labels[i] = -1  # You can choose any special index or label

    return changed_data, changed_labels


# Define parameters for the model
max_seq_length = max(len(seq.split()) for seq in X_train)
output_size = len(np.unique(y_train))
# Convert training and testing data to word vectors in batches
X_train_tensor, y_train_tensor = convert_to_word_vectors(X_train, word_to_idx, word_vectors_skip, max_seq_length, input_size)
X_test_tensor, y_test_tensor = convert_to_word_vectors(X_test, word_to_idx, word_vectors_skip, max_seq_length, input_size)

# Filter out samples with out-of-vocabulary labels
X_train_tensor = torch.tensor(X_train_tensor[y_train_tensor != -1], dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_tensor[y_train_tensor != -1], dtype=torch.long)

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Lists to store training accuracy for each epoch
train_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        optimizer.zero_grad()
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Calculate training accuracy after each epoch
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_predicted = torch.max(train_outputs, 1)
        train_accuracy = accuracy_score(y_train_tensor.numpy(), train_predicted.numpy())
        train_accuracies.append(train_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy}")

# Plotting epoch vs accuracy
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Training Accuracy')
plt.legend()
plt.show()



# Evaluation
model.eval()

# Calculate training accuracy
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    _, train_predicted = torch.max(train_outputs, 1)
    train_accuracy = accuracy_score(y_train_tensor.numpy(), train_predicted.numpy())
    train_precision = precision_score(y_train_tensor.numpy(), train_predicted.numpy(), average='weighted')
    train_recall = recall_score(y_train_tensor.numpy(), train_predicted.numpy(), average='weighted')
    train_f1 = f1_score(y_train_tensor.numpy(), train_predicted.numpy(), average='weighted')
    train_confusion = confusion_matrix(y_train_tensor.numpy(), train_predicted.numpy())

    print("Performance Metrics on Train Set:")
    print(f"Accuracy: {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1}")
    print("Confusion Matrix:")
    print(train_confusion)

# Calculate test accuracy

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs, 1)
    test_accuracy = accuracy_score(y_test_tensor, test_predicted)
    test_precision = precision_score(y_test_tensor, test_predicted, average='weighted')
    test_recall = recall_score(y_test_tensor, test_predicted, average='weighted')
    test_f1 = f1_score(y_test_tensor, test_predicted, average='weighted')
    test_confusion = confusion_matrix(y_test_tensor, test_predicted)

    print("\nPerformance Metrics on Test Set:")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")
    print("Confusion Matrix:")
    print(test_confusion)





# Save the trained model
torch.save(model.state_dict(), 'skip-gram-classification-model.pt')

print("Model saved successfully.")