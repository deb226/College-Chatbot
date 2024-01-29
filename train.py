import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Load intents
with open('dataset1.JSON', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
all_labels = []
all_preds = []
train_accuracy_list = []
val_accuracy_list = []

# Tokenize and stem words
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!','/','\\',',','<','>','@','#','$','-','&','~','`']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16  # Increase hidden layer size
output_size = len(tags)

# Define Dataset class
class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Initialize DataLoader
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Validation set
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_accuracy_list = []
val_accuracy_list = [] 

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)

        # Forward pass, backward pass, and optimization
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate training accuracy after each epoch
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            total_train = 0
            correct_train = 0

            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(device)

                outputs = model(words)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = correct_train / total_train
            train_accuracy_list.append(train_accuracy)

            # Print training accuracy after every 100 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}')

    if (epoch + 1) % 100 == 0:  # Add this condition to evaluate validation accuracy only after every 100 epochs
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracy_list.append(val_accuracy)

# Plot the learning curve
# Plot the learning curve
plt.figure(figsize=(8, 6))
epochs_to_plot = range(1, num_epochs + 1, 100)  # Adjust the interval as needed

plt.plot(epochs_to_plot, train_accuracy_list, label='Training Accuracy')
plt.plot(epochs_to_plot, val_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Loop through the dataset and get predictions
with torch.no_grad():
    for (words, labels) in dataset:
        words = torch.tensor(words).to(device)  # Convert to PyTorch tensor and send to device
        labels = torch.tensor(labels).to(device)

        output = model(words)

        if len(output.shape) == 1:
            output = output.unsqueeze(0)  
            
        _, predicted = torch.max(output, dim=1)

        all_labels.append(labels.item())
        all_preds.append(predicted.item())

# Create a confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Save trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# Save the confusion matrix plot as an image
plt.savefig('confusion_matrix.png')

# Print a message indicating that the confusion matrix plot has been saved
print('Confusion matrix plot saved as confusion_matrix.png')


FILE = "data4.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')
