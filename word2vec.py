import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Load data
df = pd.read_csv('half_cheetah_descriptions.csv')

# Preprocess text: tokenize descriptions
tokenized_descriptions = [desc.split() for desc in df['description'].tolist()]

# Train a Word2Vec model on the tokenized descriptions
word2vec = Word2Vec(sentences=tokenized_descriptions, vector_size=50, window=5, min_count=1, workers=4)

# Convert descriptions into averaged Word2Vec embeddings
def get_average_embedding(sentence, model):
    words = sentence.split()
    embeddings = [model.wv[word] for word in words if word in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        # If no word is found in the model, return a zero vector
        return np.zeros(model.vector_size)

X_embeddings = np.array([get_average_embedding(desc, word2vec) for desc in df['description'].tolist()])
y_coords = df[['x0', 'x1']].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y_coords, test_size=0.2, random_state=42
)

# Define a PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a simple neural network for regression
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Word2Vec embedding size
output_size = y_coords.shape[1]  # Predicting x0 and x1
model = SimpleNN(input_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluate the model on the test set
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        y_pred.extend(outputs.numpy())
        y_true.extend(y_batch.numpy())

mse = mean_squared_error(y_true, y_pred)
print(f"Test MSE: {mse:.4f}")

# Predict coordinates for a new prompt
new_prompt = ["This elite hops dominantly on its back leg."]
new_embedding = np.array([get_average_embedding(new_prompt[0], word2vec)])
X_new_tensor = torch.tensor(new_embedding, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_coords = model(X_new_tensor).numpy()

print(f"Predicted coordinates: x0 = {predicted_coords[0][0]}, x1 = {predicted_coords[0][1]}")

