import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv('half_cheetah_descriptions.csv')

# Initialize a pre-trained transformer model for text embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed text descriptions using the transformer model
X_text = sentence_model.encode(df['description'].tolist())
y_coords = df[['x0', 'x1']].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_coords, test_size=0.2, random_state=42
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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Sentence embedding size
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
X_new = sentence_model.encode(new_prompt)
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_coords = model(X_new_tensor).numpy()

print(f"Predicted coordinates: x0 = {predicted_coords[0][0]}, x1 = {predicted_coords[0][1]}")

