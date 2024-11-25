import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the base model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset
df = pd.read_csv('half_cheetah_descriptions.csv')

# Encode descriptions to sentence embeddings
description_embeddings = torch.tensor(
    base_model.encode(df['description'].tolist(), convert_to_tensor=True)
).to(device)

# Prepare target data
targets = torch.tensor(df[['x0', 'x1']].values,
                       dtype=torch.float32).to(device)

# Create a TensorDataset
dataset = TensorDataset(description_embeddings, targets)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Define the SentenceToVectorModel


class SentenceToVectorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(SentenceToVectorModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, embeddings):
        return self.layers(embeddings)


# Initialize the model and move to GPU
model = SentenceToVectorModel(
    input_dim=base_model.get_sentence_embedding_dimension()
).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training loop with validation loss tracking
train_losses = []
val_losses = []

for epoch in range(500):
    # Training phase
    model.train()
    train_loss = 0
    for embeddings, target in train_loader:
        embeddings, target = embeddings.to(device), target.to(device)
        predictions = model(embeddings)
        loss = criterion(predictions, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for embeddings, target in val_loader:
            embeddings, target = embeddings.to(device), target.to(device)
            predictions = model(embeddings)
            loss = criterion(predictions, target)
            val_loss += loss.item()

    # Average losses
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Plot training and validation losses

plt.plot(range(1, 501), train_losses, label="Train Loss")
plt.plot(range(1, 501), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Evaluate the model on a new description
test_description = ["This elite hops on ONLY one foot."]
test_embeddings = torch.tensor(
    base_model.encode(test_description, convert_to_tensor=True)
).to(device)

# Make predictions
model.eval()
predicted_outputs = model(test_embeddings)
print(f"Predicted x0 and x1: {predicted_outputs}")
