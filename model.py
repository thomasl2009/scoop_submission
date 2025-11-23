import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/training_data.csv", header=None)

print(df.shape)   # rows x columns
print(df.head())

MAX_LEN = 640

# Convert each row to numpy array and pad
data = []

for row in df.values:
    # Convert all items to numeric (invalid → NaN)
    row = pd.to_numeric(row, errors='coerce')

    # Drop NaNs
    row = row[~np.isnan(row)]

    # Convert to float32
    row = row.astype(np.float32)

    # Pad
    if len(row) < MAX_LEN:
        padded = np.pad(row, (0, MAX_LEN - len(row)), constant_values=0)
    else:
        padded = row[:MAX_LEN]

    data.append(padded)

data = np.array(data, dtype=np.float32)

# Remove any leftover inf/NaN
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

print(np.isnan(data).any())   # should be False
print(np.isinf(data).any())   # should be False

print(data.shape)  # (num_samples, 640)

scaler = StandardScaler()

# Example: last column is label, rest are features
# First column is the label
y = data[:, 0].astype(int)   # labels (0 or 1)
X = data[:, 1:]              # features (639 columns after padding)

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Wrap in DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self, input_dim=639, hidden_dim=128, output_dim=2):  # <-- 2 classes
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleModel(input_dim=639, hidden_dim=128, output_dim=2)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):  # 10 epochs
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

MAX_LEN = 640       # same as before
FEATURES = 639      # model input size

# Load your 1-row CSV (no label column)
row = pd.read_csv("./data/test_one.csv", header=None).values.flatten()

# Convert to numeric
row = pd.to_numeric(row, errors='coerce')

# Remove NaNs
row = row[~np.isnan(row)]

# Convert to float32
row = row.astype(np.float32)
row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

# Pad to 639 features + 1 label position (but label is missing)
# Training format = [label | 639 features]
# Here we only have the 639 features, so if row is shorter → pad to 639
if len(row) < FEATURES:
    row = np.pad(row, (0, FEATURES - len(row)), constant_values=0)

row = scaler.transform([row])  # keep 2D shape
tensor_row = torch.tensor(row, dtype=torch.float32)
model.eval()
with torch.no_grad():
    output = model(tensor_row)           # shape: [1, 2] for 2 classes
    probs = F.softmax(output, dim=1)

percent_class_0 = probs[0][0].item() * 100
percent_class_1 = probs[0][1].item() * 100

print(f"Class 0 chance: {percent_class_0:.2f}%")
print(f"Class 1 chance: {percent_class_1:.2f}%")