# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('LAeq_fulltrain.csv')

# Split data into features and target
X = data.drop(columns=['class']).values
y = data['class'].values

# Map class labels to integers starting from 0 (if necessary)
class_labels = np.unique(y)
label_mapping = {label: idx for idx, label in enumerate(class_labels)}
y_mapped = np.array([label_mapping[label] for label in y])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequential splitting: Use the last 20% of the data as validation
split_index = int(len(data) * 0.8)
X_train = torch.tensor(X_scaled[:split_index], dtype=torch.float32)
y_train = torch.tensor(y_mapped[:split_index], dtype=torch.long)
X_val = torch.tensor(X_scaled[split_index:], dtype=torch.float32)
y_val = torch.tensor(y_mapped[split_index:], dtype=torch.long)

# Define Single-Layer Perceptron (SLP)
class SingleLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerPerceptron, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Define Multi-Layer Perceptron (MLP)
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = X_train.shape[1]
output_dim = len(class_labels)
hidden_dim = 32  # Hidden layer size for MLP
num_epochs = 500
learning_rate = 0.01

# Initialize SLP and MLP
slp_model = SingleLayerPerceptron(input_dim, output_dim)
mlp_model = MultiLayerPerceptron(input_dim, hidden_dim, output_dim)

# Loss function and optimizers
criterion = nn.CrossEntropyLoss()
slp_optimizer = optim.Adam(slp_model.parameters(), lr=learning_rate)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

# Training SLP
slp_loss_curve = []
for epoch in range(num_epochs):
    outputs = slp_model(X_train)
    loss = criterion(outputs, y_train)
    slp_loss_curve.append(loss.item())

    slp_optimizer.zero_grad()
    loss.backward()
    slp_optimizer.step()

# Training MLP
mlp_loss_curve = []
for epoch in range(num_epochs):
    outputs = mlp_model(X_train)
    loss = criterion(outputs, y_train)
    mlp_loss_curve.append(loss.item())

    mlp_optimizer.zero_grad()
    loss.backward()
    mlp_optimizer.step()

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(slp_loss_curve, label='SLP Loss', color='blue')
plt.plot(mlp_loss_curve, label='MLP Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves: SLP vs MLP')
plt.legend()
plt.grid()
plt.savefig('slp_mlp_loss_comparison.png')
print("Loss curve comparison plot saved as 'slp_mlp_loss_comparison.png'")

# Evaluate SLP and MLP on validation set
with torch.no_grad():
    y_val_pred_slp = slp_model(X_val)
    y_val_pred_classes_slp = torch.argmax(y_val_pred_slp, dim=1).numpy()
    y_val_pred_mlp = mlp_model(X_val)
    y_val_pred_classes_mlp = torch.argmax(y_val_pred_mlp, dim=1).numpy()

# Metrics
accuracy_slp = accuracy_score(y_val.numpy(), y_val_pred_classes_slp)
accuracy_mlp = accuracy_score(y_val.numpy(), y_val_pred_classes_mlp)

print("\nValidation Accuracy:")
print(f"SLP Accuracy: {accuracy_slp:.4f}")
print(f"MLP Accuracy: {accuracy_mlp:.4f}")
