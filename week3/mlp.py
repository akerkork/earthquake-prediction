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

# Define Multi-Layer Perceptron model
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Activation for hidden layer
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = X_train.shape[1]
output_dim = len(class_labels)
num_epochs = 500
learning_rate = 0.01
hidden_sizes = [16, 32, 64, 128, 256]  # Different hidden layer sizes to experiment with

# Experiment with different hidden layer sizes
results = []
loss_curves = {}

for hidden_dim in hidden_sizes:
    print(f"\nTraining MLP with hidden layer size: {hidden_dim}")
    model = MultiLayerPerceptron(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_loss = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        train_loss.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Store loss curve
    loss_curves[hidden_dim] = train_loss

    # Evaluate on validation set
    with torch.no_grad():
        y_val_pred = model(X_val)
        y_val_pred_classes = torch.argmax(y_val_pred, dim=1).numpy()
    acc = accuracy_score(y_val.numpy(), y_val_pred_classes)

    # Store results
    results.append((hidden_dim, acc))
    print(f"Hidden Layer Size: {hidden_dim}, Accuracy: {acc:.4f}")

# Print summary results
print("\nSummary of Results:")
for hidden_dim, acc in results:
    print(f"Hidden Layer Size: {hidden_dim}, Accuracy: {acc:.4f}")

# Plot loss curves for each hidden layer size
plt.figure(figsize=(10, 6))
for hidden_dim, loss_curve in loss_curves.items():
    plt.plot(loss_curve, label=f'Hidden Size: {hidden_dim}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curves for Different Hidden Layer Sizes')
plt.legend()
plt.grid()
plt.savefig('hidden_layer_size_loss_curves.png')
print("Loss curve plot saved as 'hidden_layer_size_loss_curves.png'")
