# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_train.shape[1]
output_dim = len(class_labels)
model = LogisticRegressionModel(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate on validation set
with torch.no_grad():
    y_val_pred = model(X_val)
    y_val_pred_classes = torch.argmax(y_val_pred, dim=1).numpy()

# Calculate metrics
accuracy = accuracy_score(y_val.numpy(), y_val_pred_classes)
class_report = classification_report(
    y_val.numpy(), y_val_pred_classes, target_names=[str(cls) for cls in class_labels]
)
cm = confusion_matrix(y_val.numpy(), y_val_pred_classes)

# Display results
print("\nLogistic Regression Classifier (PyTorch)\n")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", class_report)

# Plot confusion matrix and save it as an image
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Logistic Regression Confusion Matrix (PyTorch)')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig('confusion_matrix.png')  # Save plot to a file
print("Confusion matrix plot saved as 'confusion_matrix.png'. Open it to view the plot.")
