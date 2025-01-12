import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.nn.models import MLP
import torch.optim as optim
import numpy as np

# QM9 property names
qm9_properties = [
    'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 
    'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'
]

# Load QM9 Dataset
dataset = QM9(root='data/QM9')
print(f"Number of samples: {len(dataset)}")

# Select HOMO-LUMO gap as the target property
target_idx = 4
print(f"\nPredicting property: {qm9_properties[target_idx]}")

# Print available target properties
print("\nAvailable target properties for the first sample:")
for i, prop in enumerate(dataset[0].y[0]):
    print(f"Index {i} ({qm9_properties[i]}): {prop.item():.4f}")

# Split Dataset into Train, Validation, Test
dataset = dataset.shuffle()
train_dataset = dataset[:100000]
val_dataset = dataset[100000:110000]
test_dataset = dataset[110000:]

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class GNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_targets):
        super().__init__()
        
        # First NNConv layer
        self.conv1 = NNConv(
            in_channels=node_features,
            out_channels=hidden_dim,
            nn=MLP([edge_features, 32, node_features * hidden_dim]),
            aggr='mean'
        )
        
        # Second NNConv layer
        self.conv2 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=MLP([edge_features, 32, hidden_dim * hidden_dim]),
            aggr='mean'
        )
        
        # Batch normalization layers
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(0.2)
        
        # Final fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_targets)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # First convolution block
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        
        # Second convolution block
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final prediction
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Initialize the Model
model = GNN(
    node_features=dataset.num_node_features, 
    edge_features=dataset.num_edge_features,
    hidden_dim=64,
    num_targets=1  # Single-property regression
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

def train_one_epoch():
    """
    Performs one epoch of training and computes:
      - MSE (via the loss)
      - MAE
      - R²
    """
    model.train()
    total_loss = 0.0
    predictions = []
    labels_list = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        label = data.y[:, target_idx].view(-1, 1)
        
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        
        # Collect predictions and labels for extra metrics
        predictions.append(out.detach().cpu().numpy())
        labels_list.append(label.cpu().numpy())

    # Convert predictions and labels to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # Compute MSE, MAE, R²
    mse = np.mean((predictions - labels_list)**2)  # Should be close to total_loss / N
    mae = np.mean(np.abs(predictions - labels_list))
    
    # Avoid division by zero in case of a very homogeneous target
    ss_res = np.sum((predictions - labels_list)**2)
    ss_tot = np.sum((labels_list - np.mean(labels_list))**2)
    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1 - ss_res / ss_tot

    # Average MSE loss across the entire dataset
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss, mae, r2


def evaluate(loader):
    """
    Evaluates the model and computes:
      - MSE (via the loss)
      - MAE
      - R²
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    labels_list = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            label = data.y[:, target_idx].view(-1, 1)

            loss = criterion(out, label)
            total_loss += loss.item() * data.num_graphs

            predictions.append(out.cpu().numpy())
            labels_list.append(label.cpu().numpy())

    # Convert predictions and labels to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # Compute MSE, MAE, R²
    mse = np.mean((predictions - labels_list)**2)
    mae = np.mean(np.abs(predictions - labels_list))
    
    ss_res = np.sum((predictions - labels_list)**2)
    ss_tot = np.sum((labels_list - np.mean(labels_list))**2)
    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1 - ss_res / ss_tot

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, mae, r2

num_epochs = 30
best_val_loss = float('inf')
best_epoch = 0

print("\nStarting training...")
for epoch in range(1, num_epochs + 1):
    train_loss, train_mae, train_r2 = train_one_epoch()
    val_loss, val_mae, val_r2 = evaluate(val_loader)
    
    # Track best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(
        f"Epoch: {epoch:02d}, "
        f"Train Loss (MSE): {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}, "
        f"Val Loss (MSE): {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}"
    )

print(f"\nBest model was from epoch {best_epoch} with validation MSE {best_val_loss:.4f}")

# Load best model for testing
model.load_state_dict(torch.load('best_model.pth'))

# Final Test Evaluation
test_loss, test_mae, test_r2 = evaluate(test_loader)
print(
    f"\nTest MSE: {test_loss:.4f}, "
    f"Test MAE: {test_mae:.4f}, "
    f"Test R²: {test_r2:.4f}"
)

print("\nPerforming inference on a sample molecule...")
model.eval()
with torch.no_grad():
    try:
        sample = dataset[0].to(device)
        batch = torch.zeros(sample.x.size(0), device=device, dtype=torch.long)
        
        pred = model(sample.x, sample.edge_index, sample.edge_attr, batch)
        actual = sample.y[0, target_idx].item()
        
        print(f"Property: {qm9_properties[target_idx]}")
        print(f"Predicted: {pred.item():.4f}")
        print(f"Actual: {actual:.4f}")
        print(f"Absolute Error: {abs(pred.item() - actual):.4f}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print(f"Sample y shape: {sample.y.shape}")
        print(f"Target index: {target_idx}")
