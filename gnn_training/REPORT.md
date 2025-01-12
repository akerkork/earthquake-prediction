# GNN for Molecular Property Prediction
## Dataset

The **QM9 dataset** is a collection of quantum chemical calculations of small organic molecules. It contains approximately 130,000 molecules made up of hydrogen (H), carbon (C), nitrogen (N), oxygen (O), and fluorine (F) atoms. For each molecule, the dataset provides:

- **Molecular geometries**: 3D positions of atoms.
- **Chemical properties**: 19 quantum mechanical properties computed using Density Functional Theory (DFT).
- **Connectivity information**: Atomic connections representing molecular bonds.

The key properties include:

- **Dipole moment (μ)**
- **Isotropic polarizability (α)**
- **Highest Occupied Molecular Orbital energy (HOMO)**
- **Lowest Unoccupied Molecular Orbital energy (LUMO)**
- **HOMO-LUMO energy gap (gap)**
- **Electronic spatial extent (r²)**
- **Zero-point vibrational energy (ZPVE)**
- **Internal energies (U₀, U, H)**
- **Free energy (G)**
- **Heat capacity (Cv)**
- **Rotational constants (A, B, C)**
  
The noode features are the atom attributes, and the edge features are the bond attributes between the atoms.

- Edge features do the following:
  - They represent bond properties such as bond type, order, and length.
  - They are important for capturing the chemical interactions between atoms.

# Purpose

The primary goal of the code is to develop and train a GNN to predict a specific molecular property from the QM9 dataset. Specifically, the code focuses on predicting the **HOMO-LUMO energy gap**.

# Methodology

### Model Architecture
1. **NNConv Layers**: 
   - Two NNConv layers are used, each employing a MLP to transform edge features into messages.
   - The first NNConv layer maps the initial node features to a hidden dimension.
   - The second NNConv layer operates on this hidden representation and outputs another embedding for each node.
   - Both layers use mean aggregation to gather messages from neighboring nodes.

2. **MLPs for Edge Transformations**:  
   - Each NNConv layer is coupled with an MLP that takes edge features as input and outputs weight matrices for the message passing.

3. **Batch Normalization and Dropout**:  
   - **BatchNorm1d** layers are applied after each NNConv to normalize the hidden representations, helping stabilize training.
   - A **Dropout** layer with a probability of 0.2 is included after each convolution block to reduce overfitting.

4. **Pooling and Final MLP**:  
   - A **global_mean_pool** operation aggregates all node embeddings into a single graph-level vector.
   - A final MLP with one hidden layer is then applied to produce the prediction.
---

**Loss Function**:  
   - **Mean Squared Error (MSE)** is used as the primary loss function during training

# Results

## Best Validation Performance
- **Best Validation MSE**: 0.1277 (achieved at epoch 30).
- **Corresponding Validation Metrics**:
  - MAE: 0.2631
  - R²: 0.9211

## Test Performance
- After loading the best model, the test set performance metrics were:
  - **MSE**: 0.1241
  - **MAE**: 0.2615
  - **R²**: 0.9247
- These results indicate that the model explains over **92% of the variance** in the “gap” property and achieves relatively low errors.

## Inference Example
- **Predicted Value**: 8.1010
- **Actual Value**: 8.1199
- **Absolute Error**: 0.0189
