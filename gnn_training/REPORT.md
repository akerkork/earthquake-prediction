# GNN for Molecular Property Prediction using the QM9 Dataset

## 1. Introduction to the Dataset

The **QM9 dataset** is a comprehensive collection of quantum chemical calculations of small organic molecules. It contains approximately 130,000 molecules made up of hydrogen (H), carbon (C), nitrogen (N), oxygen (O), and fluorine (F) atoms. For each molecule, the dataset provides:

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

## 2. Purpose of the Code

The primary goal of the code is to develop and train a GNN to predict a specific molecular property from the QM9 dataset. Specifically, the code focuses on predicting the **HOMO-LUMO energy gap**.

## 3. Explanation of the Algorithm and the Code

### 3.1 Data Loading and Preparation

- The dataset includes molecular graphs with node features (atom attributes), edge features (bond attributes), and target properties.

- The target property for prediction is the HOMO-LUMO energy gap, corresponding to index 4 in the list of properties.

- Dataset is split into training, validation, and test sets:

  - **Training set**: 100,000 samples for model training.
  - **Validation set**: 10,000 samples for hyperparameter tuning and early stopping.
  - **Test set**: Remaining samples (~20,000) for final evaluation.

**Data Loaders**:

- Data loaders are created for each split to facilitate batch processing.
- A batch size of 64 is used.

### 3.2 Model Definition

The model is a custom GNN class that extends PyTorch's `torch.nn.Module`. The architecture includes:

**NNConv Layers**:

- **First NNConv Layer**:

  - Input: Node features.
  - Output: Hidden node representations.
  - Uses an MLP (Multi-Layer Perceptron) to generate edge-specific convolutional filters.
  - The MLP takes edge features as input and outputs weights to modulate the message passing.

- **Second NNConv Layer**:

  - Takes the output of the first layer as input.
  - Further refines the hidden node representations using another set of edge-conditioned filters.

**Batch Normalization**:

- Applied after each NNConv layer to normalize the node feature distributions.

**Dropout Layers**:

- Dropout with a rate of 0.2 is applied after each activation to prevent overfitting.
- Randomly zeros some of the elements of the input tensor during training.

**Fully Connected Layers**:

- After message passing and pooling, the global graph representation is passed through fully connected layers.
- Includes a ReLU activation and another dropout layer.
- Outputs the final prediction for the target property.

### 3.3 Forward Pass

The forward pass of the model involves the following steps:

1. **First Convolution Block**:

   - Node features are updated using the first NNConv layer with edge-conditioned filters.
   - ReLU activation is applied.
   - Batch normalization helps in stabilizing the learning process.
   - Dropout is used to prevent overfitting.

2. **Second Convolution Block**:

   - The updated node features are processed through the second NNConv layer.
   - Similar activation, normalization, and dropout steps are applied.

3. **Global Pooling**:

   - The node features are aggregated into a single graph-level representation.
   - Global mean pooling computes the average of node features for each graph in the batch.

4. **Final Prediction**:

   - The global graph representations are passed through the fully connected layers.
   - The final output is the predicted value for the HOMO-LUMO gap.

### 3.4 Training and Evaluation

**Device Configuration**:

- The model is trained on a GPU if available; otherwise, it defaults to CPU.

**Model Initialization**:

- The model is instantiated with the appropriate input dimensions for node and edge features.
- A hidden dimension of 64 is used for node representations.
- The output dimension is set to 1 since it's a single-value regression task.

**Optimizer and Loss Function**:

- The **Adam optimizer** is used with a learning rate of 0.001.
- Mean Squared Error (MSE) loss is used as the criterion for training, suitable for regression tasks.

**Training Function**:

- The model is set to training mode.
- Iterates over batches from the training data loader.
- For each batch:

  - Moves data to the computation device.
  - Performs a forward pass to get predictions.
  - Computes the loss between predictions and actual target values.
  - Backpropagates the loss and updates model parameters.

**Evaluation Function**:

- The model is set to evaluation mode, disabling dropout and batch normalization updates.
- Iterates over batches from the validation or test data loader.
- Computes predictions and accumulates the loss for reporting.

### 3.5 Training Loop

- The training runs for a specified number of epochs (e.g., 30).
- After each epoch, the model is evaluated on the validation set.
- The best model is saved based on the lowest validation loss.
- Training progress is printed, showing the epoch number, training loss, and validation loss.

### 3.6 Inference

After training, inference is performed to predict the property of a sample molecule:

- A sample molecule from the dataset is selected.
- The model predicts the HOMO-LUMO gap for this molecule.
- The predicted value is compared to the actual value from the dataset.
- The absolute error is computed to assess the prediction accuracy.

## 4. Explanation of Graph Convolution (Message Passing), Inference, and the Role of Edge Features

### 4.3 Role of Edge Features

**Edge Features in Molecular Graphs**:

- Represent bond properties such as bond type, order, and length.
- Critical for capturing the chemical interactions between atoms.

**Influence in the Model**:

- **Dynamic Filters**: Edge features are input to the MLPs in NNConv layers to produce edge-conditioned filters.
- These filters modulate how messages are passed from one node to another.
- **Adaptive Message Passing**: The model can adjust the importance of messages based on bond characteristics.

**Benefits**:

- Captures more detailed chemical information.
- Improves the model's ability to learn complex relationships affecting molecular properties.
- Enhances prediction accuracy by considering both atomic and bonding information.

## 5. Results

### 5.1 Training Progress

The model was trained for 30 epochs, and the training and validation losses were recorded after each epoch. The training started with a higher loss and gradually decreased over the epochs, indicating that the model was learning effectively.

Key observations from the training process:

- **Epoch 1**:

  - **Training Loss**: 2.2164
  - **Validation Loss**: 0.3546

- **Epoch 15**:

  - **Training Loss**: 0.2515
  - **Validation Loss**: 0.1577

- **Epoch 28**:

  - **Training Loss**: 0.1831
  - **Validation Loss**: **0.1213** (Lowest)

- **Epoch 30**:

  - **Training Loss**: 0.1799
  - **Validation Loss**: 0.1271

The best model was obtained at **epoch 28** with a validation loss of **0.1213**. This model was saved for evaluation on the test set.

### 5.2 Test Set Evaluation

The saved best model was evaluated on the test set to assess its generalization performance. The test loss was calculated as Mean Squared Error (MSE) between the predicted and actual HOMO-LUMO gap values.

- **Test Loss**: **0.1222**

The test loss is close to the validation loss at the best epoch, indicating that the model generalizes well to unseen data and does not overfit the training set.

### 5.3 Inference Example

An inference was performed on a sample molecule from the dataset to illustrate the model's prediction capability.

- **Selected Property**: HOMO-LUMO energy gap (`gap`)
- **Predicted Value**: **4.8822**
- **Actual Value**: **4.5144**
- **Absolute Error**: **0.3678**

**Interpretation**:

- The model's prediction is close to the actual value, with a small absolute error.
- This demonstrates the model's ability to accurately predict the HOMO-LUMO gap for individual molecules.
- The small discrepancy could be due to inherent limitations in the model or the complexity of the molecular structure.

