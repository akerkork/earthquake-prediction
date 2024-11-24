# Earthquake Prediction - Week 3 - Abdülkerim Korkmaz

For this week's task, I have trained and compared the performance of different models, including Logistic Regression, Single Layer Perceptron (SLP), and Multi Layer Perceptron (MLP) on earthquake data. The models were evaluated using accuracy and classification metrics to assess their predictive capabilities.

---

## **Models and Hyperparameters**

### **1. Logistic Regression (scikit-learn)**
- **Hyperparameters:**
  - `multi_class='multinomial'`
  - `solver='lbfgs'`
  - `max_iter=2000`
  - `random_state=15`

### **2. Logistic Regression (PyTorch)**
- **Hyperparameters:**
  - **Architecture:**
    - Input Layer: Number of neurons equal to the number of features
    - Output Layer: Number of neurons equal to the number of classes with softmax activation
  - **Training Parameters:**
    - `num_epochs=2000`
    - Optimizer: Adam (`lr=0.01`)
    - Loss Function: CrossEntropyLoss

### **3. Multi-Layer Perceptron (MLP)**
- **Hyperparameters:**
  - **Architecture:**
    - Input Layer: Number of neurons equal to the number of features
    - Hidden Layers: Varying sizes (16, 32, 64, 128, 256 neurons)
    - Output Layer: Number of neurons equal to the number of classes with softmax activation
  - **Training Parameters:**
    - `epochs=50`
    - `batch_size=32`
    - Optimizer: Adam
    - Loss Function: Categorical Crossentropy

---

## **Results and Analysis**

### **1. Logistic Regression (scikit-learn)**
- **Accuracy:** 21.74%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.41          | 0.07       | 0.12         | 975         |
| 2         | 0.22          | 0.72       | 0.34         | 784         |
| 3         | 0.23          | 0.17       | 0.19         | 830         |
| 4         | 0.08          | 0.08       | 0.08         | 309         |
| 5         | 0.00          | 0.00       | 0.00         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---
![logreg](https://github.com/user-attachments/assets/b8d701e6-1c09-48e9-bb43-846c7cafaac2)

### **2. Logistic Regression (PyTorch)**
- **Accuracy:** 21.71%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.42          | 0.07       | 0.13         | 975         |
| 2         | 0.22          | 0.72       | 0.34         | 784         |
| 3         | 0.23          | 0.17       | 0.19         | 830         |
| 4         | 0.08          | 0.07       | 0.08         | 309         |
| 5         | 0.00          | 0.00       | 0.00         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |
![confusion_matrix_logregpytorch](https://github.com/user-attachments/assets/c8dfdc0a-2ae9-4575-8052-083dc6fcac9b)
---

### **3. Multi-Layer Perceptron (MLP) with Different Hidden Layer Sizes**
| **Hidden Layer Size** | **Accuracy** |
|-----------------------|--------------|
| 16                    | 25.60%       |
| 32                    | 26.06%       |
| 64                    | 25.60%       |
| 128                   | 23.10%       |
| 256                   | 24.24%       |

![hidden_layer_size_loss_curves](https://github.com/user-attachments/assets/5ba91929-7781-4913-8826-6f9c4faa2a82)

---

## **SLP vs MLP Comparison**

- **SLP Accuracy:** 21.85%
- **MLP Accuracy:** 26.06%

### **Comparison of SLP and MLP**
MLP with a hidden layer size of 32 achieved the highest accuracy of 26.06%, while SLP performed slightly worse with an accuracy of 21.85%. This suggests that MLP with larger hidden layers may be able to better capture the complex patterns in the data, although the improvement was modest for larger hidden layer sizes.

![slp_mlp_loss_comparison_layersize32](https://github.com/user-attachments/assets/aecfa9f4-0f7f-49a1-ac43-f36a543a4633)

---

## **Conclusions**
- Logistic Regression (scikit-learn) performed at 21.74% accuracy, showing limited capability in predicting the earthquake data with the given features.
- Logistic Regression (PyTorch) performed at 21.71% accuracy, with slightly different performance due to the differences in implementation.
- MLP with different hidden layer sizes achieved the highest accuracy of 26.06%, but no substantial improvement was observed when increasing the hidden layer sizes beyond 32 neurons.
- Despite experimenting with different architectures, the models performed poorly on certain classes.

