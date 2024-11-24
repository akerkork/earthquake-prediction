# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv('LAeq_fulltrain.csv')

# Split data into features and target
X = data.drop(columns=['class'])
y = data['class']

# Map class labels to integers starting from 0 (if necessary)
# This is important if your classes do not start at 0
class_labels = y.unique()
class_labels.sort()
label_mapping = {label: idx for idx, label in enumerate(class_labels)}
y_mapped = y.map(label_mapping)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequential splitting: Use the last 20% of the data as validation
split_index = int(len(data) * 0.8)
X_train = X_scaled[:split_index]
X_val = X_scaled[split_index:]
y_train_mapped = y_mapped.iloc[:split_index]
y_val_mapped = y_mapped.iloc[split_index:]
y_val = y.iloc[split_index:]

# Determine the majority class in the training data
unique_classes, counts = np.unique(y_train_mapped, return_counts=True)
majority_class = unique_classes[np.argmax(counts)]
print(f"The majority class is: {majority_class}")

# Create predictions for the validation set (majority class model)
y_val_pred_majority = np.full(shape=y_val_mapped.shape, fill_value=majority_class)

# Calculate metrics for majority class model
accuracy_majority = accuracy_score(y_val_mapped, y_val_pred_majority)
class_report_majority = classification_report(
    y_val_mapped, y_val_pred_majority, target_names=[str(c) for c in unique_classes]
)
cm_majority = confusion_matrix(y_val_mapped, y_val_pred_majority)

# Display results for majority class model
print("\nMajority Class Model\n")
print(f"Accuracy: {accuracy_majority:.4f}\n")
print("Classification Report:\n", class_report_majority)

# Plot confusion matrix for majority class model
plt.figure(figsize=(8,6))
sns.heatmap(cm_majority, annot=True, fmt='d', cmap='Reds',
            xticklabels=unique_classes, yticklabels=unique_classes)
plt.title('Majority Class Model Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Generate random predictions for the validation set (random class model)
np.random.seed(15)  # For reproducibility
random_classes = np.random.choice(unique_classes, size=y_val_mapped.shape)

# Calculate metrics for random class model
accuracy_random = accuracy_score(y_val_mapped, random_classes)
class_report_random = classification_report(
    y_val_mapped, random_classes, target_names=[str(c) for c in unique_classes]
)
cm_random = confusion_matrix(y_val_mapped, random_classes)

# Display results for random class model
print("\nRandom Class Model\n")
print(f"Accuracy: {accuracy_random:.4f}\n")
print("Classification Report:\n", class_report_random)

# Plot confusion matrix for random class model
plt.figure(figsize=(8,6))
sns.heatmap(cm_random, annot=True, fmt='d', cmap='Purples',
            xticklabels=unique_classes, yticklabels=unique_classes)
plt.title('Random Class Model Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
