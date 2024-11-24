# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv('LAeq_fulltrain.csv')

# Split data into features and target
X = data.drop(columns=['class'])
y = data['class']

# Map class labels to integers starting from 0 (if necessary)
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
y_train = y_mapped.iloc[:split_index]
y_val = y_mapped.iloc[split_index:]

# Initialize Logistic Regression model with default parameters
logreg = LogisticRegression(
    multi_class='multinomial',  # For multiclass classification
    solver='lbfgs',             # Recommended solver for multinomial problems
    max_iter=2000,              # Increase max iterations if needed
    random_state=15
)

# Fit the model
logreg.fit(X_train, y_train)

# Get model parameters (coefficients)
coefficients = logreg.coef_
intercept = logreg.intercept_

# Display model parameters
print("Logistic Regression Model Parameters:")
for idx, (coefs, cls) in enumerate(zip(coefficients, class_labels)):
    print(f"\nClass {cls} coefficients:")
    for feature_name, coef in zip(X.columns, coefs):
        print(f"{feature_name}: {coef:.4f}")
    print(f"Intercept: {intercept[idx]:.4f}")

# Predict on validation set
y_val_pred = logreg.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, y_val_pred)
class_report = classification_report(
    y_val, y_val_pred, target_names=[str(cls) for cls in class_labels]
)
cm = confusion_matrix(y_val, y_val_pred)

# Display results
print("\nLogistic Regression Classifier\n")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
