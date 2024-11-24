# Import necessary libraries
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('LAeq_fulltrain.csv')

# Split data into features and target
X = data.drop(columns=['class'])
y = data['class']

# Remap the class labels to start from 0
y_mapped = y - y.min()  # Shift class labels to start at 0

# Determine the index for the 80/20 split
split_index = int(len(data) * 0.8)

# Split the data sequentially
X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]
y_train = y_mapped.iloc[:split_index]
y_val = y_mapped.iloc[split_index:]

# Initialize XGBClassifier with specified parameters
xgb_clf = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=15,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Fit the model
xgb_clf.fit(X_train, y_train)

# Predict on validation set
xgb_val_pred = xgb_clf.predict(X_val)

# Calculate metrics
xgb_accuracy = accuracy_score(y_val, xgb_val_pred)
xgb_class_report = classification_report(
    y_val, 
    xgb_val_pred, 
    target_names=[str(c) for c in sorted(y_mapped.unique())]
)
xgb_cm = confusion_matrix(y_val, xgb_val_pred)

# Display results
print("XGBoost Classifier\n")
print(f"Accuracy: {xgb_accuracy:.4f}\n")
print("Classification Report:\n", xgb_class_report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=sorted(y_mapped.unique()), 
            yticklabels=sorted(y_mapped.unique()))
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
