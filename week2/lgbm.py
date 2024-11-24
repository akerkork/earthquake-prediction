import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('LAeq_fulltrain.csv')

# Split data into features and target
X = data.drop(columns=['class'])
y = data['class']

# Determine the index for the 80/20 split
split_index = int(len(data) * 0.8)

# Split the data sequentially
X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

# Initialize LGBMClassifier with specified parameters
clf = LGBMClassifier(
    num_leaves=128,
    max_depth=6,
    min_split_gain=0.5,
    min_child_samples=20,
    random_state=15
)
clf.fit(X_train, y_train)
val_pred = clf.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, val_pred)
class_report = classification_report(y_val, val_pred, target_names=[str(c) for c in sorted(y.unique())])
cm = confusion_matrix(y_val, val_pred)

# Display results
print("LGBM Classifier\n")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('LGBM Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
