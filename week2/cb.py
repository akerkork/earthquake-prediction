# Import necessary libraries
from catboost import CatBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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


# Initialize CatBoostClassifier with specified parameters
cat_clf = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    random_seed=15,
    verbose=False
)

# Fit the model
cat_clf.fit(X_train, y_train)

# Predict on validation set
cat_val_pred = cat_clf.predict(X_val)

# Calculate metrics
cat_accuracy = accuracy_score(y_val, cat_val_pred)
cat_class_report = classification_report(y_val, cat_val_pred, target_names=[str(c) for c in sorted(y.unique())])
cat_cm = confusion_matrix(y_val, cat_val_pred)

# Display results
print("CatBoost Classifier\n")
print(f"Accuracy: {cat_accuracy:.4f}\n")
print("Classification Report:\n", cat_class_report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cat_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('CatBoost Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
