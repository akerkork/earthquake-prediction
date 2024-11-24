# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# One-hot encode the target variable
num_classes = len(class_labels)
y_encoded = to_categorical(y_mapped, num_classes=num_classes)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequential splitting: Use the last 20% of the data as validation
split_index = int(len(data) * 0.8)
X_train = X_scaled[:split_index]
X_val = X_scaled[split_index:]
y_train = y_encoded[:split_index]
y_val = y_encoded[split_index:]

# Define the neural network architecture
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

# Predict on validation set
y_val_pred_prob = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)
y_val_true = np.argmax(y_val, axis=1)

# Classification report
class_report = classification_report(
    y_val_true, y_val_pred, target_names=[str(cls) for cls in class_labels]
)

# Confusion matrix
cm = confusion_matrix(y_val_true, y_val_pred)

# Display results
print("\nNeural Network Classifier\n")
print(f"Validation Accuracy: {val_accuracy:.4f}\n")
print("Classification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Neural Network Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Plot training & validation accuracy and loss
plt.figure(figsize=(12,4))

# Plot accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
