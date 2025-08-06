
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

## 1. Load Data
# This part is the same as your original notebook.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate features (pixels) and labels
X = train_df.drop('label', axis=1)
y = train_df['label']

## 2. Prepare Data

# >>> CHANGE: Added explicit data scaling.
# WHY: Neural networks train much more effectively when input data is scaled to a small
# range, like 0-1. You did this in your Keras notebook but it was missing here.
X = X / 255.0
test_df = test_df / 255.0

# Split data into training and validation sets. Your original code did this well.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert all data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
test_tensor = torch.tensor(test_df.values, dtype=torch.float32)

# >>> CHANGE: Use DataLoader for mini-batching.
# WHY: Processing the entire dataset at once is inefficient and memory-intensive.
# DataLoader automatically creates small, shuffled batches, which leads to faster
# and often more stable training. This is the standard practice.
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


## 3. Define the Neural Network Model
# Your original model definition was excellent. No changes needed.
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 128) # Increased layer size for better capacity
        self.fc2 = nn.Linear(128, 64)  # Increased layer size
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x) # No activation here, as CrossEntropyLoss expects raw logits
        return x

# Instantiate model, loss, and optimizer
model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


## 4. Train the Model with an Integrated Validation Loop

# >>> CHANGE: The entire training loop is restructured.
# WHY: This new loop processes data in mini-batches and evaluates on the
# validation set after each full pass (epoch). This allows us to monitor
# for overfitting in real-time and save the results for plotting.

# Set the number of epochs to a more reasonable value
num_epochs = 50

# Lists to store performance history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy for the batch
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # No need to calculate gradients for validation
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


## 5. Visualize Training Results

# >>> CHANGE: Added plotting.
# WHY: A graph makes it easy to see the model's learning progress and to
# spot overfitting (when the validation loss increases as training loss decreases).
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()


## 6. Evaluate Final Model on Validation Set

# This section uses your original evaluation logic, which was already very good.
print("\n--- Final Model Evaluation on Validation Set ---")
model.eval()
with torch.no_grad():
    predictions = model(X_val_tensor)
    _, final_preds = torch.max(predictions, 1)

print("\nClassification Report:")
print(classification_report(y_val, final_preds.numpy()))

# >>> CHANGE: Added a confusion matrix visualization.
# WHY: This gives a more detailed view of what specific digits the model is
# confusing with each other.
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, final_preds.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()


## 7. Generate Predictions for Submission
# Your original code for this was perfect.
print("\n--- Generating Submission File ---")
model.eval()
with torch.no_grad():
    test_predictions = model(test_tensor)
    _, test_preds_final = torch.max(test_predictions, 1)

submission_df = pd.DataFrame({
    'ImageId': np.arange(1, len(test_preds_final) + 1),
    'Label': test_preds_final.numpy()
})
submission_df.to_csv('submission_pytorch_revised.csv', index=False)

print("Submission file 'submission_pytorch_revised.csv' created successfully.")
