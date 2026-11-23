
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create required folders
Path("models").mkdir(exist_ok=True)
Path("development").mkdir(exist_ok=True)

# Load preprocessed MNIST dataset
data = np.load("development/mnist_processed.npz")

print("Available dataset keys:", data.files)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First convolution block
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Improved classification layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Display model architecture
print("\n========== CNN MODEL SUMMARY ==========")
model.summary()

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting CNN training...")

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate model on test dataset
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

# Save trained model
model.save("models/mnist_cnn_model.keras")
print("\nCNN model saved successfully!")

# Save training history
np.savez(
    "development/training_history.npz",
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"],
    loss=history.history["loss"],
    val_loss=history.history["val_loss"]
)

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("development/training_accuracy.png", dpi=150)
plt.show()

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("development/training_loss.png", dpi=150)
plt.show()

# Display final results
print("\n========== FINAL TRAINING RESULTS ==========")
print(f"Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

print("\nSTAGE 4 CNN TRAINING COMPLETED!")
