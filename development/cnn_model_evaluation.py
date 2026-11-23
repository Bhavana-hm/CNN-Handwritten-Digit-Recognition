
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os

os.makedirs("development", exist_ok=True)

print("\nLoading trained CNN model...")

# Load trained model
model = tf.keras.models.load_model(
    "models/mnist_cnn_model.keras"
)

# Load test dataset
data = np.load("development/mnist_processed.npz")
x_test = data["x_test"]
y_test = data["y_test"]

print("Model and test dataset loaded successfully!")

# Evaluate model
print("\n========== TEST PERFORMANCE ==========")

test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=1
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
print("\nGenerating predictions...")

probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

# Classification report
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("MNIST CNN Confusion Matrix")
plt.savefig(
    "development/confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

# Count correct and incorrect predictions
correct = np.sum(y_test == y_pred)
incorrect = np.sum(y_test != y_pred)

print("\n========== PREDICTION RESULTS ==========")
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)

# Display sample actual vs predicted results
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

    actual = y_test[i]
    predicted = y_pred[i]

    result = "Correct" if actual == predicted else "Incorrect"

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}\n{result}"
    )
    plt.axis("off")

plt.suptitle("Actual vs Predicted MNIST Digits")
plt.tight_layout()

plt.savefig(
    "development/actual_vs_predicted.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

# Save evaluation metrics
np.savez(
    "development/evaluation_results.npz",
    test_accuracy=test_accuracy,
    test_loss=test_loss,
    correct=correct,
    incorrect=incorrect,
    confusion_matrix=cm
)

print("\nEvaluation results saved successfully!")
print("\nSTAGE 5 MODEL EVALUATION COMPLETED!")
