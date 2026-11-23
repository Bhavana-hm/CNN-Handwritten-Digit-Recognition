
import tensorflow as tf
import numpy as np
import os

# Create output folder
os.makedirs("development", exist_ok=True)

print("\nLoading MNIST dataset...")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("\nBefore preprocessing:")
print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images to include one grayscale channel
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nAfter preprocessing:")
print("Training images shape:", x_train.shape)
print("Testing images shape:", x_test.shape)

print("\nPixel value range:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())

print("\nLabel information:")
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Save processed dataset
np.savez_compressed(
    "development/mnist_processed.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print("\nProcessed dataset saved successfully!")
print("STAGE 3 PREPROCESSING COMPLETED!")
