import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported succesfully!")
print(f"Tnsorflow version : {tf.__version__}")

# cell 2: Load MNST dataset
# MNST is a dataser of 60,000 training images and 10,000 testing iamges of handwritten digits.
# Each image is 28x28 pixels (grayscale).

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("\nMNIST Dataset Loaded:")
print(F"Training data (images) shape: {x_train.shape}")    #  (60000, 28, 28)
print(f"Training labels shape: {y_train.shape}")           #  (60000,)
print(f"Test data (images) shape: {x_test.shape}")         #  (10000, 28, 28)
print(f"Test labels shape: {y_test.shape}")                #  (10000,)  

# cell 3: Visualise an example image from the training set
print("\nVisualizing an example image:")
plt.figure(figsize=(2,2))
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show

# Check pixel values
print(f"Min pixel value: {x_train[0].min()}")
print(f"Max pixel value: {x_train[0].max()}")