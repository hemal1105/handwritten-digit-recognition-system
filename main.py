import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# Cell 2: Load the MNIST dataset
# MNIST is a dataset of 60,000 training images and 10,000 testing images of handwritten digits.
# Each image is 28x28 pixels (grayscale).

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("\nMNIST Dataset Loaded:")
print(f"Training data (images) shape: {x_train.shape}") # (60000, 28, 28)
print(f"Training labels shape: {y_train.shape}")     # (60000,)
print(f"Test data (images) shape: {x_test.shape}")       # (10000, 28, 28)
print(f"Test labels shape: {y_test.shape}")       # (10000,)

# Cell 3: Visualize an example image from the training set
print("\nVisualizing an example image:")
plt.figure(figsize=(2, 2)) # Set figure size for better display
plt.imshow(x_train[0], cmap='gray') # Display the first image in grayscale
plt.title(f"Label: {y_train[0]}") # Show its corresponding label
plt.axis('off') # Hide axis ticks for a cleaner image
plt.show()

# Check pixel values
print(f"Min pixel value: {x_train[0].min()}") # Should be 0 (black)
print(f"Max pixel value: {x_train[0].max()}") # Should be 255 (white)

# Cell 4: Normalise pixel values
# Neural networks often perform better when input values are scaled,
# typically to a range like [0,1].
# Current pixel values are 0-255, so we divide by 255.0 to scale tehm.

x_train = x_train / 255.0
x_test = x_test / 255.0

print("\nPixel values normalised to [0, 1]:")
print(f"Min pixel value after normalization: {x_train[0].min()}")
print(f"Max pixel value after normalization: {x_train[0].max()}")

# Cell 5: Reshape data for CNN input
# Convolutional Neural Networks (CNNs) expect input images in a specific format:
# (number_of_images, height, width, channels)
# MNIST images are 28x28 grayscale, so they have 1 channel.

x_train = x_train.reshape(-1, 28, 28, 1) # -1 means "infer the number of images"
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nData reshaped for CNN input:")
print(f"Reshaped training data shape: {x_train.shape}")
print(f"Reshaped test data shape: {x_test.shape}")

# Cell 6: One-Hot Encode Labels
# Our labels (y_train, y_test) are currently single integers (e.g., 0, 1, ..., 9).
# For multi-class classification with 'categorical_crossentropy' loss, Keras expects
# labels to be "one-hot encoded" (e.g., 5 becomes [0,0,0,0,0,1,0,0,0,0]).

num_classes = 10 # Digits 0-9
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

print("\nLabels one-hot encoded:")
print(f"Original label for first training image: {np.argmax(y_train[0])}") # To show original
print(f"One-hot encoded label for first training image: {y_train[0]}")
print(f"One-hot encoded training labels shape: {y_train.shape}")

# Cell 7: Define the CNN Model Architecture
# We'll use a Sequential model, which is a linear stack of layers.

model = keras.models.Sequential([
    # Convolutional Layer 1: Learn basic features (edges, textures)
    # 32 filters, each 3x3 pixels. 'relu' activation for non-linearity.
    # input_shape is crucial for the first layer, matching our preprocessed image shape.
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max Pooling Layer 1: Reduce spatial dimensions, make features more robust to position changes.
    # Takes the maximum value in a 2x2 window.
    keras.layers.MaxPooling2D((2, 2)),

    # Convolutional Layer 2: Learn more complex features
    # 64 filters, still 3x3.
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Max Pooling Layer 2
    keras.layers.MaxPooling2D((2, 2)),

    # Flatten Layer: Convert the 2D feature maps into a 1D vector
    # This prepares the data for the fully connected (Dense) layers.
    keras.layers.Flatten(),

    # Dense (Fully Connected) Layer 1: High-level reasoning
    # 128 neurons. 'relu' activation.
    keras.layers.Dense(128, activation='relu'),

    # Dropout Layer: Helps prevent overfitting by randomly setting a fraction of inputs to 0 at each update.
    # 0.3 means 30% of neurons will be "dropped out" during training.
    keras.layers.Dropout(0.3),

    # Output Layer: Make the final classification decision
    # 10 neurons (one for each digit 0-9).
    # 'softmax' activation outputs a probability distribution over the 10 classes (sums to 1).
    keras.layers.Dense(num_classes, activation='softmax')
])

print("\nModel architecture defined.")
model.summary() # Print a summary of the model's layers and parameters

# Cell 8: Compile the Model
# This configures the learning process: how the model updates itself.

model.compile(optimizer='adam',      # Optimizer: How the model adjusts weights during training. 'adam' is a popular choice.
              loss='categorical_crossentropy', # Loss function: Measures how "wrong" the model's predictions are.
                                            # Used with one-hot encoded labels.
              metrics=['accuracy'])  # Metric: What to monitor during training to evaluate performance.

print("\nModel compiled successfully.")

# Cell 9: Train the Model
print("\nTraining the model...")

epochs = 10 # How many times the model will go through the entire training dataset
batch_size = 64 # How many samples (images) are processed before updating the model's weights
validation_split = 0.1 # Use 10% of the training data as a validation set during training

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=1) # verbose=1 shows a progress bar and training metrics per epoch

print(f"\nModel training complete after {epochs} epochs.")

# Cell 10: Evaluate the Model on the Test Data
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2) # verbose=2 shows less output during evaluation

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Cell 11: Plot Training History
print("\nPlotting training history...")

plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout() # Adjusts plot parameters for a tight layout
plt.show()

# Cell 12: Make Predictions on Sample Test Images
print("\nMaking predictions on a few test images...")

# Select the first 10 test images for prediction
sample_images = x_test[:10]
sample_true_labels = np.argmax(y_test[:10], axis=1) # Get original integer labels from one-hot

predictions = model.predict(sample_images)

# Convert predicted probabilities to class labels (the digit with the highest probability)
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Predictions:")
for i in range(len(sample_images)):
    print(f"Image {i+1}: True Label = {sample_true_labels[i]}, Predicted Label = {predicted_labels[i]}")
    # You can also see the full probability distribution for a prediction:
    # print(f"Probabilities: {predictions[i].round(2)}") # Rounded to 2 decimal places

# Cell 13: Visualize Sample Predictions
print("\nVisualizing sample predictions and their true/predicted labels...")

plt.figure(figsize=(12, 6))
for i in range(len(sample_images)):
    plt.subplot(2, 5, i + 1) # Arrange plots in 2 rows, 5 columns
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray') # Reshape back to 2D for display
    color = 'green' if predicted_labels[i] == sample_true_labels[i] else 'red'
    plt.title(f"True: {sample_true_labels[i]}\nPred: {predicted_labels[i]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Cell 14: Save the Trained Model
print("\nSaving the trained model...")
model.save('mnist_cnn_model.h5') # Saves the model's architecture, weights, and training configuration
print("Model saved as 'mnist_cnn_model.h5'")

# What to Learn:
# - How to persist your trained model to a file.
# - This allows you to deploy or reuse your model later without retraining.