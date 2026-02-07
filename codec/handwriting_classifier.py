import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
MODEL_FILENAME = 'mnist_model.h5'

def load_and_preprocess_data():
    """
    Loads MNIST data directly via TensorFlow (No Pandas needed for image arrays).
    """
    print("Loading MNIST dataset via Keras API...")
    # This built-in loader returns the data as NumPy arrays (X_train, y_train), (X_test, y_test)
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reshape: (Number of samples, Height, Width, Channels)
    # MNIST is grayscale, so Channels = 1
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    """
    Builds a Convolutional Neural Network (CNN).
    """
    model = models.Sequential([
        # Extract features with 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Extract higher-level features with 64 filters
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Final Convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the 3D output to 1D for the Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        
        # Output layer with 10 neurons (one for each digit 0-9) using Softmax
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_and_save_model():
    """
    Trains the model and saves it to disk as an H5 file.
    """
    print("--- Training New Model ---")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training started (3 epochs)...")
    model.fit(train_images, train_labels, epochs=3, batch_size=64, validation_split=0.1)
    
    print("Evaluating performance on test data...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    
    print(f"Saving model to {MODEL_FILENAME}...")
    model.save(MODEL_FILENAME)
    print("Model saved successfully!")
    return model

def visualize_predictions(model, test_images, test_labels):
    """
    Picks random images from test set and displays predictions using Matplotlib.
    """
    print("\nVisualizing random predictions...")
    
    # Pick 10 random indices
    num_samples = 10
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    samples = test_images[indices]
    sample_labels = test_labels[indices]
    
    # Generate predictions for the chosen samples
    predictions = model.predict(samples)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Display image
        ax.imshow(samples[i].reshape(28, 28), cmap='gray')
        
        # Get prediction result (index of highest probability)
        pred_label = np.argmax(predictions[i])
        true_label = sample_labels[i]
        
        # Color coding: Green for correct, Red for incorrect
        color = 'green' if pred_label == true_label else 'red'
        
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("--- Handwritten Digit Recognizer ---")
    
    # Logic to either load an existing model or train a new one
    if os.path.exists(MODEL_FILENAME):
        choice = input(f"Found saved model '{MODEL_FILENAME}'.\n[1] Load & Visualize\n[2] Retrain Model\nEnter choice (1/2): ").strip()
    else:
        print("No saved model found.")
        choice = "2"

    if choice == "2":
        model = train_and_save_model()
    else:
        print(f"Loading model from {MODEL_FILENAME}...")
        model = tf.keras.models.load_model(MODEL_FILENAME)
        
    # Reload test data for verification and visualization
    _, (test_images, test_labels) = load_and_preprocess_data()
    
    print("Verifying accuracy...")
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Model Accuracy: {acc*100:.2f}%")
    
    visualize_predictions(model, test_images, test_labels)

if __name__ == "__main__":
    main()
