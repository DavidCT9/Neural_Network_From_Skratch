"""
MNIST Digit Classifier using Neural Network
Final Project - Artificial Intelligence in Videogames
Universidad Panamericana
"""

from ann import ANN
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


def load_mnist():
    """
    Load MNIST dataset from tensorflow/keras
    Returns: train_images, train_labels, test_images, test_labels
    """
    from tensorflow import keras
    
    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Dataset loaded successfully")
    print(f"Training set: {x_train.shape}")
    print(f"Test set: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test


def preprocess_image(image):
    """
    Flatten and normalize a 28x28 image to a 784-element vector
    """
    return (image.flatten() / 255.0).tolist()


def label_to_output(label):
    """
    Convert digit label (0-9) to one-hot encoded output
    """
    output = [0.0] * 10
    output[label] = 1.0
    return output


def output_to_label(output):
    """
    Convert network output to predicted digit
    """
    return output.index(max(output))


def train_network(ann, train_images, train_labels, epochs, val_images=None, val_labels=None):
    """
    Train the neural network
    """
    history = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Training samples: {len(train_images)}")
    if val_images is not None:
        print(f"Validation samples: {len(val_images)}")
    print("-" * 80)
    
    for epoch in range(epochs):
        # Training
        train_correct = 0
        train_loss = 0.0
        
        for img, label in zip(train_images, train_labels):
            input_data = preprocess_image(img)
            target = label_to_output(label)
            
            output = ann.train(input_data, target)
            
            # Calculate loss
            loss = sum((o - t) ** 2 for o, t in zip(output, target))
            train_loss += loss
            
            # Check accuracy
            if output_to_label(output) == label:
                train_correct += 1
        
        train_loss /= len(train_images)
        train_acc = (train_correct / len(train_images)) * 100
        
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_images is not None:
            val_correct = 0
            val_loss = 0.0
            
            for img, label in zip(val_images, val_labels):
                input_data = preprocess_image(img)
                target = label_to_output(label)
                output = ann.predict(input_data)
                
                loss = sum((o - t) ** 2 for o, t in zip(output, target))
                val_loss += loss
                
                if output_to_label(output) == label:
                    val_correct += 1
            
            val_loss /= len(val_images)
            val_acc = (val_correct / len(val_images)) * 100
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    
    print("-" * 80)
    return history


def evaluate(ann, test_images, test_labels):
    """
    Evaluate the neural network on test data
    """
    print(f"\nEvaluating on {len(test_images)} test samples...")
    
    correct = 0
    total_loss = 0.0
    
    for img, label in zip(test_images, test_labels):
        input_data = preprocess_image(img)
        target = label_to_output(label)
        output = ann.predict(input_data)
        
        loss = sum((o - t) ** 2 for o, t in zip(output, target))
        total_loss += loss
        
        if output_to_label(output) == label:
            correct += 1
    
    avg_loss = total_loss / len(test_images)
    accuracy = (correct / len(test_images)) * 100
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def plot_training_history(history, save_path):
    """
    Plot training and validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epochs']
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', marker='o')
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', marker='o')
    if history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.show()


def visualize_predictions(ann, images, labels, num_samples, save_path):
    """
    Visualize predictions on sample images
    """
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, img_idx in enumerate(indices):
        img = images[img_idx]
        true_label = labels[img_idx]
        
        input_data = preprocess_image(img)
        output = ann.predict(input_data)
        pred_label = output_to_label(output)
        confidence = max(output)
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        
        color = 'green' if pred_label == true_label else 'red'
        axes[idx].set_title(
            f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}',
            color=color,
            fontsize=10
        )
    
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions saved to {save_path}")
    plt.show()


def main():
    """
    Main function
    """
    print("="*80)
    print("MNIST Digit Classifier - Neural Network")
    print("Universidad Panamericana - Final Project")
    print("="*80)
    
    # Configuration
    TRAIN_SIZE = 5000
    TEST_SIZE = 1000
    VAL_SPLIT = 0.2
    
    HIDDEN_LAYERS = 2
    NEURONS_PER_LAYER = 128
    LEARNING_RATE = 0.1
    EPOCHS = 20
    
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n1. Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Use subset
    train_images = x_train[:TRAIN_SIZE]
    train_labels = y_train[:TRAIN_SIZE]
    test_images = x_test[:TEST_SIZE]
    test_labels = y_test[:TEST_SIZE]
    
    # Split train/validation
    val_size = int(len(train_images) * VAL_SPLIT)
    val_images = train_images[-val_size:]
    val_labels = train_labels[-val_size:]
    train_images = train_images[:-val_size]
    train_labels = train_labels[:-val_size]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_images)} samples")
    print(f"  Validation: {len(val_images)} samples")
    print(f"  Test: {len(test_images)} samples")
    
    # Create neural network
    print(f"\n2. Creating Neural Network...")
    print(f"   Architecture: 784 -> {NEURONS_PER_LAYER} -> {NEURONS_PER_LAYER} -> 10")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    ann = ANN(
        n_inputs=784,
        n_outputs=10,
        n_hidden=HIDDEN_LAYERS,
        n_per_hidden=NEURONS_PER_LAYER,
        alpha=LEARNING_RATE
    )
    
    # Train
    print(f"\n3. Training...")
    start_time = datetime.now()
    
    history = train_network(ann, train_images, train_labels, EPOCHS, val_images, val_labels)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Plot training history
    print(f"\n4. Generating training history plot...")
    plot_training_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))
    
    # Evaluate
    print(f"\n5. Evaluating on test set...")
    test_loss, test_acc = evaluate(ann, test_images, test_labels)
    
    # Visualize predictions
    print(f"\n6. Generating predictions visualization...")
    visualize_predictions(ann, test_images, test_labels, 15, 
                         os.path.join(OUTPUT_DIR, "predictions.png"))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Network: 784 -> {NEURONS_PER_LAYER} -> {NEURONS_PER_LAYER} -> 10")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()