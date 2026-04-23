import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import argparse
import os
import subprocess
import sys

from tensorflow.keras import Input, Model

# --- Config ---
DATASET_PATH = 'model/landmarks.csv'
MODEL_SAVE_PATH = 'model/rps_model.h5'
NUM_CLASSES = 5
SAVEDMODEL_PATH = 'model/rps_model_savedmodel'
FEATURES = 64

# Note: The number of features is now dynamically determined in load_data()


def load_data(dataset_path):
    """Loads landmark data from a CSV file."""
    print(f"Loading dataset from: {dataset_path}")
    X = []
    y = []

    with open(dataset_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header

        for i, row in enumerate(reader):
            try:
                class_id = int(row[0])
                handedness = int(row[1])
                landmarks = [float(val) for val in row[2:]]

                # Expect 21 landmarks * 3 coordinates (x, y, z) = 63 features
                if len(landmarks) != 63:
                    print(f"Warning: Row {i + 2} has {len(landmarks)} features, expected 63. Skipping row.")
                    continue

                combined_features = landmarks + [handedness]
                X.append(combined_features)
                y.append(class_id)
            except ValueError as e:
                print(f"Skipping row {i + 2} due to ValueError (e.g. non-numeric data): {row} - Error: {e}")
                continue
            except IndexError as e:
                print(f"Skipping row {i + 2} due to IndexError (e.g. incomplete row): {row} - Error: {e}")
                continue

    if not X or not y:
        print("Error: No data loaded. Check CSV path and format.")
        return None, None, 0

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # One-hot encode the labels
    y_categorical = to_categorical(y, num_classes=NUM_CLASSES)

    num_features = X.shape[1]
    print(f"Dataset loaded successfully: {len(X)} samples, {num_features} features each.")
    return X, y_categorical, num_features


def create_model(input_shape, num_classes):
    """Creates and compiles the neural network model."""
    print(f"Creating model with input shape: ({input_shape},), num classes: {num_classes}")

    model = Sequential([
        Input(shape=(input_shape,)),

        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        # Output layer with softmax for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Trains the model with early stopping and learning rate reduction."""
    print("Starting model training...")

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Increased patience slightly
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Test Loss: {loss:.4f}")

    return history


def save_and_convert_model(model, keras_save_path, savedmodel_export_path):
    """
    Saves the Keras model in the native .keras and SavedModel formats, then
    attempts to convert the SavedModel for use with TensorFlow.js.
    """
    print(f"Saving model to {keras_save_path}...")
    os.makedirs(os.path.dirname(keras_save_path), exist_ok=True)

    # UPDATED: Save in the modern, recommended native Keras format
    model.save(keras_save_path)
    print(f"Model saved in native Keras format: {keras_save_path}")

    # Export in SavedModel format, which is the best input for TF.js conversion
    model.export(savedmodel_export_path)
    print(f"Model exported in SavedModel format: {savedmodel_export_path}")

    print("\n" + "=" * 50)
    print("Attempting to convert SavedModel to TensorFlow.js format...")
    try:
        # Use sys.executable to ensure the correct python/pip environment is used
        subprocess.run([sys.executable, 'converter.py'], check=True)
        print("Model conversion script executed successfully.")
    except FileNotFoundError:
        print("Error: 'converter.py' not found. Skipping TF.js conversion.")
    except subprocess.CalledProcessError as e:
        print(f"Error during model conversion: {e}")
    print("=" * 50 + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or convert the gesture recognition model.")
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'convert'],
        help="Set the script's mode: 'train' (default) to train a new model, or 'convert' to re-convert an existing one."
    )
    parser.add_argument(
        '--input_model',
        type=str,
        default=MODEL_SAVE_PATH,
        help="Path to the model to load for 'convert' mode."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help="Number of epochs for training."
    )
    args = parser.parse_args()

    if args.mode == 'train':
        print("\n--- Running in TRAIN mode ---\n")
        X_data, y_data_categorical, num_features = load_data(DATASET_PATH)

        if X_data is not None and y_data_categorical is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data_categorical, test_size=0.2, random_state=42, stratify=y_data_categorical
            )

            print(f"\nData split complete:")
            print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
            print(f"Testing data shape:  {X_test.shape}, Testing labels shape:  {y_test.shape}\n")

            rps_model = create_model(num_features, NUM_CLASSES)
            train_model(rps_model, X_train, y_train, X_test, y_test, epochs=args.epochs)

            # After training, save and convert the final model
            save_and_convert_model(rps_model, MODEL_SAVE_PATH)

            print(f"Training complete. Model saved and converted.")
        else:
            print("Could not proceed with training due to data loading issues.")

    elif args.mode == 'convert':
        print(f"\n--- Running in CONVERT-ONLY mode ---\n")
        if not os.path.exists(args.input_model):
            print(f"Error: Model file not found at '{args.input_model}'")
            print("Please train a model first or provide a valid path using --input_model")
        else:
            print(f"Loading model from: {args.input_model}")
            try:
                # Load the existing model
                model_to_convert = tf.keras.models.load_model(args.input_model)
                model_to_convert.summary()
                save_and_convert_model(model_to_convert, MODEL_SAVE_PATH, SAVEDMODEL_PATH)
                print(f"Model successfully loaded, re-saved, and converted.")
            except Exception as e:
                print(f"An error occurred while loading or saving the model: {e}")