import subprocess
import os
import shutil

INPUT_SAVEDMODEL_DIR = 'model/rps_model_savedmodel'
OUTPUT_TFJS_DIR = 'model/rps_model_tfjs'


def convert():
    """Converts the SavedModel to TensorFlow.js format."""

    if not os.path.exists(INPUT_SAVEDMODEL_DIR):
        print(f"Error: Input SavedModel directory not found at '{INPUT_SAVEDMODEL_DIR}'")
        print("Please train the model first by running: python model.py")
        return

    if os.path.exists(OUTPUT_TFJS_DIR):
        print(f"Removing existing TF.js model at: {OUTPUT_TFJS_DIR}")
        shutil.rmtree(OUTPUT_TFJS_DIR)

    print(f"Starting conversion of '{INPUT_SAVEDMODEL_DIR}' to TF.js format...")

    command = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        f"--output_format=tfjs_graph_model",
        f"--signature_name=serving_default",
        f"--saved_model_tags=serve",
        INPUT_SAVEDMODEL_DIR,
        OUTPUT_TFJS_DIR
    ]

    try:
        subprocess.run(command, check=True)
        print("\n" + "=" * 50)
        print("Conversion successful!")
        print(f"   Model saved to: {OUTPUT_TFJS_DIR}")
        print("=" * 50)
    except FileNotFoundError:
        print("\nError: `tensorflowjs_converter` command not found.")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred during conversion: {e}")


if __name__ == "__main__":
    convert()