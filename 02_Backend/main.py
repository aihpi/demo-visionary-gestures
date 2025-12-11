import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf
import model
from preprocessing import prepocessing_hand_landmarks, select_gesture, logging, GESTURE_MAP
import preprocessing

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please download the 'hand_landmarker.task' from:")
    print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit()

KERAS_MODEL_PATH = os.path.join(script_dir, 'model/rps_model.h5')
CONFIDENCE_THRESHOLD = 0.5
PREDICTION_TEXT_COLOR = (200, 0, 0)

if os.path.exists(KERAS_MODEL_PATH):
    print(f"Loading Keras model from: {KERAS_MODEL_PATH}")
    gesture_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("Keras model loaded successfully.")
else:
    print(f"Error: Keras model not found at {KERAS_MODEL_PATH}. Please train the model first.")
    gesture_model = None

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode




MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1

current_mode = 0
current_gesture_class = -1
annotated_image = None
latest_prediction = "None"
latest_confidence = 0.0

# Helper function to draw the hand landmarks on the image (from google mediapipe doc [https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1])
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image_copy = np.copy(rgb_image)

  global latest_confidence

  color = (latest_confidence * 200, 0, (1 - latest_confidence) * 255)
  landmark_spec = solutions.drawing_styles.DrawingSpec(color=color, thickness=2, circle_radius=2)
  connection_spec = solutions.drawing_styles.DrawingSpec(color=color, thickness=3)

  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image_copy,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      landmark_spec,
        connection_spec
    )
    height, width, _ = annotated_image_copy.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]

    if not x_coordinates or not y_coordinates: continue

    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN


    #cv2.putText(annotated_image_copy, f"{handedness[0].category_name}", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image_copy


def result_processing_callback(result: HandLandmarkerResult, output_image, timestamp_ms):
    global annotated_image, current_mode, current_gesture_class, latest

    # Preprocessing landmarks
    handedness_list, handlm_list = prepocessing_hand_landmarks(result)

    predict_gesture(result)

    # logging if mode and gesture class are chosen
    if current_mode == 1 and current_gesture_class != -1:
        logging(current_mode, current_gesture_class, handedness_list , handlm_list)

    image_np = output_image.numpy_view()

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        annotated_image_temp = draw_landmarks_on_image(image_np, result)
    else:
        annotated_image_temp = np.copy(image_np)

    status_text = f"Mode: {'Recording' if current_mode == 1 else 'Normal'}"
    if current_mode == 1:
        gesture_name = GESTURE_MAP.get(current_gesture_class, "None Selected")
        status_text += f" | Gesture: {gesture_name} ({current_gesture_class if current_gesture_class != -1 else '_'})"

    annotated_image_temp = cv2.flip(annotated_image_temp, 1)
    cv2.putText(annotated_image_temp, status_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if current_mode == 0:
        cv2.putText(annotated_image_temp, f"Prediction: {latest_prediction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, PREDICTION_TEXT_COLOR, 2, cv2.LINE_AA)
    annotated_image_temp = cv2.flip(annotated_image_temp, 1)



    # Conversion to BGR for OpenCV, so the imshow function works correctly
    annotated_image = cv2.cvtColor(annotated_image_temp, cv2.COLOR_RGB2BGR)


def predict_gesture(result: HandLandmarkerResult):
    global latest_prediction, current_mode, latest_confidence
    current_prediction_text = "None"
    prediction_confidence = None

    handedness_list, handlm_list_normalized = prepocessing_hand_landmarks(result)

    if gesture_model is not None and current_mode == 0 and handlm_list_normalized:

        single_hand_landmarks_normalized = handlm_list_normalized[0]

        flattened_landmarks = []
        for xyz_tuple in single_hand_landmarks_normalized:
            flattened_landmarks.extend(xyz_tuple)
        flattened_landmarks = flattened_landmarks + [handedness_list[0]]

        if len(flattened_landmarks) == model.FEATURES:
            input_data = np.array([flattened_landmarks], dtype=np.float32)

            # Make prediction
            prediction_probabilities = gesture_model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(prediction_probabilities[0])
            prediction_confidence = np.max(prediction_probabilities[0])

            if prediction_confidence >= CONFIDENCE_THRESHOLD:
                current_prediction_text = GESTURE_MAP.get(predicted_class_index, "Unknown")
                current_prediction_text += f" ({prediction_confidence * 100:.1f}%)"
            else:
                current_prediction_text = "Uncertain"
        else:
            current_prediction_text = "Error: Landmark count mismatch"

    if prediction_confidence is not None:
        latest_confidence = prediction_confidence
    latest_prediction = current_prediction_text


# Options to create HandLandmarker instance
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate="GPU"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_processing_callback)

# Video Loop
def main():
    global current_mode, current_gesture_class, annotated_image

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open webcam")
        return

    # Main Loop with landmarker
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # Reading from webcam
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue

            # Flipping the image so the image is not mirrored
            # Converting image to RGB for mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            # monotonically increasing time in milliseconds
            timestamp_ms = int(time.perf_counter() * 1000)
            # detection function
            landmarker.detect_async(mp_image, timestamp_ms)

            # Display of Image
            if annotated_image is not None:
                flipped_image = cv2.flip(annotated_image, 1)
            else:
                flipped_image = cv2.flip(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB), 1)
            cv2.imshow('Hand Tracking', flipped_image)

            key = cv2.waitKey(5)
            if key & 0xFF == ord('q'):
                break

            current_mode, current_gesture_class = select_gesture(key, current_mode, current_gesture_class)

        cap.release()
        cv2.destroyAllWindows()






if __name__ == "__main__":
    main()