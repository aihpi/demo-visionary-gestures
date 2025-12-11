import csv
import os

from mediapipe.tasks.python.vision import HandLandmarkerResult

# Gestures to recognize
GESTURE_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissors",
    3: "Lizard",
    4: "Spock",
}


number = -1

def prepocessing_hand_landmarks(result: HandLandmarkerResult):
    '''
    Prepocessing of Landmarks by normalizing and flattening the list of tuples
    :param result: HandLandmarkerResult
    :return: Formatted lists of handedness and hand landmarks
    '''

    landmarks = result.hand_landmarks
    handedness_list = []
    handlm_list = []

    # Looping through available hand(s)
    for idx in range(len(landmarks)):
        handedness_list.append(result.handedness[idx][0].index)

        single_handlm_list = []
        if not landmarks[idx]:
            continue

        base_x, base_y, base_z = landmarks[idx][0].x, landmarks[idx][0].y, landmarks[idx][0].z

        for i in range(len(landmarks[idx])):
            landmark = landmarks[idx][i]
            x, y, z = landmark.x - base_x, landmark.y - base_y, landmark.z - base_z
            single_handlm_list.append((x, y, z))

        if not single_handlm_list:
            continue

        def absolute_value(xyz_tuple):
            x, y, z = xyz_tuple
            return (x ** 2 + y ** 2 + z ** 2) ** 0.5

        max_value = max(absolute_value(xyz) for xyz in single_handlm_list)
        if max_value == 0:
            max_value = 1

        normalized_handlm_list = []
        for xyz_tuple in single_handlm_list:
            x, y, z = xyz_tuple
            normalized_handlm_list.append(
                (x / max_value, y / max_value, z / max_value)
            )

        handlm_list.append(normalized_handlm_list)

    return handedness_list, handlm_list


def logging(mode, current_gesture_class, handedness_list, handlm_list):
    """
    Logs the preprocessed hand landmarks to a CSV file.
    :param mode: int, 0 for "normal" (not logging), 1 for "recording"
    :param current_gesture_class: int, the class index of the gesture (e.g., 0 for Rock)
    :param handedness_list: list, handedness of detected hands
    :param handlm_list: list, preprocessed landmarks for detected hands
    """

    if mode == 0:
        return

    if mode == 1 and (0 <= current_gesture_class <= len(GESTURE_MAP)):
        gesture_name = GESTURE_MAP.get(current_gesture_class, "Unknown")

        os.makedirs('model', exist_ok=True)
        csv_path = 'model/landmarks.csv'

        try:
            file_exists = os.path.isfile(csv_path)
            is_empty = True if not file_exists or os.path.getsize(csv_path) == 0 else False
            print(f"Recording data for: {gesture_name} (Class {current_gesture_class})")

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)

                if is_empty:
                    header = ['class_id', 'handedness']
                    for i in range(21):
                        header.extend([f'x{i}', f'y{i}', f'z{i}'])
                    writer.writerow(header)

                if len(handlm_list) > 0 and len(handedness_list) > 0:
                    for i in range(len(handlm_list)):

                        if len(handlm_list[i]) != 21:
                            print(f"Warning: Expected 21 landmarks, got {len(handlm_list[i])}. Skipping this entry.")
                            continue

                        flattened_list = []
                        for xyz in handlm_list[i]:
                            flattened_list.extend(xyz)

                        row = [current_gesture_class, handedness_list[i]] + flattened_list
                        print(f"Writing row with class_id: {current_gesture_class}")
                        writer.writerow(row)
        except Exception as e:
            print(f"Error logging data: {e}")


    return

def select_gesture(key, current_mode, current_gesture_class):
    """
    Updates the recording mode and gesture class
    :param key: int, key pressed
    :param current_mode: int, current mode
    :param current_gesture_class: int, current gesture class
    :return: tuple, (new_mode, new_gesture_class)
    """

    new_mode = current_mode
    new_gesture_class = current_gesture_class

    if key == ord('n'):
        new_mode = 0
        print("Mode: Normal")
    elif key == ord('r'):
        new_mode = 1
        if new_gesture_class == -1:
            print("Select gesture class to start logging")
        else:
            gesture_name = GESTURE_MAP.get(new_gesture_class, "Unknown")
            print(f"Mode: Recording, gesture class: {gesture_name} (Class {new_gesture_class})")

    if (current_mode == 1):
        if (ord('0') <= key <= ord('9')):
            new_gesture_class = key - ord('0')
            gesture_name = GESTURE_MAP.get(new_gesture_class, "Unknown")
            print(f"Gesture class selected: {gesture_name} (Class {new_gesture_class})")

    return new_mode, new_gesture_class
