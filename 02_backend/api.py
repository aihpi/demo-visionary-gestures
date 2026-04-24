import asyncio
import base64
import json
import os

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from mediapipe.solutions import drawing_utils as mp_drawing
from preprocessing import prepocessing_hand_landmarks, logging as log_landmarks, GESTURE_MAP
import model as gesture_model_module

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# MediaPipe aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_LANDMARKER_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
KERAS_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model/rps_model.h5')
FRONTEND_PATH = os.path.join(SCRIPT_DIR, '..', '01_frontend', 'index.html')

CONFIDENCE_THRESHOLD = 0.5

gesture_model = None
if os.path.exists(KERAS_MODEL_PATH):
    gesture_model = tf.keras.models.load_model(KERAS_MODEL_PATH)


def _create_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=HAND_LANDMARKER_PATH,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _draw_landmarks(rgb_image: np.ndarray, result: HandLandmarkerResult, confidence: float) -> np.ndarray:
    color = (int(confidence * 200), 0, int((1 - confidence) * 255))
    landmark_spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
    connection_spec = mp_drawing.DrawingSpec(color=color, thickness=3)
    bgr = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
    for hand_landmarks in result.hand_landmarks:
        mp_drawing.draw_landmarks(
            bgr, hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_spec, connection_spec,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _predict(result: HandLandmarkerResult) -> tuple[str, float]:
    if gesture_model is None or not result.hand_landmarks:
        return "None", 0.0
    handedness_list, handlm_list = prepocessing_hand_landmarks(result)
    if not handlm_list:
        return "None", 0.0
    flat = []
    for xyz in handlm_list[0]:
        flat.extend(xyz)
    flat.append(handedness_list[0])
    if len(flat) != gesture_model_module.FEATURES:
        return "None", 0.0
    probs = gesture_model.predict(np.array([flat], dtype=np.float32), verbose=0)
    idx = int(np.argmax(probs[0]))
    conf = float(np.max(probs[0]))
    if conf >= CONFIDENCE_THRESHOLD:
        label = GESTURE_MAP.get(idx, "Unknown") + f" ({conf * 100:.1f}%)"
    else:
        label = "Uncertain"
    return label, conf


def _process_frame(frame_bytes: bytes, mode: int, gesture_class: int, landmarker) -> tuple[str | None, str]:
    nparr = np.frombuffer(frame_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "None"

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    prediction, confidence = "None", 0.0

    if result.hand_landmarks:
        if mode == 0:
            prediction, confidence = _predict(result)
        elif mode == 1 and gesture_class >= 0:
            handedness_list, handlm_list = prepocessing_hand_landmarks(result)
            log_landmarks(mode, gesture_class, handedness_list, handlm_list)

    annotated = rgb.copy()
    if result.hand_landmarks:
        annotated = _draw_landmarks(rgb, result, confidence)

    # Draw status text (flip trick keeps text readable in mirrored feed)
    annotated = cv2.flip(annotated, 1)
    status = f"Mode: {'Recording' if mode == 1 else 'Normal'}"
    if mode == 1 and gesture_class >= 0:
        status += f" | Gesture: {GESTURE_MAP.get(gesture_class, '?')}"
    cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    if mode == 0:
        cv2.putText(annotated, f"Prediction: {prediction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    annotated = cv2.flip(annotated, 1)

    out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode(), prediction


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_PATH)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    mode = 0
    gesture_class = -1
    loop = asyncio.get_running_loop()

    with _create_landmarker() as landmarker:
        try:
            while True:
                msg = await websocket.receive()

                if "text" in msg:
                    data = json.loads(msg["text"])
                    if data.get("type") == "mode":
                        mode = int(data["value"])
                    elif data.get("type") == "gesture_class":
                        gesture_class = int(data["value"])
                    continue

                if "bytes" not in msg:
                    continue

                frame_b64, prediction = await loop.run_in_executor(
                    None, _process_frame, msg["bytes"], mode, gesture_class, landmarker
                )
                if frame_b64:
                    await websocket.send_text(json.dumps({
                        "frame": frame_b64,
                        "prediction": prediction,
                        "mode": mode,
                        "gesture_class": gesture_class,
                    }))
        except WebSocketDisconnect:
            pass
