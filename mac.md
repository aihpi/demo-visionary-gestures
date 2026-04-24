The app is running. Here's a summary of all 5 bugs that were fixed:

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │                                            Bug                                             │                                         Fix                                         │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ from mediapipe import solutions — removed in MediaPipe 0.10.x                              │ Replaced with from mediapipe.tasks.python.vision import drawing_utils as mp_drawing │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 2   │ from mediapipe.framework.formats import landmark_pb2 — protobuf API removed                │ Removed; the Tasks API gives landmark objects directly, no proto conversion needed  │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ draw_landmarks_on_image used old proto-based NormalizedLandmarkList and wrong image format │ Rewrote to pass landmarks directly, with RGB↔BGR conversion around the draw call    │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 4   │ global ..., latest — latest is undefined                                                   │ Removed latest from the global declaration                                          │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ delegate="GPU" string + Metal crash on macOS                                               │ Changed to BaseOptions.Delegate.CPU (the enum value, not a string)                  │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  The Hand Tracking window should now be open. Press q to quit, r to enter recording mode, n to return to normal mode, and 0–4 to select a gesture class while recording.



  Code Changes                                                                                                                                                                                                                                       
              
  1. Extract processing logic from main.py                                                                                                                                                                                                           
                                                                                                                                                                                                                                                     
  Remove everything that's desktop-specific:
  - cv2.VideoCapture(0) — no webcam in K8s                                                                                                                                                                                                           
  - cv2.imshow(), cv2.waitKey(), cv2.destroyAllWindows() — no display                                                                                                                                                                                
  - select_gesture(key, ...) — no keyboard                           
  - Global mutable state (current_mode, current_gesture_class, etc.) — move to per-session scope                                                                                                                                                     
  - Switch VisionRunningMode.LIVE_STREAM → VisionRunningMode.IMAGE (synchronous, stateless per-frame — simpler for a web API)                                                                                                                        
                                                                                                                                                                                                                                                     
  2. Create api.py (new file)                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                     
  A FastAPI app exposing:                                                                                                                                                                                                                            
  GET  /healthz              — liveness/readiness probe        
  WS   /ws                   — browser streams JPEG frames, server returns predictions + annotated frame
  POST /mode                 — switch between Normal/Recording modes                                                                                                                                                                                 
  POST /gesture-class        — set gesture class for recording                                                                                                                                                                                       
  The WebSocket loop replaces the while cap.isOpened() loop in main().                                                                                                                                                                               
                                                                                                                                                                                                                                                     
  3. Update requirements.txt                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                     
  Add:                                                                                                                                                                                                                                               
  fastapi                                                                                                                                                                                                                                            
  uvicorn[standard]                                            
  websockets
  python-multipart

  ---
  Dockerfile Changes
                    
  Change the CMD to start uvicorn:
  CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]                                                                                                                                                                                  
   
  ---                                                                                                                                                                                                                                                
  deployments.yaml Changes                                     
                                                                                                                                                                                                                                                     
  Remove the command/args block (let the Dockerfile CMD run), remove GPU resources, and clean up stale env vars:
                                                                                                                                                                                                                                                     
  containers:                                                  
    - name: gestures                                                                                                                                                                                                                                 
      image: ghcr.io/aihpi/demo-visionary-gestures:sha-...     
      imagePullPolicy: Always                                                                                                                                                                                                                        
      # remove: command, args
      ports:                                                                                                                                                                                                                                         
        - name: http                                           
          containerPort: 8000                                                                                                                                                                                                                        
      # remove: WHISPER_* env vars (not used)                  
      # remove: nvidia.com/gpu resource requests                                                                                                                                                                                                     
      resources:
        requests:                                                                                                                                                                                                                                    
          memory: "512Mi"                                                                                                                                                                                                                            
          cpu: "500m"
        limits:                                                                                                                                                                                                                                      
          memory: "2Gi"                                        
          cpu: "2"

  ---                                                                                                                                                                                                                                                
  Summary of file changes
                                                                                                                                                                                                                                                     
  ┌──────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │       File       │                                   Change                                    │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ main.py          │ Remove webcam/GUI/keyboard code; expose process_frame(frame_bytes) function │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ api.py           │ Create new — FastAPI app with /healthz, /ws, /mode, /gesture-class          │                                                                                                                                                 
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                 
  │ requirements.txt │ Add fastapi, uvicorn[standard], websockets                                  │                                                                                                                                                 
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                 
  │ Dockerfile       │ Change CMD to uvicorn api:app ...                                           │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤                                                                                                                                                 
  │ deployments.yaml │ Remove command/args, remove GPU resources, remove WHISPER env vars          │
  └──────────────────┴─────────────────────────────────────────────────────────────────────────────┘                                                                                                                                                 
                                                          yes