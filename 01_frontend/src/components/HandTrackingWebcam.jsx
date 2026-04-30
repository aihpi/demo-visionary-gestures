import React, { useRef, useEffect, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import useGestureRecognition from './useGestureRecognition';
import modelUrl from '../../public/model/model.json?url';
import binUrl from '../../public/model/group1-shard1of1.bin?url';

const getModelPath = () => {
  const base = import.meta.env.BASE_URL || '/';
  return `${base}model/model.json`.replace(/\/+/g, '/');
};

const LoadingLogo = ({ fillContainer }) => {
  const logoStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    zIndex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#666',
    fontSize: fillContainer ? '24px' : '18px',
    fontWeight: 'bold'
  };

  const spinnerStyle = {
    width: fillContainer ? '60px' : '40px',
    height: fillContainer ? '60px' : '40px',
    border: '4px solid #f3f3f3',
    borderTop: '4px solid #1d1d1f',
    borderRadius: '50%',
    animation: 'spin 2s linear infinite',
    marginBottom: '16px'
  };

  return (
    <>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
      <div style={logoStyle}>
        <div style={spinnerStyle}></div>
      </div>
    </>
  );
};

const HandTrackingWebcam = ({
  onGestureDetected = null,
  showLandmarks = true,
  mirrored = true,
  gameMode = 'rps',
  modelPath = getModelPath(),
  fillContainer = false,
  width = 640,
  height = 480,
  enableRegionCropping = true
}) => {

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameId = useRef(null);

  const cropCanvasRef = useRef(null);
  const cropContextRef = useRef(null);

  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [handLandmarker, setHandLandmarker] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isVideoStreaming, setIsVideoStreaming] = useState(false);
  const [videoMetrics, setVideoMetrics] = useState({ width: 0, height: 0, offsetX: 0, offsetY: 0 });

  const canvasContextRef = useRef(null);
  const lastFrameTime = useRef(0);
  const targetFPS = 60;
  const frameInterval = 1000 / targetFPS;
  const handsDetectedRef = useRef(false);

  const frameSkipCounter = useRef(0);
  const PROCESS_EVERY_N_FRAMES = 1;

  const [activeCropRegion, setActiveCropRegion] = useState({ x: 0, y: 0, width: 1, height: 1 });

  const {
    isModelLoaded,
    detectedGesture,
    gestureConfidence,
    gestureColor,
    predictGesture,
    resetGestureState
  } = useGestureRecognition(modelPath, gameMode);

  const calculateCropRegion = useCallback((videoWidth, videoHeight) => {
    if (!enableRegionCropping) {
      return { x: 0, y: 0, width: videoWidth, height: videoHeight };
    }

    const cropWidth = videoWidth * 0.5;
    return {
      x: 0,
      y: 0,
      width: cropWidth,
      height: videoHeight
    };
  }, [enableRegionCropping]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !enableRegionCropping) return;

    const updateCropRegion = () => {
      if (video.videoWidth && video.videoHeight) {
        const region = calculateCropRegion(video.videoWidth, video.videoHeight);
        setActiveCropRegion(region);

        if (!cropCanvasRef.current) {
          cropCanvasRef.current = document.createElement('canvas');
        }

        cropCanvasRef.current.width = region.width;
        cropCanvasRef.current.height = region.height;
        cropContextRef.current = cropCanvasRef.current.getContext('2d');

        const ctx = cropContextRef.current;
        ctx.imageSmoothingEnabled = false;

      }
    };

    video.addEventListener('loadedmetadata', updateCropRegion);
    if (video.videoWidth) updateCropRegion();

    return () => {
      video.removeEventListener('loadedmetadata', updateCropRegion);
    };
  }, [enableRegionCropping, calculateCropRegion]);

  // Function to map landmarks from crop space back to full video space
  const mapLandmarksToFullVideo = useCallback((landmarks) => {
    if (!enableRegionCropping || !landmarks) return landmarks;

    return landmarks.map(handLandmarks =>
      handLandmarks.map(landmark => ({
        ...landmark,
        x: (landmark.x * activeCropRegion.width + activeCropRegion.x) / (activeCropRegion.x + activeCropRegion.width + (videoRef.current?.videoWidth - activeCropRegion.x - activeCropRegion.width) || 1),
        y: (landmark.y * activeCropRegion.height + activeCropRegion.y) / (videoRef.current?.videoHeight || 1)
      }))
    );
  }, [enableRegionCropping, activeCropRegion]);

  const updateVideoMetrics = useCallback(() => {
    if (!fillContainer) return;
    
    const video = videoRef.current;
    if (!video || !isVideoStreaming) return;

    const containerWidth = video.parentElement.clientWidth;
    const containerHeight = video.parentElement.clientHeight;

    const videoAspectRatio = video.videoWidth / video.videoHeight;
    const containerAspectRatio = containerWidth / containerHeight;

    let renderedWidth, renderedHeight, offsetX, offsetY;

    if (videoAspectRatio > containerAspectRatio) {
      renderedHeight = containerHeight;
      renderedWidth = renderedHeight * videoAspectRatio;
      offsetX = 0;
      offsetY = 0;
    } else {
      renderedWidth = containerWidth;
      renderedHeight = renderedWidth / videoAspectRatio;
      offsetX = 0;
      offsetY = (containerHeight - renderedHeight) / 2;
    }

    setVideoMetrics({ width: renderedWidth, height: renderedHeight, offsetX, offsetY });

    if (canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.style.width = `${renderedWidth}px`;
      canvas.style.height = `${renderedHeight}px`;
      canvas.style.left = `${offsetX}px`;
      canvas.style.top = `${offsetY}px`;
      canvas.width = renderedWidth;
      canvas.height = renderedHeight;
    }
  }, [videoRef, canvasRef, isVideoStreaming, fillContainer]);

  // Video streaming effects (unchanged)
  useEffect(() => {
    if (!fillContainer) return;
    
    const video = videoRef.current;
    if (!video) return;

    const handlePlaying = () => setIsVideoStreaming(true);
    const handlePause = () => setIsVideoStreaming(false);
    const handleEnded = () => setIsVideoStreaming(false);
    const handleLoadedMetadata = () => updateVideoMetrics();
    const handleResize = () => updateVideoMetrics();

    if (!isWebcamActive) {
      setIsVideoStreaming(false);
    }

    video.addEventListener('playing', handlePlaying);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    window.addEventListener('resize', handleResize);

    return () => {
      video.removeEventListener('playing', handlePlaying);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      window.removeEventListener('resize', handleResize);
    };
  }, [isWebcamActive, videoRef, updateVideoMetrics, fillContainer]);

  useEffect(() => {
    if (fillContainer && isVideoStreaming) {
      updateVideoMetrics();
    }
  }, [isVideoStreaming, updateVideoMetrics, fillContainer]);

  useEffect(() => {
    let isMounted = true;

    const initializeHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          // PERFORMANCE BOOST: More aggressive detection settings
          minHandDetectionConfidence: 0.5, // Reduced from 0.7
          minHandPresenceConfidence: 0.3,  // Reduced from 0.5
          minTrackingConfidence: 0.3       // Reduced from 0.5
        });

        if (isMounted) {
          setHandLandmarker(landmarker);
          setIsInitialized(true);
        }
      } catch (error) {
        console.error("Error initializing MediaPipe:", error);
      }
    };

    initializeHandLandmarker();

    return () => {
      isMounted = false;
      if (handLandmarker) {
        handLandmarker.close();
      }
    };
  }, []);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: fillContainer ? { ideal: 1280 } : width,
          height: fillContainer ? { ideal: 720 } : height,
          facingMode: 'user',
          frameRate: { ideal: 60, max: 60 }
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsWebcamActive(true);
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  }, [width, height, fillContainer]);

  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);

    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
  }, []);

  const drawLandmarks = useCallback((landmarks) => {
    if (!showLandmarks || !landmarks || landmarks.length === 0) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    if (!canvasContextRef.current) {
      if (fillContainer) {
        canvasContextRef.current = canvas.getContext('2d');
      } else {
        canvas.width = video.videoWidth || width;
        canvas.height = video.videoHeight || height;
        canvasContextRef.current = canvas.getContext('2d');
      }

      const ctx = canvasContextRef.current;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.imageSmoothingEnabled = false;
    }

    const ctx = canvasContextRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (landmarks.length > 0) {
      const handLandmarks = landmarks[0];

      let baseColor = "128, 128, 128";
      let lineWidth = 4;
      let opacity = 0.6;

      if (gestureConfidence >= 0.3) {
        baseColor = gestureColor;
        lineWidth = 2 + (gestureConfidence * 6);
        opacity = 0.5 + (gestureConfidence * 0.5);
      }

      const color = `rgba(${baseColor}, ${opacity})`;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = lineWidth;

      const connections = [
        [0,1], [1,2], [2,3], [3,4],
        [0,5], [5,6], [6,7], [7,8],
        [0,9], [9,10], [10,11], [11,12],
        [0,13], [13,14], [14,15], [15,16],
        [0,17], [17,18], [18,19], [19,20],
        [5,9], [9,13], [13,17]
      ];

      ctx.beginPath();
      for (let i = 0; i < connections.length; i++) {
        const [start, end] = connections[i];
        const startX = handLandmarks[start].x * canvas.width;
        const startY = handLandmarks[start].y * canvas.height;
        const endX = handLandmarks[end].x * canvas.width;
        const endY = handLandmarks[end].y * canvas.height;

        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
      }
      ctx.stroke();

      const pointSize = Math.max(2, lineWidth * 0.8);
      ctx.beginPath();
      for (let i = 0; i < handLandmarks.length; i++) {
        const landmark = handLandmarks[i];
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;
        ctx.moveTo(x + pointSize, y);
        ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
      }
      ctx.fill();
    }
  }, [showLandmarks, gestureConfidence, gestureColor, fillContainer, width, height]);

  // PERFORMANCE BOOST: Optimized detection loop
  const detectHands = useCallback(async () => {
    if (!isWebcamActive) return;

    const currentTime = performance.now();

    // PERFORMANCE BOOST: Reduce frame rate limiting
    if (currentTime - lastFrameTime.current < frameInterval) {
      animationFrameId.current = requestAnimationFrame(detectHands);
      return;
    }

    lastFrameTime.current = currentTime;

    // PERFORMANCE BOOST: Frame skipping for processing-heavy operations
    frameSkipCounter.current = (frameSkipCounter.current + 1) % PROCESS_EVERY_N_FRAMES;

    const video = videoRef.current;
    if (video && handLandmarker && video.readyState >= 2) {
      try {
        let inputElement = video;

        // Use cropped canvas if region cropping is enabled
        if (enableRegionCropping && cropContextRef.current && activeCropRegion.width > 0) {
          cropContextRef.current.drawImage(
            video,
            activeCropRegion.x, activeCropRegion.y, activeCropRegion.width, activeCropRegion.height,
            0, 0, activeCropRegion.width, activeCropRegion.height
          );
          inputElement = cropCanvasRef.current;
        }

        const results = handLandmarker.detectForVideo(inputElement, currentTime);

        if (results.landmarks && results.landmarks.length > 0) {
          handsDetectedRef.current = true;

          // Map landmarks back to full video coordinates
          const mappedLandmarks = enableRegionCropping
            ? mapLandmarksToFullVideo(results.landmarks)
            : results.landmarks;

          // PERFORMANCE BOOST: Only predict gesture every frame (not rate limited)
          let predictionResult = null;
          if (isModelLoaded) {
            predictionResult = await predictGesture(results.landmarks, results.handedness);
          }

          // Draw using mapped landmarks
          drawLandmarks(mappedLandmarks);

          if (onGestureDetected) {
            onGestureDetected({
              landmarks: mappedLandmarks,
              handedness: results.handedness,
              timestamp: currentTime,
              gesture: predictionResult ? {
                name: predictionResult.gestureName,
                confidence: predictionResult.confidence,
                isValid: predictionResult.isValidForGameplay
              } : null
            });
          }
        } else {
          if (handsDetectedRef.current) {
            handsDetectedRef.current = false;
            resetGestureState();
          }

          if (canvasContextRef.current) {
            const canvas = canvasRef.current;
            canvasContextRef.current.clearRect(0, 0, canvas.width, canvas.height);
          }
        }
      } catch (error) {
        console.error('Hand detection error:', error);
      }
    }

    animationFrameId.current = requestAnimationFrame(detectHands);
  }, [isWebcamActive, handLandmarker, isModelLoaded, drawLandmarks, predictGesture, onGestureDetected, resetGestureState, enableRegionCropping, activeCropRegion, mapLandmarksToFullVideo]);

  // Effects (unchanged)
  useEffect(() => {
    if (isWebcamActive && isInitialized) {
      detectHands();
    }

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
    };
  }, [isWebcamActive, isInitialized, detectHands]);

  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  useEffect(() => {
    if (isInitialized) {
      startWebcam();
    }
  }, [isInitialized, startWebcam]);

  // Determine if loading should be shown
  const showLoading = !isInitialized || !isWebcamActive || !isVideoStreaming;

  // Render logic with loading logo
  if (fillContainer) {
    return (
      <div className="w-full h-full relative">
        {/* Loading Logo Background */}
        {showLoading && <LoadingLogo fillContainer={true} />}

        <div className="relative w-full h-full overflow-hidden shadow-lg" 
             style={{ transform: mirrored ? 'scaleX(-1)' : 'none' }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="h-full w-auto absolute left-0"
            style={{
              filter: 'grayscale(80%)',
              objectFit: 'cover',
              objectPosition: 'left center',
              zIndex: 2
            }}
          />

          <canvas
            ref={canvasRef}
            className="absolute pointer-events-none"
            style={{
              width: `${videoMetrics.width}px`,
              height: `${videoMetrics.height}px`,
              left: `${videoMetrics.offsetX}px`,
              top: `${videoMetrics.offsetY}px`,
              zIndex: 3
            }}
          />
        </div>

        {detectedGesture !== 'None' && (
          <div className="absolute top-4 left-4 bg-white bg-opacity-75 px-3 py-2 text-sm font-bold min-w-24 max-w-64" style={{ zIndex: 4 }}>
            {detectedGesture}
          </div>
        )}

      </div>
    );
  }

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      {/* Loading Logo Background */}
      {showLoading && <LoadingLogo fillContainer={false} />}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: width,
          height: height,
          transform: mirrored ? 'scaleX(-1)' : 'none',
          borderRadius: '0px',
          background: '#fff',
          zIndex: 2
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: width,
          height: height,
          transform: mirrored ? 'scaleX(-1)' : 'none',
          pointerEvents: 'none',
          borderRadius: '0px',
          zIndex: 3
        }}
      />

      {detectedGesture !== 'None' && (
        <div style={{
          position: 'absolute',
          top: '15px',
          left: '15px',
          background: '#000',
          color: 'white',
          padding: '8px 12px',
          fontSize: '18px',
          fontWeight: 'bold',
          zIndex: 4
        }}>
          {detectedGesture}
        </div>
      )}
    </div>
  );
};

export default HandTrackingWebcam;