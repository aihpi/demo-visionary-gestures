import { useState, useEffect, useCallback, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import {colors} from "../config/themeConfig.js";

// Constants for gesture recognition
const FEATURES = 64; // 21 landmarks * 3 coordinates + 1 handedness
const CONFIDENCE_THRESHOLD = 0.3;
const GESTURE_MAP = {
  0: 'Rock',
  1: 'Paper',
  2: 'Scissors',
  3: 'Lizard',
  4: 'Spock'
};


const useGestureRecognition = (modelPath, gameMode = 'rps') => {
  const [model, setModel] = useState(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [detectedGesture, setDetectedGesture] = useState('None');
  const [gestureConfidence, setGestureConfidence] = useState(0);
  const [gestureColor, setGestureColor] = useState("128, 128, 128");

  const GESTURE_COLORS = colors();

  // Performance optimization refs
  const lastPredictionTime = useRef(0);
  const predictionInterval = 100; // Limit predictions to 10 FPS for efficiency

  // OPTIMIZATION: Pre-allocated tensor buffer for reuse
  const inputTensorRef = useRef(null);
  const inputDataArrayRef = useRef(new Float32Array(FEATURES));

  // Store the current stable prediction result
  const currentPredictionRef = useRef({
    classIndex: null,
    confidence: 0,
    gestureName: 'None',
    isValidForGameplay: false,
    color: "128, 128, 128"
  });

  // Load TensorFlow model
  useEffect(() => {
    let isMounted = true;

    const loadModel = async () => {
      if (!modelPath) {
        return;
      }

      try {
        setIsLoading(true);
        setError(null);

        const model = await tf.loadLayersModel(modelPath);

        if (isMounted) {
          setModel(model);
          setIsModelLoaded(true);
        }
      } catch (error) {
        console.error('Failed to load model:', error.message);
        if (isMounted) {
          setError(`Failed to load model: ${error.message}`);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    if (modelPath) {
      loadModel();
    }

    return () => {
      isMounted = false;
    };
  }, [modelPath]);


  // Efficient landmark preprocessing with typed array reuse
  const preprocessHandLandmarks = useCallback((landmarks, handedness) => {
    if (!landmarks || landmarks.length === 0) return null;

    const handLandmarks = landmarks[0];
    const handednessIndex = handedness && handedness[0] && handedness[0][0]
      ? handedness[0][0].index
      : 0;

    // Normalize relative to wrist (landmark 0)
    const baseX = handLandmarks[0].x;
    const baseY = handLandmarks[0].y;
    const baseZ = handLandmarks[0].z;

    // OPTIMIZATION: Reuse typed array instead of creating new arrays
    const dataArray = inputDataArrayRef.current;
    let maxDistance = 0;

    // First pass: normalize and find max distance
    for (let i = 0; i < handLandmarks.length; i++) {
      const landmark = handLandmarks[i];
      const idx = i * 3;

      dataArray[idx] = landmark.x - baseX;
      dataArray[idx + 1] = landmark.y - baseY;
      dataArray[idx + 2] = landmark.z - baseZ;

      const distance = Math.sqrt(
        dataArray[idx] * dataArray[idx] +
        dataArray[idx + 1] * dataArray[idx + 1] +
        dataArray[idx + 2] * dataArray[idx + 2]
      );

      if (distance > maxDistance) {
        maxDistance = distance;
      }
    }

    if (maxDistance === 0) maxDistance = 1;

    // Second pass: scale by max distance
    for (let i = 0; i < 63; i++) { // 21 landmarks * 3 = 63
      dataArray[i] /= maxDistance;
    }

    // Add handedness at the end
    dataArray[63] = handednessIndex;

    return dataArray;
  }, []);

  // Check if gesture is valid for current game mode
  const isValidGestureForMode = useCallback((gestureName, mode) => {
    if (mode === 'rps') {
      return ['Rock', 'Paper', 'Scissors'].includes(gestureName);
    }
    return ['Rock', 'Paper', 'Scissors', 'Lizard', 'Spock'].includes(gestureName);
  }, []);

  // Get valid gesture indices for current game mode
  const getValidGestureIndices = useCallback((mode) => {
    if (mode === 'rps') {
      return [0, 1, 2]; // Rock, Paper, Scissors
    }
    return [0, 1, 2, 3, 4]; // All gestures
  }, []);

  // Reset gesture state
  const resetGestureState = useCallback(() => {
    setDetectedGesture('None');
    setGestureConfidence(0);
    setGestureColor("128, 128, 128");
    currentPredictionRef.current = {
      classIndex: null,
      confidence: 0,
      gestureName: 'None',
      isValidForGameplay: false,
      color: "128, 128, 128"
    };
  }, []);

  // OPTIMIZED: Gesture prediction with better memory management
  const predictGesture = useCallback(async (landmarks, handedness) => {
    if (!model || !isModelLoaded || !landmarks || landmarks.length === 0) {
      resetGestureState();
      return currentPredictionRef.current;
    }

    const currentTime = performance.now();

    // OPTIMIZATION: True rate limiting - skip computation entirely
    if (currentTime - lastPredictionTime.current < predictionInterval) {
      return currentPredictionRef.current;
    }

    lastPredictionTime.current = currentTime;

    try {
      const preprocessed = preprocessHandLandmarks(landmarks, handedness);
      if (!preprocessed) {
        resetGestureState();
        return currentPredictionRef.current;
      }

      // OPTIMIZATION: Use tf.tidy() for automatic tensor cleanup
      const result = tf.tidy(() => {
        // Create input tensor from pre-allocated buffer
        const inputTensor = tf.tensor2d([preprocessed], [1, FEATURES]);

        // Get predictions
        const predictions = model.predict(inputTensor);

        // OPTIMIZATION: Use dataSync() for single values (synchronous, faster)
        const predictionData = predictions.dataSync();

        // Get valid gesture indices for current game mode
        const validIndices = getValidGestureIndices(gameMode);

        // Find max confidence among ONLY valid gestures
        let maxConfidence = -1;
        let classIndex = -1;

        for (const index of validIndices) {
          if (predictionData[index] > maxConfidence) {
            maxConfidence = predictionData[index];
            classIndex = index;
          }
        }

        return { maxConfidence, classIndex };
      });

      let gestureName = 'Uncertain';
      let isValidForGameplay = false;
      let color = "128, 128, 128";

      // Only process if we found a valid gesture above threshold
      if (result.classIndex !== -1 && result.maxConfidence >= CONFIDENCE_THRESHOLD) {
        gestureName = GESTURE_MAP[result.classIndex] || 'Unknown';
        color = GESTURE_COLORS[result.classIndex] || "128, 128, 128";
        isValidForGameplay = true; // We already filtered for valid gestures

        setDetectedGesture(`${gestureName} (${(result.maxConfidence * 100).toFixed(1)}%)`);
        setGestureConfidence(result.maxConfidence);
        setGestureColor(color);
      } else {
        // Either below threshold or no valid gesture detected
        setDetectedGesture(`Uncertain${result.maxConfidence > 0 ? ` (${(result.maxConfidence * 100).toFixed(1)}%)` : ''}`);
        setGestureConfidence(result.maxConfidence);
        setGestureColor("128, 128, 128");
      }

      const newPrediction = {
        classIndex: result.classIndex,
        confidence: result.maxConfidence,
        gestureName,
        isValidForGameplay,
        color
      };

      currentPredictionRef.current = newPrediction;
      return newPrediction;

    } catch (error) {
      console.error('Gesture prediction error:', error);
      const errorPrediction = {
        classIndex: null,
        confidence: 0,
        gestureName: 'Error',
        isValidForGameplay: false,
        color: "128, 128, 128"
      };

      setDetectedGesture('Error');
      setGestureConfidence(0);
      setGestureColor("128, 128, 128");
      currentPredictionRef.current = errorPrediction;
      return errorPrediction;
    }
    // No finally block needed - tf.tidy() handles cleanup automatically
  }, [model, isModelLoaded, preprocessHandLandmarks, gameMode, getValidGestureIndices, resetGestureState]);

  return {
    model,
    isModelLoaded,
    isLoading, // ✅ ADDED: expose loading state for debugging
    error,     // ✅ ADDED: expose error state for debugging
    detectedGesture,
    gestureConfidence,
    gestureColor,
    predictGesture,
    resetGestureState,
    GESTURE_MAP,
    GESTURE_COLORS
  };
};

export default useGestureRecognition;