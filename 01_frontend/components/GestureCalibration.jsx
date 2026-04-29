import React, { useState, useEffect, useRef } from 'react';

import rockIcon from '../assets/images/rock.svg';
import paperIcon from '../assets/images/paper.svg';
import scissorsIcon from '../assets/images/scissors.svg';
import lizardIcon from '../assets/images/lizard.svg';
import spockIcon from '../assets/images/spock.svg';
import noneIcon from '../assets/images/none.svg';
import {images} from "../config/themeConfig.js";

const GestureCalibration = ({
                                currentGesture,
                                gestureConfidence,
                                gameMode = 'rps',
                                onCalibrationComplete
                            }) => {

    const gestureIcons = images();

    const getGestures = () => {
        if (gameMode === 'rps') {
            return ['Rock', 'Paper', 'Scissors'];
        }
        return ['Rock', 'Paper', 'Scissors', 'Lizard', 'Spock'];
    };

    const getGestureIcon = (gesture) => {
        const handleImageError = (e) => {
            console.error(`Failed to load image for gesture: ${gesture}`);
            console.error('Image src was:', e.target.src);
            e.target.style.display = 'none';
        };

        if (!gesture) {
            return (
                <img
                    src={noneIcon}
                    alt="none"
                    className="w-[25vmin] h-[25vmin] opacity-70"
                    onError={handleImageError}
                />
            );
        }

        const iconSrc = gestureIcons[gesture.toLowerCase()] || noneIcon;
        console.log(`Loading icon for ${gesture}:`, iconSrc); // Debug log

        return (
            <img
                src={iconSrc}
                alt={gesture}
                className="w-[25vmin] h-[25vmin]"
                onError={handleImageError}
                onLoad={() => console.log(`Successfully loaded ${gesture} icon`)} // Debug log
            />
        );
    };

    const gestures = getGestures();
    const [currentGestureIndex, setCurrentGestureIndex] = useState(0);
    const [holdProgress, setHoldProgress] = useState(0);
    const [isCalibrationComplete, setIsCalibrationComplete] = useState(false);
    const [calibratedGestures, setCalibratedGestures] = useState(new Set());
    const [holdState, setHoldState] = useState('waiting'); // 'waiting', 'holding', 'verifying'

    const holdStartTime = useRef(null);
    const initialGesture = useRef(null);
    const progressInterval = useRef(null);
    const verificationTimer = useRef(null);

    const HOLD_DURATION = 1000; // 1 second
    const MIN_CONFIDENCE = 0.3;

    const currentTargetGesture = gestures[currentGestureIndex];

    const extractGestureName = (gestureString) => {
        if (!gestureString || gestureString === 'None' || gestureString === 'Uncertain') {
            return null;
        }
        return gestureString.split(' (')[0];
    };

    const isValidGesture = () => {
        const gestureName = extractGestureName(currentGesture);
        return gestureName === currentTargetGesture && gestureConfidence >= MIN_CONFIDENCE;
    };

    const cleanupTimers = () => {
        if (progressInterval.current) {
            clearInterval(progressInterval.current);
            progressInterval.current = null;
        }
        if (verificationTimer.current) {
            clearTimeout(verificationTimer.current);
            verificationTimer.current = null;
        }
    };

    const resetHoldState = () => {
        cleanupTimers();
        setHoldState('waiting');
        setHoldProgress(0);
        holdStartTime.current = null;
        initialGesture.current = null;
    };

    useEffect(() => {
        if (calibratedGestures.has(currentTargetGesture)) {
            return;
        }

        const isValid = isValidGesture();
        const currentGestureName = extractGestureName(currentGesture);

        switch (holdState) {
            case 'waiting':
                if (isValid) {
                    console.log(`Starting hold for ${currentTargetGesture}`);
                    setHoldState('holding');
                    holdStartTime.current = Date.now();
                    initialGesture.current = currentGestureName;

                    progressInterval.current = setInterval(() => {
                        if (holdStartTime.current) {
                            const elapsed = Date.now() - holdStartTime.current;
                            const progress = Math.min(elapsed / HOLD_DURATION, 1);
                            setHoldProgress(progress);
                        }
                    }, 16);

                    verificationTimer.current = setTimeout(() => {
                        console.log(`Verifying gesture after hold period`);
                        setHoldState('verifying');
                    }, HOLD_DURATION);
                }
                break;

            case 'holding':
                if (!isValid || currentGestureName !== initialGesture.current) {
                    console.log(`Hold interrupted for ${currentTargetGesture}`);
                    resetHoldState();
                }
                break;

            case 'verifying':
                if (isValid && currentGestureName === initialGesture.current) {
                    setCalibratedGestures(prev => new Set([...prev, currentTargetGesture]));

                    if (currentGestureIndex < gestures.length - 1) {
                        setCurrentGestureIndex(prev => prev + 1);
                        resetHoldState();
                    } else {
                        setIsCalibrationComplete(true);
                        cleanupTimers();
                    }
                } else {
                    console.log(`Verification failed for ${currentTargetGesture}`);
                    resetHoldState();
                }
                break;
        }
    }, [currentGesture, gestureConfidence, holdState, currentTargetGesture, calibratedGestures, currentGestureIndex, gestures.length]);

    useEffect(() => {
        return () => {
            cleanupTimers();
        };
    }, []);

    const getGestureStatus = (gesture) => {
        if (calibratedGestures.has(gesture)) {
            return 'completed';
        } else if (gesture === currentTargetGesture) {
            return 'current';
        }
        return 'pending';
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'completed':
                return 'text-main border-main bg-main';
            case 'current':
                return 'text-main border-main bg-white';
            default:
                return 'text-gray-400 border-gray-300 bg-white';
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return '✓';
            case 'current':
                return '→';
            default:
                return '';
        }
    };

    const getHoldStateMessage = () => {
        switch (holdState) {
            case 'holding':
                return 'Hold steady...';
            case 'verifying':
                return 'Verifying...';
            default:
                return 'Show the gesture above';
        }
    };

    if (isCalibrationComplete) {
        return (
            <div className="flex flex-col items-center justify-center h-full">
                <div className="text-center mb-[3vh]">
                    <div className="text-[12vmin] mb-[2vh] text-main">✓</div>
                    <h2 className="text-[4vmin] font-bold text-main mb-[2vh]">
                        Tutorial Complete!
                    </h2>
                </div>

                <button
                    onClick={onCalibrationComplete}
                    className="px-[3vmin] py-[2vmin] text-[2.5vmin] font-medium border-2 border-main bg-white text-main hover:bg-main hover:text-white transition-colors duration-250 cursor-pointer"
                >
                    Continue
                </button>
            </div>
        );
    }

    return (
        <div className="flex flex-col items-center justify-center h-full">
            <h1 className="text-[6vmin] font-bold mb-[1vh] mt-[3vh]" onClick={onCalibrationComplete}>Tutorial</h1>

            <div className="text-center mb-[3vh]">
                <p className="text-[2vmin] text-gray-600 mb-[1vh]">
                    Make and hold each gesture for 1 second:
                </p>
            </div>

            <div className="mb-[2vh] flex flex-col items-center">
                <div className="flex justify-center">
                    {getGestureIcon(currentTargetGesture)}
                </div>
            </div>

            <div className="w-full max-w-[20vw] mb-[2vh]">
                <div className="mb-[2vh]">
                    <div className="w-full h-[1vh] bg-white overflow-hidden mb-[1vh]">
                        <div
                            className="h-full transition-all duration-75 ease-out bg-main"
                            style={{ width: `${holdProgress * 100}%` }}
                        />
                    </div>
                    <div className="text-[2vmin] italic text-center">
                        {getHoldStateMessage()}
                    </div>
                </div>

                <div className="bg-white">
                    <div className="space-y-[1vh]">
                        {gestures.map((gesture) => {
                            const status = getGestureStatus(gesture);
                            const colorClass = getStatusColor(status);
                            const isCompleted = status === 'completed';

                            return (
                                <div
                                    key={gesture}
                                    className={`flex items-center justify-between p-[1vmin] border-2 transition-colors duration-250 ${colorClass} ${isCompleted ? 'text-white' : ''}`}
                                >
                                    <span className="font-medium text-[2vmin]">{gesture}</span>
                                    <span className="text-[2.5vmin] font-bold">{getStatusIcon(status)}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GestureCalibration;