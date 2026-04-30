import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import HandTrackingWebcam from './HandTrackingWebcam'
import useGameLogic from './useGameLogic'
import '../App.css'
import InfoButton from './InfoButton'

import noneIcon from '../assets/images/none.svg';

// Import tsParticles confetti
import { confetti } from '@tsparticles/confetti';
import Logo from "./Logo.jsx";
import {images} from "../config/themeConfig.js";



const gestureIcons = images();

function Game({ gameMode = 'rps', totalRounds = 3, difficulty = 'normal' }) {
    console.log('Game component mounted with:', { gameMode, totalRounds })
    const navigate = useNavigate()
    const [detectedGesture, setDetectedGesture] = useState('None')
    const [gestureConfidence, setGestureConfidence] = useState(0)
    const [progressPercent, setProgressPercent] = useState(0)
    const [roundResults, setRoundResults] = useState([]) // Track results for each round

    // New state for overlay effect
    const [overlayOpacity, setOverlayOpacity] = useState(0)
    const [overlayColor, setOverlayColor] = useState('transparent')

    // Initialize game logic
    const {
        gameState,
        currentRound,
        playerScore,
        computerScore,
        countdown,
        playerChoice,
        computerChoice,
        roundWinner,
        startGame,
        resetGame,
        processPlayerGesture,
        getResultMessage,
        getFinalGameMessage,
        getGameWinner,
        cleanup
    } = useGameLogic(gameMode, totalRounds, difficulty);

    // Function to trigger confetti with custom SVG icons
    const triggerConfetti = () => {
        // Get available gesture icons (excluding 'none')
        const availableIcons = Object.entries(gestureIcons)
            .filter(([key]) => key !== 'none')
            .map(([key, icon]) => ({
                src: icon,
                width: 64,
                height: 64,
                particles: {
                    size: {
                        value: 32
                    }
                }
            }));

        // Create multiple confetti bursts from different positions
        const positions = [
            { x: 25, y: 25 }, // Top left
            { x: 75, y: 25 }, // Top right
            { x: 50, y: 50 }, // Center
        ];

        positions.forEach((position, index) => {
            setTimeout(() => {
                // Use the full tsParticles configuration for custom images
                confetti({
                    particleCount: 90,
                    angle: 90,
                    spread: 80,
                    startVelocity: 60,
                    decay: 0.9,
                    gravity: 1.2,
                    drift: Math.random() * 2 - 1, // Random drift between -1 and 1
                    ticks: 300,
                    position: position,
                    zIndex: 1000,
                    disableForReducedMotion: true,
                    // Use a mix of shapes and custom images
                    shapes: ['circle', 'square'],
                    colors: ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'],
                    scalar: 0.8,
                });

                // Add a second burst with images (if browser supports it)
                setTimeout(() => {
                    try {
                        confetti({
                            particleCount: 15,
                            angle: 90 + (Math.random() * 30 - 15), // Slight angle variation
                            spread: 45,
                            startVelocity: 50,
                            decay: 0.95,
                            gravity: 1,
                            ticks: 250,
                            position: {
                                x: position.x + (Math.random() * 20 - 10),
                                y: position.y + (Math.random() * 10 - 5)
                            },
                            zIndex: 1000,
                            shapes: ['star', 'circle'],
                            colors: ['#ffd700', '#ff6b6b', '#4ecdc4', '#45b7d1'],
                            scalar: 1.2,
                        });
                    } catch (error) {
                        console.log('Image confetti not supported, using shapes only');
                    }
                }, 200);
            }, index * 150); // Stagger the bursts
        });

        // Add a final celebratory burst from the bottom
        setTimeout(() => {
            confetti({
                particleCount: 100,
                angle: 90,
                spread: 120,
                startVelocity: 80,
                decay: 0.9,
                gravity: 1.5,
                ticks: 400,
                position: { x: 50, y: 100 },
                zIndex: 1000,
                shapes: ['circle', 'square', 'triangle'],
                colors: ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'],
                scalar: 1,
            });
        }, 800);
    };

    // Trigger confetti when game finishes and player wins
    useEffect(() => {
        if (gameState === 'finished') {
            const gameWinner = getGameWinner();
            if (gameWinner === 'player') {
                // Small delay to let the game over screen appear first
                const confettiTimer = setTimeout(() => {
                    triggerConfetti();
                }, 500);

                return () => clearTimeout(confettiTimer);
            }
        }
    }, [gameState, getGameWinner]);

    // New effect for overlay animation
    useEffect(() => {
        if (gameState === 'result' && roundWinner && roundWinner !== 'draw') {
            // Set overlay color based on result (solid colors, opacity handled separately)
            const color = roundWinner === 'player' ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)';
            setOverlayColor(color);

            // Start with 0 opacity, then fade in after small delay
            setOverlayOpacity(0);

            const fadeInTimer = setTimeout(() => {
                setOverlayOpacity(0.3);
            }, 50); // Small delay to ensure transition triggers

            // Start fade out after fade in (400ms) + stay duration (2000ms)
            const fadeOutTimer = setTimeout(() => {
                setOverlayOpacity(0);
            }, 1050); // 400ms fade in + 2000ms stay + 50ms delay

            return () => {
                clearTimeout(fadeInTimer);
                clearTimeout(fadeOutTimer);
            };
        } else {
            // Reset overlay for other states
            setOverlayOpacity(0);
            setOverlayColor('transparent');
        }
    }, [gameState, roundWinner]);

    // Update round results when a round completes
    useEffect(() => {
        if (gameState === 'result' && roundWinner !== null) {
            setRoundResults(prev => {
                const newResults = [...prev];
                // Ensure we don't add duplicate results for the same round
                if (newResults.length < currentRound) {
                    newResults.push(roundWinner);
                }
                return newResults;
            });
        }
    }, [gameState, roundWinner, currentRound]);

    // Reset round results when game resets
    useEffect(() => {
        if (gameState === 'waiting') {
            setRoundResults([]);
        }
    }, [gameState]);

    useEffect(() => {
        if (gameState === 'waiting') {
            setProgressPercent(100);
        } else if (gameState === 'countdown') {
            // Only animate if we're starting from 100%
            let targetPercent;
            if (countdown === 3) {
                targetPercent = 66;
            } else if (countdown === 2) {
                targetPercent = 33;
            } else if (countdown === 1) {
                targetPercent = 0;
            }

            const timer = setTimeout(() => {
                setProgressPercent(targetPercent);
            }, 100);
            return () => clearTimeout(timer);
        } else if (gameState === 'result') {
            // Keep at current position during result
        } else if (gameState === 'break') {
            // IMMEDIATELY set to 100% when break starts - no animation needed
            setProgressPercent(100);
        } else if (gameState === 'playing' || gameState === 'reveal') {
            setProgressPercent(0);
        }
    }, [gameState, countdown]);


    const handleGestureDetected = (gestureData) => {
        if (gestureData.gesture && gestureData.gesture.name) {
            const gestureName = gestureData.gesture.name.split(' (')[0]; // Remove confidence percentage
            setDetectedGesture(gestureData.gesture.name);
            setGestureConfidence(gestureData.gesture.confidence);

            // Process the gesture for gameplay
            processPlayerGesture(gestureName, gestureData.gesture.confidence);
        } else {
            setDetectedGesture('None');
            setGestureConfidence(0);
        }
    }

    // Auto-start game when component mounts
    useEffect(() => {
        if (gameState === 'waiting') {
            const timer = setTimeout(() => {
                startGame();
            }, 2000); // 1 second delay before starting

            return () => clearTimeout(timer);
        }
    }, [gameState, startGame]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            cleanup();
        };
    }, [cleanup]);

    // Get gesture icon helper
    const getGestureIcon = (gesture) => {
        const handleImageError = (e) => {
            console.error(`Failed to load image for gesture: ${gesture}`);
            e.target.style.display = 'none';
        };

        if (!gesture) {
            return (
                <img
                    src={noneIcon}
                    alt="none"
                    className="w-[50vmin] h-[50vmin] opacity-70 transition-all duration-500"
                    onError={handleImageError}
                />
            );
        }

        const iconSrc = gestureIcons[gesture.toLowerCase()] || noneIcon;
        return (
            <img
                src={iconSrc}
                alt={gesture}
                className="w-[50vmin] h-[50vmin] transition-all duration-500"
                onError={handleImageError}
            />
        );
    };

    const renderSquareScore = () => {
        const squares = [];

        for (let i = 0; i < totalRounds; i++) {
            let squareColor = '#6b7280'; // Default grey
            let squareOpacity = 0.3; // Default opacity for unfilled squares
            let isUnfilled = i >= roundResults.length;

            if (i < roundResults.length) {
                squareOpacity = 1; // Full opacity for completed rounds
                switch (roundResults[i]) {
                    case 'player':
                        squareColor = '#22c55e'; // Green for player win
                        break;
                    case 'computer':
                        squareColor = '#ef4444'; // Red for computer win
                        break;
                    case 'tie':
                        squareColor = '#6b7280'; // Grey for tie
                        break;
                }
            }

            squares.push(
                <div
                    key={i}
                    className={`w-[8vmin] h-[8vmin] transition-all duration-300 ${
                        isUnfilled ? 'border-2 border-gray-500 bg-white' : ''
                    }`}
                    style={{
                        backgroundColor: isUnfilled ? undefined : squareColor,
                        opacity: squareOpacity
                    }}
                />
            );
        }

        return (
            <div className="flex space-x-[2vmin] overflow-hidden">
                {squares}
            </div>
        );
    };

    // Render game state specific content
    const renderGameContent = () => {
        switch (gameState) {
            case 'waiting':
                return (
                    <div className="text-center">
                        <h2 className="text-[4vmin] font-bold text-main mb-[1vh]">Pick your next Move!</h2>
                    </div>
                );

            case 'countdown':
                return (
                    <div className="text-center">
                        <h1 className="text-[4vmin] font-bold text-main mb-[1vh]">Show your move now!</h1>
                        <div className="text-[12vmin] font-bold text-main mb-[1vh]">{countdown}</div>
                    </div>
                );

            case 'playing':
                return (
                    <div className="text-center">
                        <h2 className="text-[3vmin] font-bold text-main mb-[1vh]">Round {currentRound}</h2>
                        <div className="text-[8vmin] font-bold text-main mb-[1vh]">GO!</div>
                    </div>
                );

            case 'result':
                return (
                    <div className="text-center">
                        <h2 className="text-[3vmin] font-bold text-main mb-[1vh]">Round {currentRound}</h2>
                        <div className="text-center mb-[3vh]">
                            <p className="text-[2.5vmin] font-bold mb-[2vh]">Computer</p>
                            <div className="flex items-center"> {/* Add container for centering */}
                                {getGestureIcon(computerChoice)}
                            </div>
                        </div>
                        {/*
            <div className="text-[2.5vmin] font-bold mb-[1vh] mt-[2vh] z-10" style={{
              color: roundWinner === 'player' ? '#22c55e' :
                    roundWinner === 'computer' ? '#ef4444' : '#6b7280'
            }}>
              {getResultMessage()}
            </div>
            */}
                    </div>
                );

            case 'break':
                return (
                    <div className="text-center">
                        <h2 className="text-[4vmin] font-bold text-main mb-[1vh]">Pick your next Move!</h2>
                    </div>
                );

            case 'finished':
                const gameWinner = getGameWinner();
                return (
                    <div className="text-center">
                        <h2 className="text-[6vmin] font-bold text-main mb-[2vh]">Game Over!</h2>
                        {gameWinner === 'player' && (
                            <div className="text-[3vmin] font-bold text-green-500 mb-[1vh]"> You Won!</div>
                        )}
                        {gameWinner === 'computer' && (
                            <div className="text-[3vmin] font-bold text-red-500 mb-[1vh]">Computer Wins!</div>
                        )}
                        {gameWinner === 'tie' && (
                            <div className="text-[3vmin] font-bold text-gray-500 mb-[1vh]">It's a Tie!</div>
                        )}
                        <div className="flex text-[2vmin] mb-[2.2vh] mt-[2.2vh] items-center justify-center">
                            {renderSquareScore()}
                        </div>
                        <div className="flex flex-row items-center space-x-[1vh]">
                            <button
                                onClick={() => navigate('/rounds')}
                                className="min-w-[16vmin] max-w-[25vmin] px-[3vmin] py-[2vmin] text-[2.5vmin] font-medium border-2 border-main bg-transparent text-main hover:bg-main hover:text-white transition-colors duration-250 cursor-pointer"
                            >
                                Play Again
                            </button>
                            <button
                                onClick={() => navigate('/')}
                                className="min-w-[16vmin] max-w-[25vmin] px-[3vmin] py-[2vmin] text-[2.5vmin] font-medium border-2 border-main bg-transparent text-main hover:bg-main hover:text-white transition-colors duration-250 cursor-pointer"
                            >
                                Main Menu
                            </button>
                        </div>
                    </div>
                );

            default:
                return (
                    <div className="text-center">
                        <h2 className="text-[4vmin] font-bold text-main mb-[1vh]">Game</h2>
                    </div>
                );
        }
    };

    return (
        <div className="w-full h-dvh relative">
            <div className="flex w-full h-full">
                <InfoButton />
                <Logo/>
                {/* Left Column: Game Content */}
                <div className="w-1/2 h-full items-center font-bold justify-center flex flex-col p-[1vmin]">
                    {renderGameContent()}

                    {/* Square Score Display - Always visible during active game */}
                    {gameState !== 'waiting' && gameState !== 'finished' && (
                        <div className="absolute bottom-[3vh] w-full flex justify-center bg-opacity-75 p-[0.5vmin] z-20">
                            <div className="flex flex-col items-center space-y-[1vh]">
                                {renderSquareScore()}
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column: Webcam Display */}
                <div className="w-1/2 h-full relative">
                    <HandTrackingWebcam
                        onGestureDetected={handleGestureDetected}
                        showLandmarks={true}
                        mirrored={true}
                        fillContainer={true}
                        gameMode={gameMode}
                        enableRegionCropping={true}
                    />

                    {/* Player gesture overlay on webcam during result */}
                    {gameState === 'result' && playerChoice && (
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                            <div className="opacity-80 transition-all duration-500 ">
                                {getGestureIcon(playerChoice)}
                            </div>
                        </div>
                    )}

                    {/* Overlay for win/lose effect */}
                    <div
                        className="absolute inset-0 pointer-events-none"
                        style={{
                            background: `linear-gradient(to left, transparent 55%, ${overlayColor} 90%)`,
                            opacity: overlayOpacity,
                            transition: overlayOpacity > 0 ? 'opacity 0.3s ease-in' : 'opacity 0.9s ease-out'
                        }}
                    />
                </div>

            </div>

            {/* Progress Bar */}
            <div
                className="absolute top-0 left-1/2 transform -translate-x-1/2 h-full z-10 w-[2vmin]"
            >
                <div
                    className="w-full bg-main origin-bottom"
                    style={{
                        height: '100%',
                        backgroundColor: '#1d1d1f',
                        transform: `scaleY(${progressPercent / 100})`,
                        transition: gameState === 'break'
                            ? 'transform 1.9s ease-out'  // Match the break duration (3s minus 100ms delay)
                            : gameState === 'countdown'
                                ? 'transform 1s ease-out'  // Smooth countdown animation
                                : gameState === 'result'
                                    ? 'none'                     // No transition during result
                                    : 'transform 1s ease-out'
                    }}
                />
            </div>
        </div>
    )
}

export default Game