import { useState, useEffect, useRef, useCallback } from 'react';

const CONFIDENCE_THRESHOLD = 0.3;

// Game configuration constants
const GAME_OPTIONS = {
    rps: {
        gestures: ['Rock', 'Paper', 'Scissors'],
        rules: {
            Rock: { beats: 'Scissors' },
            Paper: { beats: 'Rock' },
            Scissors: { beats: 'Paper' }
        }
    },
    rpls: {
        gestures: ['Rock', 'Paper', 'Scissors', 'Lizard', 'Spock'],
        rules: {
            Rock: { beats: 'Scissors', beats2: 'Lizard' },
            Paper: { beats: 'Rock', beats2: 'Spock' },
            Scissors: { beats: 'Paper', beats2: 'Lizard' },
            Lizard: { beats: 'Paper', beats2: 'Spock' },
            Spock: { beats: 'Rock', beats2: 'Scissors' }
        }
    }
};

const FIRST_ROUND_EASINESS = 0.8;

// Game utility functions
const getComputerChoice = (mode) => {
    const choices = GAME_OPTIONS[mode].gestures;
    return choices[Math.floor(Math.random() * choices.length)];
};

// Get computer choice that beats the player (for easy mode)
const getWinningChoice = (playerChoice, mode) => {
    const rules = GAME_OPTIONS[mode].rules;
    const gestures = GAME_OPTIONS[mode].gestures;

    // Find a gesture that beats the player's choice
    for (const gesture of gestures) {
        if (rules[gesture]?.beats === playerChoice || rules[gesture]?.beats2 === playerChoice) {
            return gesture;
        }
    }

    // Fallback to random if no winning choice found
    return getComputerChoice(mode);
};

// Smart computer choice for easy mode
const getSmartComputerChoice = (playerChoice, mode, currentRound, playerScore, computerScore, totalRounds, difficulty) => {
    if (difficulty !== 'easy') {
        return getComputerChoice(mode);
    }

    if (difficulty === 'normal' && currentRound === 1) {
        if (Math.random() < FIRST_ROUND_EASINESS) {
            const rules = GAME_OPTIONS[mode].rules;
            const gestures = GAME_OPTIONS[mode].gestures;

            const losingChoices = gestures.filter(gesture => {
                const playerRule = rules[playerChoice];
                return playerRule?.beats === gesture || playerRule?.beats2 === gesture;
            });

            if (losingChoices.length > 0) {
                return losingChoices[Math.floor(Math.random() * losingChoices.length)];
            }
        }
        return getComputerChoice(mode);
    }

    const roundsNeededToWin = Math.ceil(totalRounds / 2);
    const remainingRounds = totalRounds - currentRound + 1;
    const roundsComputerCanAffordToLose = remainingRounds - (roundsNeededToWin - computerScore);

    // If computer is already winning enough, occasionally let player win
    if (computerScore >= roundsNeededToWin) {
        return getComputerChoice(mode);
    }

    // If computer needs to win this round to stay in the game
    if (roundsComputerCanAffordToLose <= 0) {
        return getWinningChoice(playerChoice, mode);
    }

    // Strategic decisions based on game state
    const isEarlyGame = currentRound <= Math.ceil(totalRounds / 3);
    const isLateGame = currentRound >= totalRounds - 1;

    // Early game: occasionally let player win to seem fair (30% chance)
    if (isEarlyGame && Math.random() < 0.3) {
        return getComputerChoice(mode);
    }

    // Late game: more aggressive if computer is behind
    if (isLateGame && computerScore < playerScore) {
        return getWinningChoice(playerChoice, mode);
    }

    // Default strategy: 70% chance to win, 30% random
    if (Math.random() < 0.7) {
        return getWinningChoice(playerChoice, mode);
    } else {
        return getComputerChoice(mode);
    }
};

const determineWinner = (player, computer, mode) => {
    const rules = GAME_OPTIONS[mode].rules;
    if (player === computer || player === 'None') return 'draw';
    if (rules[player]?.beats === computer || rules[player]?.beats2 === computer) {
        return 'player';
    }
    return 'computer';
};

const isValidGestureForMode = (gestureName, gameMode) => {
    if (!gameMode || !gestureName || gestureName === 'None' || gestureName === 'Uncertain') {
        return false;
    }
    const validGestures = GAME_OPTIONS[gameMode]?.gestures || [];
    return validGestures.includes(gestureName);
};

// Find majority gesture from captured gestures
const findMajorityGesture = (gestures) => {
    if (gestures.length === 0) return 'None';
    const counts = gestures.reduce((acc, gesture) => {
        acc[gesture] = (acc[gesture] || 0) + 1;
        return acc;
    }, {});
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
};

const useGameLogic = (gameMode = 'rps', totalRounds = 3, difficulty = 'normal') => {
    // Game state
    const [gameState, setGameState] = useState('waiting');
    const [currentRound, setCurrentRound] = useState(1);
    const [playerScore, setPlayerScore] = useState(0);
    const [computerScore, setComputerScore] = useState(0);
    const [countdown, setCountdown] = useState(3);
    const [playerChoice, setPlayerChoice] = useState(null);
    const [computerChoice, setComputerChoice] = useState(null);
    const [roundWinner, setRoundWinner] = useState(null);

    // Refs for managing timers and gesture capture
    const gameTimer = useRef(null);
    const capturedGestures = useRef([]);
    const currentGestureRef = useRef('None');

    // Game timing constants
    const COUNTDOWN_DURATION = 1000; // ms per countdown number
    const RESULT_DURATION = 3000; // ms to show result
    const BREAK_DURATION = 2200;

    // Clean up timers
    const cleanup = useCallback(() => {
        if (gameTimer.current) {
            clearTimeout(gameTimer.current);
            gameTimer.current = null;
        }
    }, []);

    // Reset game to initial state
    const resetGame = useCallback(() => {
        cleanup();
        setGameState('waiting');
        setCurrentRound(1);
        setPlayerScore(0);
        setComputerScore(0);
        setCountdown(3);
        setPlayerChoice(null);
        setComputerChoice(null);
        setRoundWinner(null);
        capturedGestures.current = [];
        currentGestureRef.current = 'None';
    }, [cleanup]);

    // Start a new game
    const startGame = useCallback(() => {
        resetGame();
        setGameState('countdown');
        setCurrentRound(1);
        setCountdown(3);
    }, [resetGame]);

    // Process player gesture - store current gesture for capture window
    const processPlayerGesture = useCallback((gestureName, confidence) => {
        // Extract clean gesture name (remove confidence percentage)
        const cleanGestureName = gestureName?.split(' (')[0] || 'None';
        currentGestureRef.current = cleanGestureName;

        // During the final countdown (1), capture valid gestures in the last 200ms
        if (gameState === 'countdown' && countdown === 1) {
            if (isValidGestureForMode(cleanGestureName, gameMode) && confidence >= CONFIDENCE_THRESHOLD) {
                capturedGestures.current.push(cleanGestureName);
            }
        }
    }, [gameState, countdown, gameMode]);

    // Get result message for current round
    const getResultMessage = useCallback(() => {
        if (!roundWinner) return '';

        switch (roundWinner) {
            case 'player':
                return 'You Win!';
            case 'computer':
                return 'You Lose!';
            case 'draw':
                return "It's a Tie!";
            default:
                return '';
        }
    }, [roundWinner]);

    // Get final game winner
    const getGameWinner = useCallback(() => {
        if (playerScore > computerScore) return 'player';
        if (computerScore > playerScore) return 'computer';
        return 'draw';
    }, [playerScore, computerScore]);

    // Get final game message
    const getFinalGameMessage = useCallback(() => {
        const winner = getGameWinner();

        switch (winner) {
            case 'player':
                return 'Congratulations! You Won!';
            case 'computer':
                return 'Game Over! You Lost!';
            case 'draw':
                return "It's a Draw!";
            default:
                return 'Game Complete!';
        }
    }, [getGameWinner]);

    // Main game state machine
    useEffect(() => {
        const clearTimers = () => { if (gameTimer.current) clearTimeout(gameTimer.current); };

        switch (gameState) {
            case 'countdown':
                if (countdown === 3) {
                    gameTimer.current = setTimeout(() => {
                        setCountdown(2);
                    }, COUNTDOWN_DURATION);
                } else if (countdown === 2) {
                    gameTimer.current = setTimeout(() => {
                        setCountdown(1);
                        capturedGestures.current = [];
                    }, COUNTDOWN_DURATION);
                } else if (countdown === 1) {
                    gameTimer.current = setTimeout(() => {
                        // Get player choice first
                        const userChoice = findMajorityGesture(capturedGestures.current);

                        // Then determine computer choice (potentially based on player choice in easy mode)
                        const computerChoice = getSmartComputerChoice(
                            userChoice,
                            gameMode,
                            currentRound,
                            playerScore,
                            computerScore,
                            totalRounds,
                            difficulty
                        );

                        const result = determineWinner(userChoice, computerChoice, gameMode);

                        // Set choices and winner
                        setPlayerChoice(userChoice);
                        setComputerChoice(computerChoice);
                        setRoundWinner(result);

                        // Update scores
                        if (result === 'player') {
                            setPlayerScore(prev => prev + 1);
                        } else if (result === 'computer') {
                            setComputerScore(prev => prev + 1);
                        }

                        setGameState('result');
                    }, COUNTDOWN_DURATION);
                }
                break;

            case 'result':
                gameTimer.current = setTimeout(() => {
                    if (currentRound < totalRounds) {
                        setGameState('break');
                    } else {
                        setGameState('finished');
                    }
                }, RESULT_DURATION);
                break;

            case 'break':
                gameTimer.current = setTimeout(() => {
                    setCurrentRound(prev => prev + 1);
                    setPlayerChoice(null);
                    setComputerChoice(null);
                    setRoundWinner(null);
                    setCountdown(3);
                    capturedGestures.current = [];
                    setGameState('countdown');
                }, BREAK_DURATION);
                break;

            case 'waiting':
            case 'finished':
            default:
                clearTimers();
                break;
        }

        return clearTimers;
    }, [gameState, countdown, currentRound, totalRounds, gameMode, playerScore, computerScore, difficulty]);

    // Clean up on unmount
    useEffect(() => {
        return cleanup;
    }, [cleanup]);

    return {
        // Game state
        gameState,
        currentRound,
        playerScore,
        computerScore,
        countdown,
        playerChoice,
        computerChoice,
        roundWinner,

        // Actions
        startGame,
        resetGame,
        processPlayerGesture,
        cleanup,

        // Computed values
        getResultMessage,
        getFinalGameMessage,
        getGameWinner,

        // Game configuration
        totalRounds,
        gameMode,
        difficulty
    };
};

export default useGameLogic;