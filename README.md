<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="00_aisc/img/logo_aisc_bmftr.jpg">
<h1>Rock Paper Scissors AI: Hand Gesture Recognition Game</h1>
</div>

## Overview

This project lets you play Rock Paper Scissors against a computer using real-time hand gesture recognition with AI.

## Features

- **Two Game Modes**: Choose between classic Rock Paper Scissors or the more complex Rock Paper Scissors Lizard Spock variant (known from the Big Bang Theory)
- **Real-Time Hand Detection**: Uses Google's MediaPipe Hands model to accurately track hand and knuckle positions in 3D space
- **Custom Gesture Recognition MLP**: An MLP trained to interpret hand gestures by analyzing the 3D coordinates of knuckles
- **Privacy-Focused**: All processing happens locally in your browser
- **Adjustable Difficulty**: Choose preferred challenge level and number of rounds

## How It Works

The game uses a two-stage AI pipeline to understand your hand gestures:

**Stage 1 - Hand Detection:** Your webcam feed is processed by Google's MediaPipe Hands model, which identifies where your hand appears in the video, crops that part and tracks the 3D position of each knuckle in real-time.

**Stage 2 - Gesture Recognition:** These 3D knuckle coordinates are then fed into a custom-trained neural network that interprets which gesture you're making.

During gameplay, here's what happens behind the scenes:

1. Your webcam continuously captures video frames
2. The MediaPipe model detects your hand and extracts the 3D coordinates of your knuckles
3. These coordinates are analyzed by the gesture recognition model to determine which move you're making
4. Finally, your gesture is compared against the computer's independently chosen gesture to determine the winner


## Setup and Installation

### Prerequisites

- A modern web browser (Chrome, Firefox, or Edge recommended)
- A working webcam

### Quick Start

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Running the Frontend
   ```bash
   cd beatsheldon
   npm install
   npm run dev
   ```
3. (Optional) Training the model
   ```bash
   cd 02_Backend
   ```
   - Run the main.py file with your preferred venv tool




## User Guide

### Getting Started

When you first open the game, two AI models load in your browser. This might take a few moments, but once loaded, everything runs locally and smoothly.

### Choosing Your Game Mode

**Standard Rock Paper Scissors:**
If you're new to the game, the classic rules are simple:
- Rock beats Scissors
- Scissors beats Paper  
- Paper beats Rock

**Rock Paper Scissors Lizard Spock:**
This variant adds two more gestures, making the game more strategic. The complete rules can be found in the provided diagram, or you can watch the video tutorial linked in the game for a detailed explanation of how each gesture interacts.

### Playing the Game

1. Select your preferred game mode from the menu
2. Choose your difficulty level and decide how many rounds you want to play
3. When ready, press "Start Game"
4. During the 3-2-1 countdown, position your hand clearly in front of your webcam
5. Hold your chosen gesture still during the countdown - this gives the neural network time to confidently recognize your move
6. The computer plays fairly in normal mode and doesn't peek at your gesture before making its choice

### Tips for Best Results

**Camera Position:** Make sure your hand is well-lit and clearly visible against a contrasting background. The neural network works best when it can clearly see your hand's outline and finger positions.

**Gesture Clarity:** Hold your gesture firmly and distinctly. Transitioning between gestures or holding ambiguous hand positions can confuse the recognition model.

**Stay Still:** During the countdown, keep your hand as steady as possible. This helps the model get multiple consistent readings and make a confident prediction.

**Calibration:** Try the interactive tutorial mode first to see how the neural network interprets your specific way of making each gesture. Everyone's hands are slightly different, and this helps you understand what works best with your particular hand shape and positioning.

## Technical Details

### Neural Network Architecture

The system employs two distinct neural networks working in sequence:

The **Hand Detection Model** (MediaPipe Hands) identifies hand landmarks by processing video frames through a convolutional neural network that has been trained on diverse hand images. It outputs 21 3D coordinate points representing key locations on your hand.

The **Gesture Recognition Model** takes these 21 coordinate points as input and processes them through fully connected layers to classify which gesture you're making. This model was custom-trained for this specific application.

### Privacy and Security

All processing happens client-side in your browser. Your webcam feed is analyzed frame-by-frame locally, and no video data is transmitted to external servers. The neural networks download once when you first load the page, then run entirely on your device.

## Limitations

- **Lighting Conditions**: The hand detection model performs best in well-lit environments. Very dim lighting or harsh backlighting can reduce accuracy
- **Hand Visibility**: The system requires your hand to be clearly visible to the webcam. Partial occlusion or hands too close/far from the camera may cause recognition issues
- **Gesture Ambiguity**: Some hand positions fall between standard gestures and may be inconsistently classified. The tutorial mode helps you find the clearest way to form each gesture
- **Browser Compatibility**: While most modern browsers are supported, some older browsers may not have the necessary WebGL support for running the neural networks efficiently
- **Single Player Only**: Currently, the game is designed for one player against the computer. Multiplayer modes are not yet implemented

## References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html) - Technical details on the hand detection model
- [Rock Paper Scissors Lizard Spock Rules](https://bigbangtheory.fandom.com/wiki/Rock,_Paper,_Scissors,_Lizard,_Spock) - Complete explanation of the extended game rules
- [HPI AISC](https://hpi.de/kisz)

## Author

Developed as part of research at the HPI Artificial Intelligence & Sustainability Center. For questions or contributions, please visit the project repository.

## License

GPLv3

---

## Acknowledgements

This project was made possible through support from:

<img src="00_aisc/img/logo_bmftr_de.png" alt="BMFTR Logo" style="width:170px;"/>

Special thanks to Google's MediaPipe team for their excellent hand tracking model, and to the open-source community for the tools and frameworks that made this project possible.
