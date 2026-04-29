import { useState } from 'react'
import { HashRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'
import HomePage from "./components/Homepage.jsx"
import HowToPlay from "./components/HowToPlay.jsx";
import Tutorial from "./components/Tutorial.jsx";
import Difficulty from "./components/Difficulty.jsx";
import Rounds from "./components/Rounds.jsx";
import Game from "./components/Game.jsx";
import InfoPage from "./components/InfoPage.jsx";
import ImprintPage from "./components/ImprintPage.jsx";
import SnowfallEffect from "./components/SnowFallEffect.jsx";

function App() {


    const [gameMode, setGameMode] = useState(() => {
        return localStorage.getItem('gameMode') || 'null';
    });

    const [difficulty, setDifficulty] = useState(() => {
        return localStorage.getItem('difficulty') || null;
    });

    const [totalRounds, setTotalRounds] = useState(() => {
        const saved = localStorage.getItem('totalRounds');
        return saved ? parseInt(saved) : null;
    });


    return (
        <Router>
            <SnowfallEffect />
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route
                    path="/howtoplay"
                    element={<HowToPlay gameMode={gameMode} setGameMode={setGameMode} />}
                />
                <Route
                    path="/tutorial"
                    element={<Tutorial gameMode={gameMode} />}
                />
                <Route
                    path="/difficulty"
                    element={
                        <Difficulty
                            difficulty={difficulty}
                            setDifficulty={setDifficulty}
                        />
                    }
                />
                <Route
                    path="/rounds"
                    element={
                        <Rounds
                            gameMode={gameMode}
                            totalRounds={totalRounds}
                            setTotalRounds={setTotalRounds}
                        />
                    }
                />
                <Route
                    path="/game"
                    element={
                        <Game
                            gameMode={gameMode}
                            totalRounds={totalRounds}
                            difficulty={difficulty}
                        />
                    }
                />
                <Route path="/info" element={<InfoPage />} />
                <Route path="/imprint" element={<ImprintPage />} />
            </Routes>
        </Router>
    )
}

export default App