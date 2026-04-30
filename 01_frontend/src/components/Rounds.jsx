import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import '../App.css'
import logoHPI from '../assets/images/combologo.png';
import InfoButton from './InfoButton'
import Logo from "./Logo.jsx";

function Rounds({ gameMode, totalRounds, setTotalRounds }) {
    const navigate = useNavigate()

    const handleRoundsSelect = (rounds) => {
        setTotalRounds(rounds);
        localStorage.setItem('totalRounds', rounds.toString());
    }

    const startGame = () => {
        navigate('/game')
    }

    const roundOptions = [
        { value: 3, label: '3' },
        { value: 5, label: '5' },
        { value: 7, label: '7' }
    ];

    return (
        <div className="h-dvh w-full flex flex-col items-center justify-center overflow-hidden">
            <InfoButton />
            <Logo/>

            {/* Title Section */}
            <div className="pt-[3vh] pb-[2vh]">
                <h1 className="text-center font-bold text-main text-[4vw] leading-tight">
                    Rounds
                </h1>
            </div>

            {/* Description */}
            <div className="mb-[3vh]">
                <p className="text-center font-medium text-gray-600 text-[2vmin] leading-relaxed">
                    Choose how many rounds to play:
                </p>
            </div>

            {/* Round Options */}
            <div className="grid grid-cols-3 gap-[2vmin] mb-[3vh]">
                {roundOptions.map((option) => (
                    <button
                        key={option.value}
                        onClick={() => handleRoundsSelect(option.value)}
                        className={`font-medium border-2 border-main transition-colors duration-250 cursor-pointer
                       px-[3vmin] py-[1vmin] text-[2.5vmin]
                       ${totalRounds === option.value
                            ? 'bg-main text-white'
                            : 'bg-white text-main hover:bg-main hover:text-white'
                        }`}
                    >
                        {option.label}
                    </button>
                ))}
            </div>

            {/* Start Game Button */}
            <div className="flex flex-col items-center">
                <button
                    onClick={startGame}
                    disabled={!totalRounds}
                    className={`font-medium border-2 border-main bg-white text-main hover:bg-main hover:text-white transition-all duration-250 cursor-pointer
                          px-[3vmin] py-[1vmin] text-[2.5vmin]
                          ${totalRounds ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                >
                    Start Game
                </button>
            </div>
        </div>
    )
}

export default Rounds