import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import '../App.css'
import InfoButton from './InfoButton'
import Logo from "./Logo.jsx";

function Difficulty({ difficulty, setDifficulty }) {
    const navigate = useNavigate()

    const handleDifficultySelect = (selectedDifficulty) => {
        setDifficulty(selectedDifficulty);
        localStorage.setItem('difficulty', selectedDifficulty);
    }

    const proceedToRounds = () => {
        navigate('/rounds')
    }

    const difficultyOptions = [
        { value: 'normal', label: 'Normal' },
        { value: 'easy', label: 'Impossible' }
    ];

    return (
        <div className="h-dvh w-full flex flex-col items-center justify-center overflow-hidden">
            <InfoButton />
            <Logo/>

            {/* Title Section */}
            <div className="pt-[3vh] pb-[2vh]">
                <h1 className="text-center font-bold text-main text-[4vw] leading-tight">
                    Difficulty
                </h1>
            </div>

            {/* Description */}
            <div className="mb-[3vh]">
                <p className="text-center font-medium text-gray-600 text-[2vmin] leading-relaxed">
                    Choose your preferred difficulty:
                </p>
            </div>

            {/* Difficulty Options */}
            <div className="grid grid-cols-2 gap-[3vmin] mb-[3vh]">
                {difficultyOptions.map((option) => (
                    <button
                        key={option.value}
                        onClick={() => handleDifficultySelect(option.value)}
                        className={`font-medium border-2 border-main transition-colors duration-250 cursor-pointer
                           px-[2vmin] py-[1vmin] text-[2.5vmin]
                           ${difficulty === option.value
                            ? 'bg-main text-white'
                            : 'bg-white text-main hover:bg-main hover:text-white'
                        }`}
                    >
                        {option.label}
                    </button>
                ))}
            </div>

            {/* Continue Button */}
            <div className="flex flex-col items-center">
                <button
                    onClick={proceedToRounds}
                    disabled={!difficulty}
                    className={`font-medium border-2 border-main bg-white text-main hover:bg-main hover:text-white transition-all duration-250 cursor-pointer
                    px-[2vmin] py-[1vmin] text-[2.5vmin]
                    ${difficulty ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                >
                    Continue
                </button>
            </div>
        </div>
    )
}

export default Difficulty