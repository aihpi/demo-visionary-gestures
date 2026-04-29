import { useNavigate } from 'react-router-dom'
import RPSIcon from '../assets/images/rps.svg'
import RPSLSIcon from '../assets/images/rpsls.svg'
import logoHPI from '../assets/images/combologo.png';
import InfoButton from './InfoButton'
import Logo from "./Logo.jsx";
import {images} from "../config/themeConfig.js";

function HowToPlay({ gameMode, setGameMode }) {
    const navigate = useNavigate()
    const themeImages = images()

    const handleGameModeSelect = (mode) => {
        setGameMode(mode);
        localStorage.setItem('gameMode', mode);
        navigate('/tutorial');
    }

    return (
        <div className="h-dvh w-full flex flex-col overflow-hidden">
            <InfoButton />
            <Logo/>

            {/* Title Section */}
            <div className="pt-[3vh] pb-[2vh]">
                <h1 className="text-center font-bold text-main text-[4vw] leading-tight">
                    Choose your Gamemode:
                </h1>
            </div>

            {/* Game Mode Selection */}
            <div className="flex-1 flex justify-center px-[2vw] pb-[2vh]">
                <div className="flex w-full max-w-[90vw] h-full">

                    {/* Left Column: Rock Paper Scissors */}
                    <div className="w-1/2 flex flex-col items-center justify-center p-[1vmin] bg-white border-r border-gray-200">

                        {/* Title */}
                        <h2 className="text-center font-bold text-main mb-[2vh] text-[4vmin] leading-tight">
                            Rock Paper Scissors
                        </h2>

                        {/* Image Container */}
                        <div className="flex items-center justify-center mb-[2vh] w-full">
                            <div className="w-[25vmin] h-[25vmin] flex items-center justify-center">
                                <img
                                    src={themeImages.rpsIcon}
                                    alt="Rock Paper Scissors"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        </div>

                        {/* Description */}
                        <p className="text-center font-medium text-gray-600 mb-[3vh] px-[1vw] text-[2vmin] leading-relaxed max-w-[20vw]">
                            The standard Rock Paper Scissors Game.
                        </p>

                        {/* Play Button */}
                        <button
                            onClick={() => handleGameModeSelect('rps')}
                            className={`font-medium border-2 border-main transition-colors duration-250 cursor-pointer
                         px-[3vmin] py-[1vmin] text-[2.5vmin]
                         ${gameMode === 'rps'
                                ? 'bg-main text-white'
                                : 'bg-white text-main hover:bg-main hover:text-white'
                            }`}
                        >
                            Play
                        </button>
                    </div>

                    {/* Right Column: Rock Paper Scissors Lizard Spock */}
                    <div className="w-1/2 flex flex-col items-center justify-center p-[1vmin] bg-white">

                        {/* Title */}
                        <h2 className="text-center font-bold text-main mb-[2vh] text-[4vmin] leading-tight">
                            Rock Paper Scissors <br/>+ Lizard Spock
                        </h2>

                        {/* Image Container */}
                        <div className="flex items-center justify-center mb-[2vh] w-full">
                            <div className="w-[25vmin] h-[25vmin] flex items-center justify-center">
                                <img
                                    src={themeImages.rpslsIcon}
                                    alt="Rock Paper Scissors Lizard Spock"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        </div>

                        {/* Description */}
                        <p className="text-center font-medium text-gray-600 mb-[3vh] px-[1vw] text-[2vmin] leading-relaxed max-w-[20vw]">
                            A Variation with two more moves: Lizard and Spock.
                        </p>

                        {/* Play Button */}
                        <button
                            onClick={() => handleGameModeSelect('rpls')}
                            className={`font-medium border-2 border-main transition-colors duration-250 cursor-pointer
                         px-[3vmin] py-[1vmin] text-[2.5vmin]
                         ${gameMode === 'rpls'
                                ? 'bg-main text-white'
                                : 'bg-white text-main hover:bg-main hover:text-white'
                            }`}
                        >
                            Play
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default HowToPlay