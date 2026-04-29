import { useNavigate } from 'react-router-dom'
import howitworks from "../assets/images/HowItWorks.png";
import {images, toggleTheme} from "../config/themeConfig.js";

function InfoPage() {
    const navigate = useNavigate()
    const themeImages = images()

    return (
        <div className="h-dvh w-full flex flex-col overflow-hidden">
            <div className="fixed bottom-4 left-4 flex gap-2 z-50">
                <button
                    onClick={() => navigate(-1)}
                    className="w-12 h-12 text-lg font-bold border-2 border-main bg-transparent text-main hover:bg-main hover:text-[#fbf4ed] transition-colors duration-250 cursor-pointer flex items-center justify-center backdrop-blur-lg"
                    title="Information"
                >
                    ✖
                </button>
                <button
                    onClick={toggleTheme}
                    className="w-auto p-2 h-12 text-lg font-bold border-2 border-main bg-transparent text-main hover:bg-main hover:text-[#fbf4ed] transition-colors duration-250 cursor-pointer flex items-center justify-center backdrop-blur-lg"
                    title="Imprint"
                >
                    {localStorage.getItem('theme') === 'christmas' ? 'Turn Christmas off' : 'Turn Christmas on'}
                </button>
            </div>


            <div className="pt-[1vh] pb-[3vh] px-[4vw]">
            </div>

            <div className="flex-1 overflow-y-auto px-[4vw] pb-[4vh]">
                <div className="max-w-[90vw] mx-auto flex gap-[4vw]">
                    <div className="border-2 border-main p-[2vmin]">
                        <h2 className="text-left font-bold text-main mb-[1vh] text-[3vmin] leading-tight">
                            How to Play
                        </h2>
                        <p className="font-medium text-gray-900 text-left text-[1.5vmin] leading-relaxed">
                            There are two game modes: <br/><br/>

                            First, the standard Rock Paper Scissors. If you're unfamiliar with the rules: <br/>
                            <b>Rock beats Scissors, Scissors beats Paper and Paper beats Rock.</b> <br/><br/>

                            Second, the Rock Paper Scissors Lizard Spock mode (also known from the TV show The Big Bang Theory). The rules here are more complex. For a detailed explanation beyond the provided diagram, you can watch the video tutorial: <a
                            href="/beatsheldon/videos/BBT_Explanation.mp4"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-main underline hover:font-bold"
                        >
                            Watch Video
                        </a><br/><br/>

                            After choosing your game mode, select your gesture and follow the tutorial to see how the neural network recognizes your hand movements. <br/><br/>

                            Choose your preferred difficulty and number of rounds, then press start game. <br/><br/>

                            Show your hand clearly during the 3-2-1 countdown and hold it still. Don't worry, the computer doesn't cheat (in normal mode). <br/><br/>

                            <b>Good Luck and have fun!</b>
                        </p>
                    </div>

                    <div className="border-2 border-main p-[2vmin]">
                        <h2 className="text-left font-bold text-main mb-[1vh] text-[3vmin] leading-tight">
                            How it Works
                        </h2>
                        <p className="font-medium text-gray-900 text-left text-[1.5vmin] leading-relaxed">
                            When you first open the website, two neural networks (AI models) are loaded. These are running completely local, meaning nothing leaves your browser:<br/><br/>

                            - Hand Detection AI: Recognizes where your hand appears in the webcam feed and tracks the 3D
                            position of each knuckle (powered by Google's MediaPipe Hands model)<br/>
                            - Gesture Recognition AI: Interprets the 3D coordinates of your knuckles to identify which
                            gesture you're making (developed by us)<br/><br/>

                            <img
                                src={howitworks}
                                alt="gesture recognition process diagram"
                                className="w-8/9 h-auto mx-auto my-[2vh]"

                            />

                            During gameplay:<br/>

                            1. Your webcam feed is processed by the MediaPipe model, which detects your hand and
                            captures the 3D coordinates of your knuckles<br/>
                            2. These coordinates are then analyzed by the gesture recognition model to determine which
                            move you're making<br/>
                            3. Finally, your gesture and the computer's independently chosen gesture are compared to
                            calculate the winner<br/>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default InfoPage