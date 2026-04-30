import { useNavigate } from 'react-router-dom'
import '../App.css'
import {images} from "../config/themeConfig.js";

function Logo() {
    const navigate = useNavigate()
    const themeImages = images()

    const handleLogoClick = () => {
        // Clear all game settings from localStorage
        localStorage.removeItem('gameMode')
        localStorage.removeItem('difficulty')
        localStorage.removeItem('totalRounds')

        window.location.href = '/'
    }

    return (
        <img
            src={themeImages.logo}
            alt="KISZ Logo"
            className="absolute top-[1vw] left-[1vw] h-[7vmin] z-10 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleLogoClick}
        />
    )
}

export default Logo