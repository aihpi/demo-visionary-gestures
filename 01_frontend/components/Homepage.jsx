import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom'
import logoHPI from '../assets/images/combologo.png';
import InfoButton from './InfoButton'
import '../App.css'

import Logo from "./Logo.jsx";
import {images} from "../config/themeConfig.js";

function HomePage() {
    const navigate = useNavigate()
    const themeImages = images()

    return (
      <div className="h-dvh flex flex-col items-center justify-center bg-white">
        <InfoButton />
          <Logo/>

        {/*}<img src={gradientImage} alt="Gradient" className="absolute -bottom-100 -right-100 h-256 w-auto z-00"/>*/}
          <div className="relative mt-16">

              <img
                  src={themeImages.comboImage}
                  alt="RPSLS Gestures"
                  className="absolute top-[1.3vw] right-1 h-[3vw] object-contain"
              />

              <h1 className="text-[10vw] font-bold leading-none text-main mt-8">Visionary</h1>
              <h1 className="text-[10vw] font-bold leading-none text-main mb-8">Gestures</h1>
          </div>

          {/*<div className="relative mt-16">
            <h3 className="absolute top-[1.2vw] left-[1vw] text-[3vw] font-bold text-main">Beat</h3>

            <img
              src={comboImage}
              alt="RPSLS Gestures"
              className="absolute top-[1.3vw] right-1 h-[3vw] object-contain"
            />

            <h1 className="text-[15vw] font-bold text-main mb-8">Sheldon</h1>
          </div>*/}

          <div className="flex flex-col items-center space-y-4">
            <button
              onClick={() => navigate('/howtoplay')}
              className="font-medium border-2 border-main bg-white text-main hover:bg-main hover:text-white transition-colors duration-250 cursor-pointer
                           px-[3vmin] py-[1vmin] text-[2.5vmin]"
            >
              Play
            </button>
          </div>
      </div>
    )
  }
  export default HomePage