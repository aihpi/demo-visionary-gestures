import { useNavigate } from 'react-router-dom'
import { useState } from 'react'
import HandTrackingWebcam from './HandTrackingWebcam'
import GestureCalibration from './GestureCalibration'
import logoHPI from '../assets/images/combologo.png';
import InfoButton from './InfoButton'
import Logo from "./Logo.jsx";

function Tutorial({ gameMode = 'rps' }) {
    const navigate = useNavigate()
    const [currentGesture, setCurrentGesture] = useState('None')
    const [gestureConfidence, setGestureConfidence] = useState(0)
    const [showCalibration, setShowCalibration] = useState(true)

    const handleGestureDetected = (gestureData) => {
        console.log('Tutorial - Gesture detected:', gestureData)

        if (gestureData.gesture) {
            setCurrentGesture(gestureData.gesture.name)
            setGestureConfidence(gestureData.gesture.confidence)
        } else {
            setCurrentGesture('None')
            setGestureConfidence(0)
        }
    }

    const handleCalibrationComplete = () => {
        navigate('/difficulty')
    }

    return (
        <div className="w-full h-dvh">
            <div className="flex w-full h-full">
                <InfoButton />
                <Logo/>

                <div className="w-1/2 h-full items-center justify-center flex flex-col p-[1vmin]" style={{backgroundColor: '#fff'}}>
                    <GestureCalibration
                        currentGesture={currentGesture}
                        gestureConfidence={gestureConfidence}
                        gameMode={gameMode}
                        onCalibrationComplete={handleCalibrationComplete}
                    />
                </div>

                <div className="w-1/2 h-full" style={{backgroundColor: '#fff'}}>
                    <HandTrackingWebcam
                        onGestureDetected={handleGestureDetected}
                        showLandmarks={true}
                        mirrored={true}
                        fillContainer={true}
                        gameMode={gameMode}
                    />
                </div>
            </div>
        </div>
    )
}

export default Tutorial
