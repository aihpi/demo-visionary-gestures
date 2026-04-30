import { useNavigate } from 'react-router-dom'

function InfoButton() {
    const navigate = useNavigate()

    return (
        <div className="fixed bottom-4 left-4 flex gap-2 z-50">
            <button
                onClick={() => navigate('/info')}
                className="w-12 h-12 text-lg font-bold border-2 border-main bg-transparent text-main hover:bg-main hover:text-[#fbf4ed] transition-colors duration-250 cursor-pointer flex items-center justify-center backdrop-blur-lg"
                title="Information"
            >
                i
            </button>
            <button
                onClick={() => navigate('/imprint')}
                className="w-auto p-2 h-12 text-lg font-bold border-2 border-main bg-transparent text-main hover:bg-main hover:text-[#fbf4ed] transition-colors duration-250 cursor-pointer flex items-center justify-center backdrop-blur-lg"
                title="Imprint"
            >
                Imprint
            </button>
        </div>
    )
}

export default InfoButton