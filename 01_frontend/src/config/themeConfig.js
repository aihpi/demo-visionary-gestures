// Import all images so Vite can properly bundle them
import defaultLogo from '../assets/images/combologo.png';
import defaultComboImage from '../assets/images/comboImage.svg';
import defaultRpsIcon from '../assets/images/rps.svg';
import defaultRpslsIcon from '../assets/images/rpsls.svg';
import defaultHowItWorks from '../assets/images/HowItWorks.png';

import defaultRockIcon from '../assets/images/rock.svg';
import defaultPaperIcon from '../assets/images/paper.svg';
import defaultScissorsIcon from '../assets/images/scissors.svg';
import defaultLizardIcon from '../assets/images/lizard.svg';
import defaultSpockIcon from '../assets/images/spock.svg';

import christmasLogo from '../assets/images/christmas-combologo.png';
import christmasComboImage from '../assets/images/christmas-comboImage.svg';
import christmasRpsIcon from '../assets/images/christmas-rps.svg';
import christmasRpslsIcon from '../assets/images/christmas-rpsls.svg';
// import christmasHowItWorks from '../assets/images/christmas-howitworks.png';


import christmasRockIcon from '../assets/images/christmas-rock.svg';
import christmasPaperIcon from '../assets/images/christmas-paper.svg';
import christmasScissorsIcon from '../assets/images/christmas-scissors.svg';
import christmasLizardIcon from '../assets/images/christmas-lizard.svg';
import christmasSpockIcon from '../assets/images/christmas-spock.svg';

// Get current theme from localStorage
const getCurrentTheme = () => {
    return localStorage.getItem('theme') || 'default';
};

// Default theme
const defaultImages = {
    logo: defaultLogo,
    comboImage: defaultComboImage,
    rpsIcon: defaultRpsIcon,
    rpslsIcon: defaultRpslsIcon,
    howItWorks: defaultHowItWorks,
    rock: defaultRockIcon,
    paper: defaultPaperIcon,
    scissors: defaultScissorsIcon,
    lizard: defaultLizardIcon,
    spock: defaultSpockIcon
};

const christmasImages = {
    logo: christmasLogo,
    comboImage: christmasComboImage,
    rpsIcon: christmasRpsIcon,
    rpslsIcon: christmasRpslsIcon,
    rock: christmasRockIcon,
    paper: christmasPaperIcon,
    scissors: christmasScissorsIcon,
    lizard: christmasLizardIcon,
    spock: christmasSpockIcon,

    // howItWorks: christmasHowItWorks
};

// Default colors
const defaultColors = {
    0: "177, 6, 58",
    1: "221, 97, 8",
    2: "246, 168, 0",
    3: "119, 154, 11",
    4: "0, 122, 158"
};

const christmasColors = {
    0: "142, 0, 3",   // Rock - #e05c6f (pink/rose)
    1: "247, 1, 35",     // Paper - #f70123 (bright red)
    2: "224, 92, 111",      // Scissors - #8e0003 (dark red)
    3: "174, 211, 216",
    4: "64, 86, 40" // Spock - #405628 (olive green)
};

// Export functions that return current themed values
export const images = () => {
    return getCurrentTheme() === 'christmas' ? christmasImages : defaultImages;
};

export const colors = () => {
    return getCurrentTheme() === 'christmas' ? christmasColors : defaultColors;
};

export const toggleTheme = () => {
    const current = getCurrentTheme();
    const newTheme = current === 'christmas' ? 'default' : 'christmas';
    localStorage.setItem('theme', newTheme);
    window.location.reload(); // Simple reload to apply theme
};