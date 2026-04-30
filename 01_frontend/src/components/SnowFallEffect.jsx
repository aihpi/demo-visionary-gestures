import React from 'react';
import Snowfall from 'react-snowfall';

const SnowfallEffect = () => {
    const currentTheme = localStorage.getItem('theme') || 'default';

    // Only show snowfall when Christmas theme is active
    if (currentTheme !== 'christmas') {
        return null;
    }

    return (
        <Snowfall
            color="#AED3D8"
            snowflakeCount={200}
            radius={[0.5, 3.0]}
            speed={[0.5, 3.0]}
            wind={[-0.5, 2.0]}
            style={{
                position: 'fixed',
                width: '100vw',
                height: '100vh',
                top: 0,
                left: 0,
                zIndex: 9999,
                pointerEvents: 'none'
            }}
        />
    );
};

export default SnowfallEffect;