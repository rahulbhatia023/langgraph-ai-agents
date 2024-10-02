import React from 'react';
import 'tailwindcss/tailwind.css';

const RainbowCircle = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-64 h-64 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 animate-spin"></div>
    </div>
  );
};

const App = () => {
  return (
    <RainbowCircle />
  );
};

export default App;