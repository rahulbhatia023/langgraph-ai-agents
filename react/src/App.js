import React from 'react';
import 'tailwindcss/tailwind.css';

const ExcelSheet = () => {
  const rows = 10;
  const cols = 5;
  const headers = ['A', 'B', 'C', 'D', 'E'];

  return (
    <div className="p-4">
      <table className="table-auto border-collapse border border-gray-400">
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index} className="border border-gray-300 px-4 py-2 bg-gray-200">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, rowIndex) => (
            <tr key={rowIndex}>
              {Array.from({ length: cols }).map((_, colIndex) => (
                <td key={colIndex} className="border border-gray-300 px-4 py-2">
                  {rowIndex + 1},{headers[colIndex]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const App = () => (
  <div className="min-h-screen flex items-center justify-center bg-gray-100">
    <ExcelSheet />
  </div>
);

export default App;