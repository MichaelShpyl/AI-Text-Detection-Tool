import React, { useState } from 'react';
import SingleAnalysis from './components/SingleAnalysis';
import BatchUpload   from './components/BatchUpload';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [page, setPage] = useState('single');

  const toggleDark = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark', !darkMode);
  };

  return (
    <div className={`${darkMode ? 'dark' : ''} flex h-screen bg-gray-100 dark:bg-gray-900`}>
      {/* Desktop sidebar */}
      <aside className="hidden md:flex flex-col w-64 bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
        <div className="p-4 border-b border-gray-300 dark:border-gray-700">
          <h2 className="text-xl font-bold">AI Text Detector</h2>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <button
            onClick={() => setPage('single')}
            className={`w-full text-left px-3 py-2 rounded hover:bg-gray-300 dark:hover:bg-gray-700 ${
              page==='single' ? 'bg-gray-300 dark:bg-gray-700' : ''
            }`}
          ><span role="img" aria-label="single text">🔎</span> Single Text
</button>

          <button
            onClick={() => setPage('batch')}
            className={`w-full text-left px-3 py-2 rounded hover:bg-gray-300 dark:hover:bg-gray-700 ${
              page==='batch' ? 'bg-gray-300 dark:bg-gray-700' : ''
            }`}
          ><span role="img" aria-label="batch upload">📂</span> Batch Upload</button>
        </nav>
        <div className="p-4 border-t border-gray-300 dark:border-gray-700">
          <button
            onClick={toggleDark}
            className="w-full text-left px-3 py-2 rounded hover:bg-gray-300 dark:hover:bg-gray-700"
          >
            {darkMode ? '🌙 Dark: ON' : '🌞 Dark: OFF'}
          </button>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="flex md:hidden items-center justify-between p-4 bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
      <button onClick={() => setMenuOpen(true)} className="focus:outline-none" aria-label="Open menu"> ☰ </button>
        <h2 className="text-lg font-semibold">AI Text Detector</h2>
        <button onClick={toggleDark}     className="focus:outline-none" aria-label="Toggle dark mode">
        {darkMode
        ? <span role="img" aria-label="dark mode on">🌙</span>
        : <span role="img" aria-label="dark mode off">🌞</span>
        } 
        </button>
      </header>

      {/* Mobile drawer */}
      {menuOpen && <div className="fixed inset-0 bg-black bg-opacity-50 z-20" onClick={() => setMenuOpen(false)} />}
      <aside
        className={`fixed inset-y-0 left-0 w-64 p-4 bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100 z-30 transform transition-transform duration-300 ${
          menuOpen ? 'translate-x-0' : '-translate-x-full'
        } md:-translate-x-full`}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">AI Text Detector</h2>
          <button onClick={() => setMenuOpen(false)}>✕</button>
        </div>
        <nav className="space-y-2">
          <button
            onClick={() => { setPage('single'); setMenuOpen(false); }}
            className="block w-full text-left px-3 py-2 rounded hover:bg-gray-300 dark:hover:bg-gray-700"
          ><span role="img" aria-label="single text">🔎</span> Single Text
</button>
          <button
            onClick={() => { setPage('batch'); setMenuOpen(false); }}
            className="block w-full text-left px-3 py-2 rounded hover:bg-gray-300 dark:hover:bg-gray-700"
          ><span role="img" aria-label="batch upload">📂</span> Batch Upload</button>
        </nav>
        <div className="mt-4 pt-4 border-t border-gray-400">
          <button
            onClick={toggleDark}
            className="w-full text-left px-3 py-2 rounded hover:bg-gray-300 dark:hover:bg-gray-700"
          >
            {darkMode ? '🌙 Dark: ON' : '🌞 Dark: OFF'}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-4">
        {page === 'single' ? <SingleAnalysis /> : <BatchUpload />}
      </main>
    </div>
  );
}

export default App;
