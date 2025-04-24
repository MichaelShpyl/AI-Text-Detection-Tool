// frontend/src/App.js

import React, { useState, useRef, useEffect } from 'react';

function App() {
  // View mode: 'single' for one-off text analysis, 'batch' for file uploads
  const [mode, setMode] = useState('single');
  // Mobile nav drawer open/closed
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  // Theme toggle: light vs. dark
  const [darkMode, setDarkMode] = useState(false);
  // Single-text input value
  const [textInput, setTextInput] = useState('');
  // Analysis result from backend
  const [analysisResult, setAnalysisResult] = useState(null);
  // Batch upload files and their statuses
  const [files, setFiles] = useState([]);
  const fileInputRef = useRef(null);

  // Apply or remove `<html class="dark">` based on darkMode
  useEffect(() => {
    if (darkMode) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [darkMode]);

  // Toggle between light and dark themes
  const toggleDarkMode = () => setDarkMode(!darkMode);

  // Send single text to backend API for AI detection
  const analyzeText = async () => {
    if (!textInput.trim()) return;
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textInput }),
      });
      const data = await res.json();
      setAnalysisResult(data);
    } catch (err) {
      console.error('Analysis error:', err);
    }
  };

  // Utility: copy JSON result to clipboard
  const copyToClipboard = () => {
    if (!analysisResult) return;
    navigator.clipboard.writeText(JSON.stringify(analysisResult, null, 2));
  };

  // Utility: download JSON result as file
  const downloadResult = () => {
    if (!analysisResult) return;
    const blob = new Blob([JSON.stringify(analysisResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'analysis_result.json';
    link.click();
  };

  // Utility: share via Web Share API if available
  const shareResult = () => {
    if (navigator.share && analysisResult) {
      navigator.share({
        title: 'AI Text Detector Result',
        text: JSON.stringify(analysisResult, null, 2),
      }).catch(console.error);
    } else {
      alert('Sharing not supported or no result.');
    }
  };

  // Handle batch file selection / drag-and-drop
  const handleFiles = (selectedFiles) => {
    Array.from(selectedFiles).forEach((file) => {
      const entry = { file, status: 'uploading' };
      setFiles((prev) => [...prev, entry]);

      const form = new FormData();
      form.append('file', file);

      fetch('http://localhost:8000/analyze-file', {
        method: 'POST',
        body: form,
      })
        .then(() =>
          setFiles((prev) =>
            prev.map((f) =>
              f.file === file ? { ...f, status: 'done' } : f
            )
          )
        )
        .catch(() =>
          setFiles((prev) =>
            prev.map((f) =>
              f.file === file ? { ...f, status: 'error' } : f
            )
          )
        );
    });
  };

  return (
    <div className="min-h-screen flex bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex md:flex-col w-64 bg-white dark:bg-gray-800 border-r border-gray-300 dark:border-gray-700">
        <div className="px-6 py-4">
          <h1 className="text-2xl font-bold">AI Text Detector</h1>
        </div>
        <nav className="flex-1 px-6 space-y-2">
          <button
            onClick={() => setMode('single')}
            className={`w-full text-left px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
              mode === 'single' ? 'bg-gray-200 dark:bg-gray-700' : ''
            }`}
          >
            Single Text
          </button>
          <button
            onClick={() => setMode('batch')}
            className={`w-full text-left px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
              mode === 'batch' ? 'bg-gray-200 dark:bg-gray-700' : ''
            }`}
          >
            Batch Upload
          </button>
        </nav>
        <div className="px-6 py-4 border-t border-gray-300 dark:border-gray-700">
          <button
            onClick={toggleDarkMode}
            className="w-full text-left px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
          </button>
        </div>
      </aside>

      {/* Mobile Header */}
      <header className="md:hidden flex items-center justify-between w-full px-4 py-3 bg-white dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
        <h1 className="text-lg font-bold">AI Text Detector</h1>
        <button
          onClick={() => setMobileNavOpen(!mobileNavOpen)}
          aria-label="Toggle menu"
          className="space-y-1"
        >
          <span className="block w-6 h-0.5 bg-current"></span>
          <span className="block w-6 h-0.5 bg-current"></span>
          <span className="block w-6 h-0.5 bg-current"></span>
        </button>
      </header>

      {/* Mobile Drawer */}
      {mobileNavOpen && (
        <aside className="fixed inset-0 z-30 bg-white dark:bg-gray-800 p-6">
          <button
            onClick={() => setMobileNavOpen(false)}
            className="mb-6 text-xl"
            aria-label="Close menu"
          >
            &times;
          </button>
          <nav className="space-y-4">
            <button
              onClick={() => { setMode('single'); setMobileNavOpen(false); }}
              className="block w-full text-left px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              Single Text
            </button>
            <button
              onClick={() => { setMode('batch'); setMobileNavOpen(false); }}
              className="block w-full text-left px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              Batch Upload
            </button>
            <button
              onClick={() => { toggleDarkMode(); setMobileNavOpen(false); }}
              className="block w-full text-left px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </button>
          </nav>
        </aside>
      )}

      {/* Main Content */}
      <main className="flex-1 p-6 overflow-y-auto">
        {mode === 'single' ? (
          <div className="max-w-xl mx-auto space-y-4">
            {/* Single Text Input */}
            <textarea
              className="
                w-full h-48
                p-3
                bg-white dark:bg-gray-800
                text-gray-900 dark:text-gray-100
                placeholder-gray-500 dark:placeholder-gray-400
                border border-gray-300 dark:border-gray-700
                rounded focus:outline-none focus:ring
              "
              placeholder="Paste or type your text here..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
            />

            {/* Analyze Button */}
            <button
              onClick={analyzeText}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
            >
              Analyze
            </button>

            {/* Copy / Download / Share */}
            {analysisResult && (
              <div className="flex space-x-2">
                <button
                  onClick={copyToClipboard}
                  className="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 px-3 py-1 rounded"
                >
                  Copy
                </button>
                <button
                  onClick={downloadResult}
                  className="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 px-3 py-1 rounded"
                >
                  Download
                </button>
                <button
                  onClick={shareResult}
                  className="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 px-3 py-1 rounded"
                >
                  Share
                </button>
              </div>
            )}

            {/* Results Display */}
            {analysisResult && (
              <div className="mt-6 space-y-4">
                <div>
                  <strong>Prediction:</strong> {analysisResult.prediction}
                </div>

                {/* Probability Bars */}
                <div className="space-y-2">
                  {Object.entries(analysisResult.probabilities).map(
                    ([label, prob]) => (
                      <div key={label}>
                        <div className="flex justify-between text-sm">
                          <span>{label}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 h-2 rounded overflow-hidden">
                          <div
                            className={`h-2 rounded ${
                              label === 'Human-written'
                                ? 'bg-green-500'
                                : label === 'AI-paraphrased'
                                ? 'bg-purple-500'
                                : 'bg-red-500'
                            }`}
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                      </div>
                    )
                  )}
                </div>

                {/* LIME-style Highlights */}
                <div className="mt-4 leading-relaxed">
                  {analysisResult.explanation.map((token, i) => {
                    const weight = token.weight;
                    const intensity = Math.min(0.8, Math.abs(weight));
                    const bgColor = weight > 0
                      ? `rgba(34,197,94,${intensity})`   // green highlight
                      : `rgba(239,68,68,${intensity})`; // red highlight
                    return (
                      <span
                        key={i}
                        style={{ backgroundColor: bgColor }}
                        title={`Weight: ${weight.toFixed(3)}`}
                      >
                        {token.word}{' '}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ) : (
          // Batch Upload Mode
          <div className="max-w-xl mx-auto space-y-4">
            <div
              className="border-2 border-dashed p-6 text-center cursor-pointer
                         bg-white dark:bg-gray-800
                         text-gray-900 dark:text-gray-100
                         placeholder-gray-500 dark:placeholder-gray-400"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleFiles(e.dataTransfer.files);
              }}
            >
              <p>Drag & drop files here, or{' '}
                <button
                  onClick={() => fileInputRef.current.click()}
                  className="underline text-blue-500"
                >
                  browse
                </button>
              </p>
              <input
                type="file"
                multiple
                accept=".txt,.pdf,.docx,.html"
                ref={fileInputRef}
                onChange={(e) => handleFiles(e.target.files)}
                className="hidden"
              />
            </div>
            <ul className="space-y-2">
              {files.map((fObj, idx) => (
                <li
                  key={idx}
                  className="flex justify-between p-3 bg-white dark:bg-gray-800
                             text-gray-900 dark:text-gray-100
                             border rounded"
                >
                  <span>{fObj.file.name}</span>
                  <span className={fObj.status === 'error' ? 'text-red-500' : ''}>
                    {fObj.status === 'uploading' && 'Uploading...'}
                    {fObj.status === 'done' && 'Done'}
                    {fObj.status === 'error' && 'Error'}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
