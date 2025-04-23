import React, { useState } from 'react';

/**
 * BatchUpload Component
 * Provides a drag-and-drop area and file selection for uploading multiple files.
 * Displays a list of files with their analysis status/results.
 */
function BatchUpload() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  // Handle file selection (from drag-drop or file input)
  const handleFilesSelected = (selectedFiles) => {
    const fileArray = Array.from(selectedFiles);
    if (!fileArray.length) return;
    const initialFileStates = fileArray.map(file => ({
      file, status: 'pending', result: null
    }));
    setFiles(initialFileStates);
    analyzeFiles(initialFileStates);
  };

  // Send each file to the backend for analysis
  const analyzeFiles = async (fileStates) => {
    setUploading(true);
    for (let i = 0; i < fileStates.length; i++) {
      const { file } = fileStates[i];
      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('http://127.0.0.1:8000/analyze-file', {
          method: 'POST', body: formData
        });
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        setFiles(prev => {
          const copy = [...prev];
          copy[i] = { file, status: 'done', result: {
            label: data.prediction,
            confidence: data.confidence
          }};
          return copy;
        });
      } catch {
        setFiles(prev => {
          const copy = [...prev];
          copy[i] = { file, status: 'done', result: {
            label: 'Error', confidence: 0
          }};
          return copy;
        });
      }
    }
    setUploading(false);
  };

  // Drag-n-drop handlers
  const handleDragOver = e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; };
  const handleDrop     = e => { e.preventDefault(); handleFilesSelected(e.dataTransfer.files); };
  const handleInput    = e => { handleFilesSelected(e.target.files); };

  return (
    <div className="max-w-xl mx-auto">
      <h3 className="text-lg font-semibold mb-2">Batch File Analysis</h3>

      {/* Drag & drop zone */}
      <div
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className="flex items-center justify-center p-6 mb-4 border-2 border-dashed border-gray-400 rounded cursor-pointer hover:border-gray-600 dark:border-gray-500 dark:hover:border-gray-300"
      >
        <div className="text-center">
          <p>Drag & drop files here, or <span className="text-blue-600 underline">browse</span></p>
          <input
            id="file-upload-input"
            type="file"
            multiple
            className="hidden"
            onChange={handleInput}
          />
          <label htmlFor="file-upload-input" className="cursor-pointer text-blue-600 underline">
            choose files
          </label>
        </div>
      </div>

      {/* Results table */}
      {files.length > 0 && (
        <table className="min-w-full text-sm border-t border-b border-gray-300 dark:border-gray-700">
          <thead className="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <tr>
              <th className="py-2 px-2 text-left">File Name</th>
              <th className="py-2 px-2">Result</th>
              <th className="py-2 px-2">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {files.map((fs, idx) => (
              <tr key={idx} className="border-b border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800">
                <td className="py-2 px-2">{fs.file.name}</td>
                <td className="py-2 px-2">
                  {fs.status === 'pending'
                    ? <em className="text-gray-600 dark:text-gray-400">Processing...</em>
                    : fs.result.label}
                </td>
                <td className="py-2 px-2">
                  {fs.status === 'pending'
                    ? '...'
                    : `${(fs.result.confidence * 100).toFixed(1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {uploading && <p className="mt-2 text-gray-600 dark:text-gray-400">Analyzing files...</p>}
    </div>
  );
}

export default BatchUpload;
