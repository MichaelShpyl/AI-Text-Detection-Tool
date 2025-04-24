import React, { useState } from 'react';

/**
 * SingleAnalysis Component
 * Allows user to input text and get a prediction with highlighted explanation.
 */
function SingleAnalysis() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);  // will hold { prediction, confidence, probabilities, explanation }
  const [loading, setLoading] = useState(false);

  const analyzeText = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Analysis error:", err);
      setResult({ prediction: "Error", confidence: 0, probabilities: {}, explanation: [] });
    } finally {
      setLoading(false);
    }
  };

  // Highlight the input text based on explanation weights
  const renderHighlightedText = () => {
    if (!result || !result.explanation) return text;
    // Compute max weight for normalization
    const weights = result.explanation;
    const maxWeight = Math.max(...weights.map(w => Math.abs(w.weight)), 0.001);
    // Split text by spaces to highlight each word that matches
    return text.split(" ").map((word, idx) => {
      // Find if this word (case-insensitive) is in top features
      const match = weights.find(w => w.word.toLowerCase() === word.replace(/[^\w]/g, "").toLowerCase());
      if (match) {
        const opacity = Math.min(Math.abs(match.weight) / maxWeight, 1).toFixed(2);
        const highlightColor = `rgba(255, 255, 0, ${opacity})`;  // yellow highlight
        return (
          <span key={idx} style={{ backgroundColor: highlightColor }} title={`Weight: ${match.weight.toFixed(2)}`}>
            {word + " "}
          </span>
        );
      } else {
        return <span key={idx}>{word + " "}</span>;
      }
    });
  };

  return (
    <div className="max-w-xl mx-auto">
      <h3 className="text-lg font-semibold mb-2">Single Text Analysis</h3>
      <textarea 
        className="w-full p-2 border border-gray-400 rounded dark:bg-gray-800 dark:text-gray-100" 
        rows="5" 
        placeholder="Enter text here..." 
        value={text} 
        onChange={(e) => setText(e.target.value)}>
      </textarea>
      <button 
        onClick={analyzeText} 
        className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>

      {result && (
        <div className="mt-4 p-3 border border-gray-300 dark:border-gray-700 rounded">
          <h4 className="font-bold mb-2">Prediction: {result.prediction}</h4>
          {result.probabilities && (
            <div className="mb-2">
              {/* Probability bars for each class */}
              {Object.entries(result.probabilities).map(([label, prob]) => (
                <div key={label} className="flex items-center mb-1">
                  <div className="h-2 rounded bg-green-500" style={{ width: `${prob*100}%` }} />
                  <span className="ml-2 text-xs">{label}: {(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
           {result.explanation && (
            <p className="text-sm leading-relaxed">
            {renderHighlightedText()}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default SingleAnalysis;
