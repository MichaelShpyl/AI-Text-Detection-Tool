
console.log("AI Text Detector extension content script loaded");

// Only proceed if the page is top-level (not in an iframe) and likely contains an article or substantial text
if (window === window.top) {
  // Create the "Scan this Article" button
  const scanButton = document.createElement('button');
  scanButton.innerText = 'Scan this article';
  scanButton.id = 'ai-text-detector-button';
  Object.assign(scanButton.style, {
    position: 'fixed',
    bottom: '20px',
    right: '20px',
    zIndex: '10000',
    padding: '10px 16px',
    backgroundColor: '#4c8bf5',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    boxShadow: '0 0 5px rgba(0,0,0,0.3)'
  });
  document.body.appendChild(scanButton);

  scanButton.addEventListener('click', () => {
    scanButton.disabled = true;
    scanButton.innerText = 'Scanning...';

    // Gather page text (prefer <article> if present)
    let pageText = '';
    const articleElement = document.querySelector('article');
    if (articleElement) {
      pageText = articleElement.innerText;
    } else {
      pageText = document.body.innerText;
    }

    // Send to local API
    fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: pageText.substring(0, 10000) })
    })
      .then(res => res.json())
      .then(data => displayResults(data))
      .catch(err => {
        console.error("Error scanning article:", err);
        alert("Failed to scan the article. Please ensure the backend is running.");
        scanButton.disabled = false;
        scanButton.innerText = 'Scan this article';
      });
  });

  function displayResults(data) {
    // Remove old overlay if present
    const old = document.getElementById('ai-detector-results');
    if (old) old.remove();

    // Build overlay
    const overlay = document.createElement('div');
    overlay.id = 'ai-detector-results';
    Object.assign(overlay.style, {
      position: 'fixed',
      bottom: '80px',
      right: '20px',
      backgroundColor: '#ffffffcc',
      color: '#000',
      padding: '10px',
      border: '1px solid #888',
      borderRadius: '4px',
      maxWidth: '300px',
      zIndex: '10000',
      fontFamily: 'Arial, sans-serif',
      fontSize: '14px'
    });

    // Prediction text
    const pred = document.createElement('div');
    pred.innerHTML = `<strong>Detected as:</strong> ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
    overlay.appendChild(pred);

    // Probability bars
    if (data.probabilities) {
      for (const [label, prob] of Object.entries(data.probabilities)) {
        const barContainer = document.createElement('div');
        Object.assign(barContainer.style, {
          display: 'flex',
          alignItems: 'center',
          margin: '3px 0'
        });

        const bar = document.createElement('div');
        Object.assign(bar.style, {
          height: '6px',
          width: (prob * 100) + '%',
          backgroundColor: '#4c8bf5'
        });
        barContainer.appendChild(bar);

        const labelSpan = document.createElement('span');
        Object.assign(labelSpan.style, {
          marginLeft: '8px',
          fontSize: '12px'
        });
        labelSpan.innerText = `${label}: ${(prob * 100).toFixed(1)}%`;
        barContainer.appendChild(labelSpan);

        overlay.appendChild(barContainer);
      }
    }

    // Highlight words if explanation provided
    if (data.explanation && data.explanation.length > 0) {
      highlightWords(data.explanation);
    }

    // Append overlay and re-enable button
    document.body.appendChild(overlay);
    scanButton.disabled = false;
    scanButton.innerText = 'Scan this article';
  }

  function highlightWords(explanation) {
    const container = document.querySelector('article') || document.body;

    // Hide our own UI to avoid highlighting it
    const resultsOverlay = document.getElementById('ai-detector-results');
    if (resultsOverlay) resultsOverlay.style.display = 'none';
    const scanBtn = document.getElementById('ai-text-detector-button');
    if (scanBtn) scanBtn.style.display = 'none';

    // Prepare highlighting parameters
    const maxWeight = Math.max(...explanation.map(item => Math.abs(item.weight)));
    const highlightColor = weight => {
      const intensity = Math.min(Math.abs(weight) / maxWeight, 1);
      return `rgba(255, 255, 0, ${intensity})`;
    };

    // Walk text nodes
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    let node;
    while (node = walker.nextNode()) {
      textNodes.push(node);
    }

    textNodes.forEach(textNode => {
      const parent = textNode.parentNode;
      let text = textNode.nodeValue;
      if (!text) return;

      explanation.forEach(({ word, weight }) => {
        const regex = new RegExp(`\\b${word}\\b`, 'gi');
        if (regex.test(text)) {
          text = text.replace(regex, match => {
            const span = document.createElement('span');
            span.textContent = match;
            span.style.backgroundColor = highlightColor(weight);
            span.title = `Weight: ${weight.toFixed(2)}`;
            return span.outerHTML;
          });
        }
      });

      if (text !== textNode.nodeValue) {
        const temp = document.createElement('div');
        temp.innerHTML = text;
        while (temp.firstChild) {
          parent.insertBefore(temp.firstChild, textNode);
        }
        parent.removeChild(textNode);
      }
    });

    // Restore our UI
    if (resultsOverlay) resultsOverlay.style.display = 'block';
    if (scanBtn) scanBtn.style.display = 'block';
  }
}
