console.log("AI Text Detector extension content script loaded");

// Only proceed if the page is top-level (not in an iframe) and likely contains an article or substantial text
if (window === window.top) {
  // Create the "Scan this Article" button
  const scanButton = document.createElement('button');
  scanButton.innerText = 'Scan this article';
  scanButton.id = 'ai-text-detector-button';
  scanButton.style.position = 'fixed';
  scanButton.style.bottom = '20px';
  scanButton.style.right = '20px';
  scanButton.style.zIndex = '10000';
  scanButton.style.padding = '10px 16px';
  scanButton.style.backgroundColor = '#4c8bf5';
  scanButton.style.color = 'white';
  scanButton.style.border = 'none';
  scanButton.style.borderRadius = '4px';
  scanButton.style.cursor = 'pointer';
  scanButton.style.boxShadow = '0 0 5px rgba(0,0,0,0.3)';
  
  document.body.appendChild(scanButton);

  scanButton.addEventListener('click', () => {
    scanButton.disabled = true;
    scanButton.innerText = 'Scanning...';
    // Gather page text (prefer <article> content if exists, otherwise whole body)
    let pageText = "";
    const articleElement = document.querySelector('article');
    if (articleElement) {
      pageText = articleElement.innerText;
    } else {
      pageText = document.body.innerText;
    }
    // Send the text to the local API for analysis
    fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: pageText.substring(0, 10000) })  // limit size for very large pages
    })
      .then(response => response.json())
      .then(data => {
        displayResults(data);
      })
      .catch(err => {
        console.error("Error scanning article:", err);
        alert("Failed to scan the article. Please ensure the backend is running.");
        scanButton.disabled = false;
        scanButton.innerText = 'Scan this article';
      });
  });

  function displayResults(data) {
    // Remove any existing result overlay
    const oldOverlay = document.getElementById('ai-detector-results');
    if (oldOverlay) oldOverlay.remove();

    // Create a result overlay div
    const overlay = document.createElement('div');
    overlay.id = 'ai-detector-results';
    overlay.style.position = 'fixed';
    overlay.style.bottom = '80px';
    overlay.style.right = '20px';
    overlay.style.backgroundColor = '#ffffffcc';
    overlay.style.color = '#000';
    overlay.style.padding = '10px';
    overlay.style.border = '1px solid #888';
    overlay.style.borderRadius = '4px';
    overlay.style.maxWidth = '300px';
    overlay.style.zIndex = '10000';
    overlay.style.fontFamily = 'Arial, sans-serif';
    overlay.style.fontSize = '14px';

    const pred = document.createElement('div');
    pred.innerHTML = `<strong>Detected as:</strong> ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
    overlay.appendChild(pred);

    // Probability bars
    const probs = data.probabilities;
    if (probs) {
      for (const [label, prob] of Object.entries(probs)) {
        const barContainer = document.createElement('div');
        barContainer.style.display = 'flex';
        barContainer.style.alignItems = 'center';
        barContainer.style.margin = '3px 0';
        const bar = document.createElement('div');
        bar.style.height = '6px';
        bar.style.width = (prob * 100) + '%';
        bar.style.backgroundColor = '#4c8bf5';
        barContainer.appendChild(bar);
        const labelSpan = document.createElement('span');
        labelSpan.style.marginLeft = '8px';
        labelSpan.style.fontSize = '12px';
        labelSpan.innerText = `${label}: ${(prob * 100).toFixed(1)}%`;
        barContainer.appendChild(labelSpan);
        overlay.appendChild(barContainer);
      }
    }

    // Highlight important words in the page text (using explanation if available)
    if (data.explanation && data.explanation.length > 0) {
      highlightWords(data.explanation);
    }

    // Append overlay
    document.body.appendChild(overlay);
    scanButton.disabled = false;
    scanButton.innerText = 'Scan this article';
  }

  function highlightWords(explanation) {
    const topWords = explanation.map(item => item.word);
    const maxWeight = Math.max(...explanation.map(item => Math.abs(item.weight)));
    // Use yellow highlight for important words
    const highlightColor = (weight) => {
      const intensity = Math.min(Math.abs(weight) / maxWeight, 1);
      return `rgba(255, 255, 0, ${intensity})`;
    };
    // Iterate through text nodes in the article or body and wrap top words
    const container = document.querySelector('article') || document.body;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const nodesToHighlight = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      nodesToHighlight.push(node);
    }
    nodesToHighlight.forEach(node => {
      const parent = node.parentNode;
      let text = node.nodeValue;
      if (!text) return;
      topWords.forEach((wordObj) => {
        const word = wordObj.toString();
      });
      explanation.forEach(({ word, weight }) => {
        // Create regex to find whole words (case-insensitive)
        const regex = new RegExp(`\\b${word}\\b`, 'gi');
        if (text.match(regex)) {
          const span = document.createElement('span');
          span.style.backgroundColor = highlightColor(weight);
          span.title = `Weight: ${weight.toFixed(2)}`;
          span.textContent = text.match(regex)[0];
          text = text.replace(regex, span.outerHTML);
        }
      });
      // Replace node with new HTML (we have to do this via a container since node is a text node)
      if (text !== node.nodeValue) {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = text;
        while (tempDiv.firstChild) {
          parent.insertBefore(tempDiv.firstChild, node);
        }
        parent.removeChild(node);
      }
    });
  }
}
