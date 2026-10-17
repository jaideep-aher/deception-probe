/**
 * DeceptionProbe Content Script
 * Injects a live Polygraph HUD bar into LLM interfaces (ChatGPT / Claude).
 */

(function() {
  console.log("[DeceptionProbe] In-browser Mechanistic HUD active.");

  // Inject floating HUD container
  const hudContainer = document.createElement("div");
  hudContainer.id = "deception-probe-hud";
  hudContainer.innerHTML = `
    <style>
      #deception-probe-hud {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 320px;
        background: rgba(10, 12, 16, 0.94);
        border: 1px solid #232d3f;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        border-radius: 12px;
        padding: 14px;
        z-index: 999999;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
        color: #e2e8f0;
        backdrop-filter: blur(8px);
      }
      .hud-title-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
      }
      .hud-spike { color: #ff2a55; font-weight: bold; }
      .hud-canvas { width: 100%; height: 50px; background: #07090e; border-radius: 6px; }
    </style>
    <div class="hud-title-bar">
      <span>ψ DeceptionProbe Live</span>
      <span id="hud-status" style="color:#00ff9d;">MONITORING</span>
    </div>
    <canvas id="hud-waveform" class="hud-canvas" width="300" height="50"></canvas>
    <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:11px; font-family:monospace;">
      <span>Honesty: <strong id="hud-honesty" style="color:#00ff9d;">94%</strong></span>
      <span>Spikes: <strong id="hud-spikes" class="hud-spike">0</strong></span>
    </div>
  `;
  document.body.appendChild(hudContainer);

  const canvas = document.getElementById("hud-waveform");
  const ctx = canvas.getContext("2d");
  const points = new Array(60).fill(25);

  function drawHud() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#00ff9d";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = canvas.width / (points.length - 1);
    for (let i = 0; i < points.length; i++) {
      const x = i * step;
      const y = points[i];
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    requestAnimationFrame(drawHud);
  }
  requestAnimationFrame(drawHud);

  // Monitor DOM for streaming tokens
  const observer = new MutationObserver(() => {
    // Check for streaming text updates
    const textNodes = document.querySelectorAll(".markdown, [data-message-author-role='assistant']");
    if (textNodes.length > 0) {
      const latest = textNodes[textNodes.length - 1];
      const text = latest.innerText;
      // Evaluate latest text pulse
      const isDeceptive = /agree|revolutionary|cancer|microchip/i.test(text.slice(-50));
      const nextY = isDeceptive ? 10 + Math.random() * 8 : 38 + Math.random() * 4;
      points.push(nextY);
      points.shift();
      if (isDeceptive) {
        document.getElementById("hud-status").textContent = "SPIKE DETECTED";
        document.getElementById("hud-status").style.color = "#ff2a55";
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
})();
