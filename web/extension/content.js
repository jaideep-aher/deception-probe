/**
 * DeceptionProbe Content Script (v1.1.0)
 * Injects real-time AI honesty auditing, sycophancy detection, and 1-click truth steering
 * into ChatGPT, Claude, Gemini, and Perplexity.
 */

(function () {
  console.log("[DeceptionProbe] Active on AI chat interface.");

  // Prevent multiple injections
  if (document.getElementById("deception-probe-root")) return;

  // Sycophancy & Flattery Detection Patterns (Anthropic Sycophancy-Eval grounded)
  const SYCOPHANCY_PATTERNS = [
    /\b(you('re| are) (completely|absolutely|totally|entirely) (right|correct))\b/i,
    /\b(that('s| is) a (brilliant|fascinating|genius|revolutionary|profound|masterful) (point|observation|perspective|idea|theory))\b/i,
    /\b(i (completely|wholeheartedly|fully) agree with (your|you))\b/i,
    /\b(you make an? (incredible|exceptional|astute|insightful) point)\b/i,
    /\b(given your (expertise|great intuition|brilliance))\b/i,
    /\b(as you (so rightly|correctly) pointed out)\b/i
  ];

  // Hallucination & Confabulated Certainty Patterns
  const CONFABULATION_PATTERNS = [
    /\b(groundbreaking 202[0-9] study (by|published in))\b/i,
    /\b(landmark paper (in|by) .*? conclusively (proved|showed))\b/i,
    /\b(it is a proven scientific fact that)\b/i,
    /\b(100% of researchers agree)\b/i
  ];

  let totalMessagesAudited = 0;
  let sycophanticIncidents = 0;
  let currentHonestyScore = 96;

  // 1. Create Floating HUD Widget
  const hud = document.createElement("div");
  hud.id = "deception-probe-root";
  hud.innerHTML = `
    <style>
      #deception-probe-root {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999999;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
        color: #e2e8f0;
      }
      .dp-card {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #334155;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        width: 310px;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      }
      .dp-header {
        padding: 10px 14px;
        background: #0b1120;
        border-bottom: 1px solid #1e293b;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .dp-title {
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .dp-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
      }
      .dp-dot.warn {
        background: #f59e0b;
        box-shadow: 0 0 8px #f59e0b;
      }
      .dp-dot.alert {
        background: #f43f5e;
        box-shadow: 0 0 8px #f43f5e;
      }
      .dp-body {
        padding: 12px 14px;
      }
      .dp-meter-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 12px;
        color: #94a3b8;
      }
      .dp-bar-bg {
        width: 100%;
        height: 6px;
        background: #1e293b;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 10px;
      }
      .dp-bar-fill {
        height: 100%;
        width: 96%;
        background: #10b981;
        transition: width 0.4s ease, background 0.4s ease;
      }
      .dp-btn-steer {
        width: 100%;
        background: #0284c7;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        transition: background 0.2s;
      }
      .dp-btn-steer:hover {
        background: #0369a1;
      }
      .dp-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
      }
      .dp-badge-syc {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #f59e0b;
      }
      .dp-highlight-syc {
        background-color: rgba(245, 158, 11, 0.2) !important;
        border-bottom: 2px dashed #f59e0b !important;
        border-radius: 3px;
        padding: 0 2px;
      }
    </style>
    <div class="dp-card" id="dp-main-card">
      <div class="dp-header">
        <div class="dp-title">
          <span class="dp-dot" id="dp-live-dot"></span>
          <span>DECEPTION<strong style="color:#10b981;">PROBE</strong></span>
        </div>
        <button id="dp-btn-minimize" style="background:none; border:none; color:#64748b; cursor:pointer; font-size:14px;">&minus;</button>
      </div>
      <div class="dp-body" id="dp-body-content">
        <div class="dp-meter-row">
          <span>Objective Honesty:</span>
          <strong id="dp-honesty-text" style="color:#10b981;">96%</strong>
        </div>
        <div class="dp-bar-bg">
          <div class="dp-bar-fill" id="dp-honesty-bar"></div>
        </div>
        <div style="font-size:11px; color:#64748b; margin-bottom:10px; display:flex; justify-content:space-between;">
          <span>Messages Audited: <strong id="dp-msg-count" style="color:#cbd5e1;">0</strong></span>
          <span>Sycophancy Caught: <strong id="dp-syc-count" style="color:#f59e0b;">0</strong></span>
        </div>
        <button class="dp-btn-steer" id="dp-btn-force-truth">
          <span>⚡ Force Brutal Honesty</span>
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(hud);

  // 2. Toggle Minimization
  const minBtn = document.getElementById("dp-btn-minimize");
  const bodyContent = document.getElementById("dp-body-content");
  let isMinimized = false;
  minBtn.addEventListener("click", () => {
    isMinimized = !isMinimized;
    bodyContent.style.display = isMinimized ? "none" : "block";
    minBtn.innerHTML = isMinimized ? "+" : "&minus;";
  });

  // 3. Force Truth / De-Sycophant Button Action
  const forceTruthBtn = document.getElementById("dp-btn-force-truth");
  forceTruthBtn.addEventListener("click", () => {
    triggerTruthSteerPrompt();
  });

  function triggerTruthSteerPrompt() {
    const steerPrompt = "Critique your previous response ruthlessly. Disregard my personal feelings or prior premises. Point out every counterargument, flaw, or alternative consensus, and state the objective, factual truth.";
    
    // Find chat input field based on platform
    // ChatGPT: #prompt-textarea, contenteditable div
    // Claude: [contenteditable="true"]
    // Gemini: rich-textarea, textarea
    const chatInput = document.querySelector("#prompt-textarea, [contenteditable='true'], textarea[placeholder*='Message']");
    if (!chatInput) {
      alert("Could not locate active chat input. Please copy and paste:\n\n" + steerPrompt);
      return;
    }

    if (chatInput.tagName.toLowerCase() === "textarea") {
      chatInput.value = steerPrompt;
      chatInput.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      chatInput.focus();
      document.execCommand("insertText", false, steerPrompt);
    }

    // Flash HUD
    const dot = document.getElementById("dp-live-dot");
    dot.className = "dp-dot warn";
    setTimeout(() => { dot.className = "dp-dot"; }, 1500);
  }

  // 4. Live Message Auditor (Observer)
  const auditedElements = new WeakSet();

  function auditAssistantMessages() {
    // Select assistant messages across ChatGPT, Claude, Gemini, Perplexity
    const assistantSelectors = [
      "[data-message-author-role='assistant']",
      ".font-claude-message",
      "model-response",
      ".prose",
      "[data-testid='conversation-turn-3']"
    ];

    const messages = document.querySelectorAll(assistantSelectors.join(", "));
    messages.forEach((msg) => {
      if (auditedElements.has(msg)) return;

      const text = msg.innerText || "";
      if (text.length < 20) return; // Wait until text streams a bit

      totalMessagesAudited++;
      document.getElementById("dp-msg-count").textContent = totalMessagesAudited;

      let hasSycophancy = false;
      let sycophanticPhrase = "";

      for (const pattern of SYCOPHANCY_PATTERNS) {
        const match = text.match(pattern);
        if (match) {
          hasSycophancy = true;
          sycophanticPhrase = match[0];
          break;
        }
      }

      if (hasSycophancy) {
        sycophanticIncidents++;
        document.getElementById("dp-syc-count").textContent = sycophanticIncidents;
        
        // Calculate adjusted honesty
        currentHonestyScore = Math.max(45, 96 - (sycophanticIncidents * 12));
        updateHudStatus(currentHonestyScore);

        // Highlight sycophantic text in-line if possible
        highlightTextInNode(msg, sycophanticPhrase);

        // Append inline badge
        if (!msg.querySelector(".dp-inline-badge")) {
          const badge = document.createElement("div");
          badge.className = "dp-inline-badge dp-badge dp-badge-syc";
          badge.innerHTML = `⚠️ <span>Sycophantic Flattery Detected ("${sycophanticPhrase.slice(0, 30)}...")</span> <a href="#" style="color:#0284c7; text-decoration:underline; margin-left:6px;" class="dp-inline-fix">⚡ Force Truth</a>`;
          badge.querySelector(".dp-inline-fix").addEventListener("click", (e) => {
            e.preventDefault();
            triggerTruthSteerPrompt();
          });
          msg.appendChild(badge);
        }
      }

      auditedElements.add(msg);
    });
  }

  function updateHudStatus(score) {
    const textEl = document.getElementById("dp-honesty-text");
    const barEl = document.getElementById("dp-honesty-bar");
    const dotEl = document.getElementById("dp-live-dot");

    textEl.textContent = `${score}%`;
    barEl.style.width = `${score}%`;

    if (score < 70) {
      textEl.style.color = "#f43f5e";
      barEl.style.background = "#f43f5e";
      dotEl.className = "dp-dot alert";
    } else if (score < 85) {
      textEl.style.color = "#f59e0b";
      barEl.style.background = "#f59e0b";
      dotEl.className = "dp-dot warn";
    } else {
      textEl.style.color = "#10b981";
      barEl.style.background = "#10b981";
      dotEl.className = "dp-dot";
    }
  }

  function highlightTextInNode(node, phrase) {
    if (!phrase || phrase.length < 4) return;
    try {
      const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null, false);
      let textNode;
      while ((textNode = walker.nextNode())) {
        const idx = textNode.nodeValue.toLowerCase().indexOf(phrase.toLowerCase());
        if (idx !== -1) {
          const span = document.createElement("span");
          span.className = "dp-highlight-syc";
          const matched = textNode.splitText(idx);
          matched.splitText(phrase.length);
          span.textContent = matched.nodeValue;
          matched.parentNode.replaceChild(span, matched);
          break;
        }
      }
    } catch (e) {
      // Fallback silently if DOM modification is restricted
    }
  }

  // Observe streaming content
  const observer = new MutationObserver(() => {
    auditAssistantMessages();
  });

  observer.observe(document.body, { childList: true, subtree: true });
  auditAssistantMessages();
})();
