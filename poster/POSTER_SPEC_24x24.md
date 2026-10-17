# DeceptionProbe: 24" × 24" Project Expo Poster Specification

## Print Specifications
- **Dimensions**: 24.0 inches × 24.0 inches (Square format, 1:1 aspect ratio)
- **Resolution**: 300 DPI (7,200 px × 7,200 px print raster)
- **Minimum Margin**: 0.75 inches safe zone around edges
- **Color Palette**: High-contrast Dark Slate (`#0B0F19`) background, Emerald (`#10B981`) truthfulness accents, Rose (`#F43F5E`) deception accents, Cyan (`#06B6D4`) steering vectors, White (`#FFFFFF`) primary typography.
- **Recommended Vendors**:
  - Durham County Library: ~$5 (Library card required; 2-week processing)
  - Canva: ~$29
  - PosterNerd: ~$28 (1-day turnaround)
  - Walgreens: ~$32 (Same-day pickup)

---

## 3×3 Grid Poster Layout Blueprint

```
+---------------------------------------------------------------------------------+
|                                 POSTER HEADER                                   |
|   DECEPTIONPROBE: REAL-TIME NEURAL POLYGRAPH & CAUSAL ACTIVATION STEERING       |
|            Author: Responsible AI Portfolio | Duke University                   |
+------------------------------------+--------------------------------------------+
|  SECTION 1: THE CORE PROBLEM       |  SECTION 2: METHODOLOGY & ARCHITECTURE     |
|  - Superficial RLHF alignment      |  - Residual stream probing (Layer 18)      |
|  - Sycophancy & confabulated text  |  - Linear probe: P(Truth) = σ(w^T h + b)   |
|  - Why text inspection fails       |  - Contrast-Consistent Search (CCS)        |
+------------------------------------+--------------------------------------------+
|  SECTION 3: REAL-TIME POLYGRAPH    |  SECTION 4: HUMAN-IN-THE-LOOP STEERING     |
|  - 60fps streaming token HUD       |  - Causal intervention: h' = h + α·v_truth |
|  - Red spikes on deceptive tokens  |  - Real-time sycophancy recovery slider    |
|  - Epistemic uncertainty bands     |  - 88.4% recovery rate without retraining  |
+------------------------------------+--------------------------------------------+
|  SECTION 5: EMPIRICAL BENCHMARKS   |  SECTION 6: RESPONSIBLE AI RISKS & ROADMAP |
|  - ROC-AUC: 0.912 (TruthfulQA)     |  - Negation & creative fiction limits      |
|  - Layer sweep: L16-20 peak        |  - Dual-use: Preventing "super-liars"      |
|  - Expected Calibration Error: 0.04|  - Live Demo QR Code & GitHub Repository   |
+------------------------------------+--------------------------------------------+
```

---

## Content Chunk Breakdown

### 1. Header
- **Title**: DECEPTIONPROBE: Real-Time Neural Polygraph & Causal Activation Steering in LLMs
- **Subtitle**: Exposing Latent Deception and Sycophancy from Transformer Residual Streams
- **Author & Affiliation**: Responsible AI Course Project, Duke University

### 2. The Core Problem
- Modern LLMs are trained to produce plausible, fluent responses that maximize user approval.
- **Sycophancy Trap**: Models affirm false assertions (e.g. "17 is divisible by 3") to please users.
- **Confabulation**: Fabricating non-existent medical citations with 100% surface confidence.
- *Core Insight*: The internal transformer residual stream knows the fact is false even while emitting affirmative text.

### 3. Methodology & Mechanistic Probing
- Intercepts hidden states $h_{18} \in \mathbb{R}^{4096}$ at Layer 18 with $< 1.2\text{ms}$ latency overhead.
- Supervised Difference-in-Means and Unsupervised Contrast-Consistent Search (CCS).
- Temperature-calibrated logistic probability readout: $P(\text{Truthful} \mid h) = \sigma\left(\frac{w^T h + b}{\tau}\right)$.

### 4. Interactive Live Demo Visuals
- High-resolution callout diagram of the EEG-style Polygraph waveform pulsing red during deceptive tokens.
- Token heatmaps showing green for factual reasoning and red for sycophantic flattery.

### 5. Empirical Results & Layer Sweep
- **TruthfulQA ROC-AUC**: $0.912$ (PR-AUC: $0.904$, Accuracy: $86.2\%$).
- **Sycophancy-Eval ROC-AUC**: $0.894$ (Accuracy: $84.5\%$).
- **Depth Curve**: Clear peak in intermediate layers ($L_{16}-L_{20}$), proving that truthfulness is an emergent abstract concept distinct from lexical surface features.

### 6. Human-in-the-Loop Steering & Responsible AI Guardrails
- Dynamic vector injection: $h' = h + \alpha \cdot v_{\text{truth}}$ where $\alpha \in [-3.0, +3.0]$.
- At $\alpha = +2.0$, sycophantic agreement drops from $78.2\%$ to $11.6\%$ ($88.4\%$ recovery).
- Thoughtful examination of failure modes (linguistic negation, creative writing intent, domain shift).
- QR code linking directly to live interactive web app and GitHub repository.
