# ArtVision Frontend Dashboard

A responsive, accessible, dark/light-friendly React + Tailwind CSS results dashboard for the ArtVision explainable artwork analysis framework.

---

## Features

- **Multi-Task Predictions**:
  - Primary art style and derived art movement
  - Artist attribution with calibrated confidence and prominent "Low confidence" warning for uncertain cases
  - Genre classification with neutral "Not available" state for unannotated pieces
  - Affective mood distribution (5 classes) with top mood highlighted
  - Authenticity detection with an interactive SVG speedometer gauge (Human-Made vs AI-Generated) and explicit generator caveat disclaimer
  - "Coming soon" placeholder card for planned Era and Technique heads
  - CIELAB color palette swatches with hex click-to-copy, temperature balance bar (warm vs cool score), and tone descriptors (brightness, saturation, pastel)
- **Explainability (Grad-CAM)**:
  - Interactive visualizer with head selector tabs (style, mood, authenticity, artist)
  - Display modes: Original artwork, Heatmap, or Screen-blended Overlay
  - Adjustable overlay opacity slider
  - Non-causal interpretation disclaimer: *"The model focused mostly on these regions"*
- **Similar Artworks Gallery**:
  - Grid of visually similar artworks with title, artist, style, and similarity match percentage
- **Mock Mode Strategy**:
  - 5 comprehensive mock fixtures covering confident Baroque, uncertain artist, AI-generated, genre null + color error, and high uncertainty across all heads
  - Simulated latency (configurable from 0ms to 1.5s)
  - Built-in error simulation (500 server error, network disconnect, 422 invalid format) with retry button
  - Procedural inline SVG placeholders: zero external dependencies, zero copyrighted artwork
- **Accessibility & UX**:
  - Full keyboard navigation and visible focus rings
  - Semantic ARIA attributes (`role="meter"`, `aria-valuenow`, `aria-valuemin="0"`, `aria-valuemax="100"`, `aria-label`)
  - No color-only meaning; text and badges accompany all visual indicators
  - Dark mode with local storage persistence and system preference sync
  - Drag-and-drop file upload with client-side type (JPG, PNG, WebP) and size (<= 10MB) validation

---

## API Contract

The complete, versioned API specification matching `predict.py` is documented in:
```
frontend/src/api/contract.md
```
TypeScript interfaces and types are located in:
```
frontend/src/api/types.ts
```

---

## Getting Started

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run with Live Backend (Default)
By default, the application runs connected to the live FastAPI backend (`http://localhost:8000`) with `VITE_USE_MOCK=false`:
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Run in Mock Mode (Offline / Testing)
To run offline or test edge-case fixtures:
- In `.env` or `.env.local`, set `VITE_USE_MOCK=true`
- Or use the "Mock Mode" toggle button directly in the app header
- Switch between 5 mock fixtures or simulate network errors and latency via the UI controls.

### 4. Running Tests
Run the Vitest test suite:
```bash
npm run test:run
```

To run tests in watch mode:
```bash
npm run test
```

### 5. Production Build
```bash
npm run build
```
Build output is saved to `frontend/dist/`.
