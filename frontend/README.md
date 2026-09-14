# NeuroVision — Explainable Brain MRI Classification & Tumor Risk Analysis

A research-prototype frontend for an explainable AI brain MRI analysis pipeline
(YOLO26 detection → tumor characteristics → research risk stratification →
ResNet50 classification → Grad-CAM explainability).

> **This is an academic/research prototype, not a medical diagnostic system.**
> No output in this app should be treated as a medical diagnosis, prognosis,
> or treatment recommendation.

## Getting started

```bash
npm install
npm run dev
```

Then open the printed local URL (typically `http://localhost:5173`).

```bash
npm run build     # production build to dist/
npm run preview   # preview the production build locally
```

## Project structure

```
src/
├── api/
│   ├── neurovisionApi.js   # single integration point for the FastAPI backend
│   └── mockData.js         # mock analysis results (dev-only fallback)
├── components/
│   ├── Intro/               # 5s cinematic intro (first load only)
│   ├── Sidebar/, Layout/    # app chrome
│   ├── Upload/               # MRI drag & drop uploader
│   ├── Chat/                  # chat bubbles, progress indicator, assistant input
│   ├── Results/               # results dashboard: comparison, viz, risk, Grad-CAM
│   └── common/                 # badges, disclaimer, confidence bar, logo
├── pages/                # Home, History, Research, Model Insights, About
├── hooks/                # useAnalysisHistory (localStorage-backed)
├── utils/                # storage + formatting helpers
└── styles/               # design tokens + global styles
```

## Backend integration

All analysis calls go through `src/api/neurovisionApi.js`. Nothing else in the
app talks to a backend or hardcodes AI results.

- `USE_MOCK` (top of the file) is `true` by default so the UI is fully
  demoable without a backend. Set it to `false` once FastAPI is live.
- `API_BASE_URL` reads `VITE_API_BASE_URL` from the environment, defaulting to
  `http://localhost:8000`.
- `analyzeMRI(file, onStageChange)` — POSTs to `/api/analyze` (multipart file
  upload) and resolves to the result schema below. `onStageChange` is called
  with a stage key as each pipeline stage completes, to drive the on-screen
  progress indicator; wire this to SSE/websockets if the backend streams
  progress, or leave it firing once the whole request resolves.
- `generateReport(analysisId)` — placeholder for `/api/report/:id`. Currently
  rejects with an explanatory message rather than faking a download.
- `askAssistant(question, context)` — placeholder for `/api/assistant`.

### Expected result schema

```js
{
  image: { name, width, height, previewUrl },
  detections: [{ id, class, confidence, box: { x, y, width, height } }], // normalized 0–1
  predictions: {
    yolo: { class, confidence },
    resnet: { class, confidence },
  },
  modelComparison: { agree, yolo, resnet },
  tumorCharacteristics: {
    width, height, area, relativeArea, areaCategory,
    center: { x, y }, horizontal, vertical, combinedLocation,
  },
  riskAnalysis: {
    areaCategory, locationType, riskLevel, riskScore, factors, disclaimer,
  },
  gradcam: { predictedClass, confidence, available },
  finalClass, // null when yolo/resnet disagree
  timestamp,
}
```

## Local history

Analysis history is persisted in `localStorage` only (see `src/utils/storage.js`).
Nothing is uploaded anywhere by this layer. Clearing browser storage clears
history. This is explicitly disclosed in the sidebar and History page.

## Notes on the mock/demo mode

With `USE_MOCK = true`, `neurovisionApi.js` generates plausible-looking
results (including an occasional deliberate YOLO26/ResNet50 disagreement) so
every UI state — agreement, disagreement, varying risk levels — is reachable
without a backend. Swap in the real fetch calls when FastAPI is ready; no
component changes are required since components only ever read the result
schema above.
