// Mock analysis results — used ONLY as a temporary frontend development fallback.
// The shape here mirrors exactly what /api/analyze is expected to return from
// the real neurovision_pipeline.py (YOLO26 + ResNet50 + Grad-CAM).

const CLASS_POOL = ['Glioma', 'Meningioma', 'Pituitary'];

function randomFloat(min, max, digits = 1) {
  return Number((Math.random() * (max - min) + min).toFixed(digits));
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function buildMockResult({ agree = null } = {}) {
  const tumorClass = pick(CLASS_POOL);
  const disagree = agree === null ? Math.random() < 0.12 : !agree;
  const resnetClass = disagree ? pick(CLASS_POOL.filter((c) => c !== tumorClass)) : tumorClass;

  const yoloConfidence = randomFloat(88, 98.5);
  const resnetConfidence = randomFloat(84, 97);

  const relativeArea = randomFloat(0.8, 9.5, 2);
  const areaCategory = relativeArea < 2 ? 'Small' : relativeArea < 5 ? 'Moderate' : 'Large';
  const horizontal = pick(['Left', 'Center', 'Right']);
  const vertical = pick(['Upper', 'Middle', 'Lower']);
  const locationType = horizontal === 'Center' && vertical === 'Middle' ? 'Central' : 'Peripheral';

  let riskScore = 0;
  if (areaCategory === 'Moderate') riskScore += 1;
  if (areaCategory === 'Large') riskScore += 2;
  if (locationType === 'Peripheral') riskScore += 1;
  const riskLevel = riskScore >= 3 ? 'Higher' : riskScore >= 1 ? 'Moderate' : 'Lower';

  const factors = [];
  factors.push(`${areaCategory} relative tumor area (${relativeArea}%)`);
  factors.push(`${locationType} location (${vertical}-${horizontal})`);

  return {
    image: {
      name: 'uploaded-mri.jpg',
      width: 512,
      height: 512,
    },
    detections: [
      {
        id: 'det-1',
        class: tumorClass,
        confidence: yoloConfidence,
        box: { x: 0.32, y: 0.28, width: 0.28, height: 0.24 },
      },
    ],
    predictions: {
      yolo: { class: tumorClass, confidence: yoloConfidence },
      resnet: { class: resnetClass, confidence: resnetConfidence },
    },
    modelComparison: {
      agree: !disagree,
      yolo: { class: tumorClass, confidence: yoloConfidence },
      resnet: { class: resnetClass, confidence: resnetConfidence },
    },
    tumorCharacteristics: {
      width: randomFloat(28, 96, 0),
      height: randomFloat(24, 88, 0),
      area: randomFloat(900, 5200, 0),
      relativeArea,
      areaCategory,
      center: { x: randomFloat(0.3, 0.7, 2), y: randomFloat(0.3, 0.7, 2) },
      horizontal,
      vertical,
      combinedLocation: `${vertical}-${horizontal}`,
    },
    riskAnalysis: {
      areaCategory,
      locationType,
      riskLevel,
      riskScore,
      factors,
      disclaimer:
        'This research-defined stratification is based on dataset-derived image characteristics and is not a clinical diagnosis, prognosis, or medical risk assessment.',
    },
    gradcam: {
      predictedClass: resnetClass,
      confidence: resnetConfidence,
      available: true,
    },
    finalClass: disagree ? null : tumorClass,
    timestamp: new Date().toISOString(),
  };
}
