// ============================================================================
// NeuroVision API Service Layer
// ============================================================================

import { buildMockResult } from './mockData';

// ============================================================================
// CONFIG
// ============================================================================

export const USE_MOCK = false;

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  'http://localhost:8000';


// ============================================================================
// PIPELINE STAGES
// ============================================================================

export const PIPELINE_STAGES = [
  {
    key: 'preprocess',
    label: 'Image preprocessing',
  },
  {
    key: 'yolo',
    label: 'YOLO26 tumor localization',
  },
  {
    key: 'characteristics',
    label: 'Tumor characteristics',
  },
  {
    key: 'resnet',
    label: 'ResNet50 classification',
  },
  {
    key: 'risk',
    label: 'Research risk analysis',
  },
  {
    key: 'gradcam',
    label: 'Grad-CAM generation',
  },
];


// ============================================================================
// ANALYZE MRI
// ============================================================================

export async function analyzeMRI(
  file,
  onStageChange = () => {}
) {

  if (USE_MOCK) {
    return mockAnalyze(
      file,
      onStageChange
    );
  }


  // ----------------------------------------------------------
  // Build request
  // ----------------------------------------------------------

  const formData = new FormData();

  formData.append(
    'file',
    file
  );


  // ----------------------------------------------------------
  // Send to FastAPI
  // ----------------------------------------------------------

  const response = await fetch(
    `${API_BASE_URL}/api/analyze`,
    {
      method: 'POST',
      body: formData,
    }
  );


  // ----------------------------------------------------------
  // Handle errors
  // ----------------------------------------------------------

  if (!response.ok) {

    let message =
      `Analysis request failed: ${response.status}`;

    try {

      const errorData =
        await response.json();

      if (errorData.detail) {
        message = errorData.detail;
      }

    } catch {
      // Ignore JSON parsing errors
    }

    throw new Error(message);
  }


  // ----------------------------------------------------------
  // Read result
  // ----------------------------------------------------------

  const data =
    await response.json();


  // ----------------------------------------------------------
  // Simulate frontend stage completion
  // ----------------------------------------------------------

  for (const stage of PIPELINE_STAGES) {

    onStageChange(
      stage.key
    );
  }


  // ----------------------------------------------------------
  // IMPORTANT
  // Preserve the backend analysis ID
  // ----------------------------------------------------------

  if (
    !data.analysisId &&
    data.id
  ) {

    data.analysisId =
      data.id;
  }


  return data;
}


// ============================================================================
// MOCK ANALYSIS
// ============================================================================

async function mockAnalyze(
  file,
  onStageChange
) {

  const previewUrl =
    URL.createObjectURL(file);


  const dims =
    await readImageDimensions(
      previewUrl
    );


  for (
    const stage of PIPELINE_STAGES
  ) {

    await wait(
      420 +
      Math.random() * 260
    );

    onStageChange(
      stage.key
    );
  }


  const result =
    buildMockResult();


  result.image = {

    name:
      file.name,

    width:
      dims.width,

    height:
      dims.height,

    previewUrl
  };


  return result;
}


// ============================================================================
// WAIT
// ============================================================================

function wait(ms) {

  return new Promise(
    (resolve) =>
      setTimeout(
        resolve,
        ms
      )
  );
}


// ============================================================================
// IMAGE DIMENSIONS
// ============================================================================

function readImageDimensions(
  src
) {

  return new Promise(
    (resolve) => {

      const img =
        new Image();


      img.onload = () => {

        resolve({

          width:
            img.naturalWidth,

          height:
            img.naturalHeight
        });
      };


      img.onerror = () => {

        resolve({

          width: null,

          height: null
        });
      };


      img.src = src;
    }
  );
}


// ============================================================================
// GENERATE PDF REPORT
// ============================================================================

export async function generateReport(
  analysisId
) {

  if (USE_MOCK) {

    throw new Error(
      'Report generation requires the FastAPI backend.'
    );
  }


  // ----------------------------------------------------------
  // Accept either:
  //
  // generateReport("abc123")
  //
  // OR
  //
  // generateReport(result)
  //
  // This prevents [object Object] errors.
  // ----------------------------------------------------------

  let id =
    analysisId;


  if (
    typeof analysisId === 'object' &&
    analysisId !== null
  ) {

    id =
      analysisId.analysisId ||
      analysisId.id;
  }


  if (!id) {

    throw new Error(
      'Analysis ID is missing. Please analyze the MRI again.'
    );
  }


  console.log(
    'Generating report for analysis:',
    id
  );


  // ----------------------------------------------------------
  // POST request
  // ----------------------------------------------------------

  const response =
    await fetch(
      `${API_BASE_URL}/api/report/${encodeURIComponent(id)}`,
      {
        method: 'POST',
      }
    );


  // ----------------------------------------------------------
  // Error handling
  // ----------------------------------------------------------

  if (!response.ok) {

    let errorMessage =
      `Report generation failed: ${response.status}`;

    try {

      const errorData =
        await response.json();

      if (errorData.detail) {
        errorMessage =
          errorData.detail;
      }

    } catch {
      // Response wasn't JSON
    }


    console.error(
      'Report error:',
      errorMessage
    );


    throw new Error(
      errorMessage
    );
  }


  // ----------------------------------------------------------
  // Return PDF blob
  // ----------------------------------------------------------

  const blob =
    await response.blob();


  console.log(
    'PDF report generated successfully.'
  );


  return blob;
}


// ============================================================================
// ASK AI ASSISTANT
// ============================================================================

export async function askAssistant(
  question,
  context
) {

  if (USE_MOCK) {

    await wait(
      500 +
      Math.random() * 400
    );

    return mockAssistantAnswer(
      question,
      context
    );
  }


  // ----------------------------------------------------------
  // Make sure question is a string
  // ----------------------------------------------------------

  const cleanQuestion =
    String(
      question || ''
    ).trim();


  if (!cleanQuestion) {

    throw new Error(
      'Please enter a question.'
    );
  }


  // ----------------------------------------------------------
  // Make sure context exists
  // ----------------------------------------------------------

  const cleanContext =
    context || {};


  console.log(
    'AI question:',
    cleanQuestion
  );

  console.log(
    'AI context:',
    cleanContext
  );


  // ----------------------------------------------------------
  // Send to FastAPI
  // ----------------------------------------------------------

  const response =
    await fetch(
      `${API_BASE_URL}/api/assistant`,
      {
        method: 'POST',

        headers: {
          'Content-Type':
            'application/json',
        },

        body: JSON.stringify({

          question:
            cleanQuestion,

          context:
            cleanContext
        }),
      }
    );


  // ----------------------------------------------------------
  // Handle errors
  // ----------------------------------------------------------

  if (!response.ok) {

    let errorMessage =
      'Assistant request failed.';


    try {

      const errorData =
        await response.json();

      if (errorData.detail) {

        errorMessage =
          errorData.detail;
      }

    } catch {
      // Ignore parsing errors
    }


    console.error(
      'Assistant error:',
      errorMessage
    );


    throw new Error(
      errorMessage
    );
  }


  // ----------------------------------------------------------
  // Read response
  // ----------------------------------------------------------

  const data =
    await response.json();


  console.log(
    'AI answer:',
    data.answer
  );


  return data.answer;
}


// ============================================================================
// MOCK ASSISTANT
// ============================================================================

function mockAssistantAnswer(
  question,
  context
) {

  const q =
    question.toLowerCase();


  if (
    q.includes('grad-cam') ||
    q.includes('gradcam')
  ) {

    return (
      "Grad-CAM highlights the image regions "
      + "that most influenced ResNet50's prediction. "
      + "It is an interpretability tool and does not "
      + "prove that a highlighted region is medically significant."
    );
  }


  if (
    q.includes('yolo') &&
    q.includes('resnet')
  ) {

    return (
      "YOLO26 performs tumor localization on the "
      + "MRI, while ResNet50 classifies the detected "
      + "tumor crop. Comparing both predictions gives "
      + "an additional consistency check."
    );
  }


  if (
    q.includes('confidence')
  ) {

    const predictions =
      context?.predictions;


    if (predictions) {

      return (
        `YOLO26 confidence is `
        + `${predictions.yolo.confidence}%, `
        + `while ResNet50 confidence is `
        + `${predictions.resnet.confidence}%. `
        + `These scores represent model confidence, `
        + `not clinical certainty.`
      );
    }


    return (
      'Upload and analyze an MRI to see '
      + 'model confidence scores.'
    );
  }


  if (
    q.includes('detect')
  ) {

    if (
      context?.predictions
    ) {

      return (
        `YOLO26 localized a region classified as `
        + `${context.predictions.yolo.class}.`
      );
    }


    return (
      'No analysis has been run yet.'
    );
  }


  if (
    q.includes('risk factor')
  ) {

    return (
      'General research literature identifies '
      + 'several factors that may be associated with '
      + 'brain tumor risk, but risk factors do not '
      + 'determine whether an individual has a tumor.'
    );
  }


  if (
    q.includes('region') &&
    q.includes('influenc')
  ) {

    return (
      "The Grad-CAM visualization shows image "
      + "regions contributing to ResNet50's prediction. "
      + "Warmer regions indicate stronger attribution."
    );
  }


  return (
    "I can explain the current detection, "
    + "confidence scores, YOLO26 vs ResNet50 "
    + "comparison, tumor characteristics, "
    + "research risk analysis, or Grad-CAM."
  );
}