import { PredictResponse, MockScenarioKey, SimilarArtwork, HeadExplanation } from './types';
import { mockFixtures } from './mocks/fixtures';

export interface PredictClientOptions {
  mockScenario?: MockScenarioKey;
  simulatedLatencyMs?: number;
  forceMock?: boolean;
  explain?: boolean;
  explainTask?: string;
  similarTopK?: number;
}

// Configurable latency for mock testing (in ms)
let globalSimulatedLatencyMs = 400;

export function setSimulatedLatency(ms: number) {
  globalSimulatedLatencyMs = Math.max(0, ms);
}

export function getSimulatedLatency(): number {
  return globalSimulatedLatencyMs;
}

/**
 * Returns whether mock mode is currently enabled.
 * Defaults to FALSE for local runs per user specification.
 * Enabled only if VITE_USE_MOCK is explicitly true or 'true'.
 */
export function isMockMode(): boolean {
  const envVal = import.meta.env.VITE_USE_MOCK;
  if (envVal === true || envVal === 'true') {
    return true;
  }
  return false;
}

/**
 * Gets the backend API base URL from env or default http://localhost:8000
 */
export function getApiBaseUrl(): string {
  const url = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  return url.replace(/\/+$/, '');
}

/**
 * Helper to ensure static image URLs served by the API have the full host prefix
 */
export function resolveApiStaticUrl(path: string | undefined): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('data:')) {
    return path;
  }
  const base = getApiBaseUrl();
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${cleanPath}`;
}

/**
 * Validates the artwork image before upload:
 * - Allowed types: image/jpeg, image/png, image/webp
 * - Max size: 15MB (aligned with backend 15MB limit)
 */
export function validateArtworkFile(file: File): { valid: boolean; error?: string } {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: `Unsupported file format (${file.type || file.name.split('.').pop() || 'unknown'}). Please upload a JPEG, PNG, or WebP image.`,
    };
  }

  const maxSizeBytes = 15 * 1024 * 1024; // 15MB
  if (file.size > maxSizeBytes) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `File is too large (${sizeMb} MB). Maximum supported size is 15 MB.`,
    };
  }

  return { valid: true };
}

/**
 * Performs artwork prediction.
 * In mock mode: returns chosen scenario fixture or simulated error after delay.
 * In real mode: executes POST /predict with Grad-CAM and parallel POST /similar retrieval.
 */
export async function predictArtwork(
  file: File,
  options?: PredictClientOptions
): Promise<PredictResponse> {
  const validation = validateArtworkFile(file);
  if (!validation.valid) {
    throw new Error(validation.error);
  }

  const useMock = options?.forceMock !== undefined ? options.forceMock : isMockMode();
  const latency = options?.simulatedLatencyMs ?? globalSimulatedLatencyMs;
  const scenario: MockScenarioKey = options?.mockScenario || 'confidentBaroque';

  if (useMock) {
    // Simulate network delay
    if (latency > 0) {
      await new Promise((resolve) => setTimeout(resolve, latency));
    }

    // Handle mock failure modes
    if (scenario === 'errorNetwork') {
      throw new Error('Network error: Unable to connect to ArtVision inference server.');
    }
    if (scenario === 'error500') {
      throw new Error('HTTP 500: Internal server error occurred during PyTorch inference.');
    }
    if (scenario === 'errorInvalidFile') {
      throw new Error('HTTP 422: The server could not process the image stream.');
    }

    const fixture = mockFixtures[scenario];
    if (fixture) {
      return fixture;
    }

    return mockFixtures.confidentBaroque;
  }

  // --- Real Backend Execution ---
  const baseUrl = getApiBaseUrl();
  const explain = options?.explain !== false; // Default to true so user gets Grad-CAM
  const explainTask = options?.explainTask || 'style';
  const topK = options?.similarTopK || 4;

  const predictUrl = `${baseUrl}/predict?explain=${explain}&explain_task=${encodeURIComponent(explainTask)}`;
  const similarUrl = `${baseUrl}/similar?top_k=${topK}`;

  // Prepare FormData
  const predictFormData = new FormData();
  predictFormData.append('image', file);

  const similarFormData = new FormData();
  similarFormData.append('image', file);

  // Execute predict and similar concurrently
  const [predictResult, similarResult] = await Promise.allSettled([
    fetch(predictUrl, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      body: predictFormData,
    }),
    fetch(similarUrl, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      body: similarFormData,
    }),
  ]);

  if (predictResult.status === 'rejected') {
    throw new Error(`Failed to connect to ArtVision API at ${baseUrl}: ${predictResult.reason.message}`);
  }

  const response = predictResult.value;
  if (!response.ok) {
    let errorDetail = `API Error ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && (errJson.detail || errJson.error)) {
        errorDetail = errJson.detail || errJson.error;
      }
    } catch {
      // Non-JSON body
    }
    throw new Error(errorDetail);
  }

  const data: PredictResponse = await response.json();

  // Normalize explanation URLs from API
  if (data.explanation) {
    const rawExp = data.explanation as Record<string, unknown>;
    if (typeof rawExp.overlay_url === 'string') {
      rawExp.overlay_url = resolveApiStaticUrl(rawExp.overlay_url as string);
    }
    if (typeof rawExp.heatmap_url === 'string') {
      rawExp.heatmap_url = resolveApiStaticUrl(rawExp.heatmap_url as string);
    }
  }

  // Merge results from /similar endpoint if available
  if (similarResult.status === 'fulfilled' && similarResult.value.ok) {
    try {
      const simJson = await similarResult.value.json();
      if (simJson && Array.isArray(simJson.results)) {
        data.similar = simJson.results.map((r: Record<string, unknown>): SimilarArtwork => ({
          image_id: String(r.image_id || ''),
          image_url: resolveApiStaticUrl(String(r.image_url || '')),
          title: String(r.title || r.image_id || 'Catalog Artwork'),
          artist: String(r.artist || 'Unknown Artist'),
          style: String(r.style || 'Unspecified'),
          genre: r.genre ? String(r.genre) : undefined,
          similarity: Number(r.similarity || 0),
          is_ai_generated: Boolean(r.is_ai_generated),
        }));
      }
    } catch (e) {
      console.warn('Could not parse similarity search response:', e);
    }
  }

  return data;
}

/**
 * Specifically fetches an updated Grad-CAM visual explanation for a different head
 */
export async function fetchExplanationForHead(
  file: File,
  head: string
): Promise<HeadExplanation | null> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/predict?explain=true&explain_task=${encodeURIComponent(head)}`;

  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(url, {
    method: 'POST',
    headers: { Accept: 'application/json' },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to generate explanation for ${head}`);
  }

  const data = await response.json();
  if (data.explanation) {
    const exp = data.explanation;
    if (exp.overlay_url) {
      exp.overlay_url = resolveApiStaticUrl(exp.overlay_url);
    }
    return exp as HeadExplanation;
  }

  return null;
}
