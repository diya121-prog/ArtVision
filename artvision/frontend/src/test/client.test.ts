import { describe, it, expect, beforeEach } from 'vitest';
import {
  predictArtwork,
  validateArtworkFile,
  isMockMode,
  getApiBaseUrl,
  setSimulatedLatency,
  resolveApiStaticUrl,
} from '../api/client';

describe('API Client and Mock Mode Suite', () => {
  beforeEach(() => {
    setSimulatedLatency(0); // Run instantly in tests
  });

  describe('Validation', () => {
    it('accepts valid JPEG, PNG, and WebP images within 15MB', () => {
      const jpg = new File(['dummy content'], 'art.jpg', { type: 'image/jpeg' });
      const png = new File(['dummy content'], 'art.png', { type: 'image/png' });
      const webp = new File(['dummy content'], 'art.webp', { type: 'image/webp' });

      expect(validateArtworkFile(jpg).valid).toBe(true);
      expect(validateArtworkFile(png).valid).toBe(true);
      expect(validateArtworkFile(webp).valid).toBe(true);
    });

    it('rejects unsupported file formats like SVG, PDF, or text', () => {
      const svg = new File(['<svg></svg>'], 'vector.svg', { type: 'image/svg+xml' });
      const pdf = new File(['%PDF'], 'document.pdf', { type: 'application/pdf' });

      const resSvg = validateArtworkFile(svg);
      expect(resSvg.valid).toBe(false);
      expect(resSvg.error).toMatch(/Unsupported file format/i);

      const resPdf = validateArtworkFile(pdf);
      expect(resPdf.valid).toBe(false);
      expect(resPdf.error).toMatch(/Unsupported file format/i);
    });

    it('rejects files exceeding 15MB limit', () => {
      const bigBuffer = new Uint8Array(16 * 1024 * 1024);
      const bigFile = new File([bigBuffer], 'huge.png', { type: 'image/png' });

      const res = validateArtworkFile(bigFile);
      expect(res.valid).toBe(false);
      expect(res.error).toMatch(/File is too large/i);
    });
  });

  describe('Mock Predictions and Scenarios', () => {
    const validFile = new File(['content'], 'sample.jpg', { type: 'image/jpeg' });

    it('returns confident Baroque fixture when mock is active', async () => {
      const res = await predictArtwork(validFile, {
        forceMock: true,
        mockScenario: 'confidentBaroque',
        simulatedLatencyMs: 0,
      });
      expect(res.style.prediction).toBe('Baroque');
      expect(res.artist?.prediction).toBe('Rembrandt van Rijn');
      expect(res.authenticity?.is_ai_generated).toBe(false);
    });

    it('returns uncertain artist fixture when requested', async () => {
      const res = await predictArtwork(validFile, {
        forceMock: true,
        mockScenario: 'uncertainArtist',
        simulatedLatencyMs: 0,
      });
      expect(res.artist?.uncertain).toBe(true);
      expect(res.artist?.prediction).toBe('Claude Monet');
      expect(res.genre).toBeNull();
    });

    it('returns AI generated fixture with high probability', async () => {
      const res = await predictArtwork(validFile, {
        forceMock: true,
        mockScenario: 'aiGenerated',
        simulatedLatencyMs: 0,
      });
      expect(res.authenticity?.is_ai_generated).toBe(true);
      expect(res.authenticity?.probability_ai).toBeGreaterThan(0.9);
      expect(res.artist).toBeNull();
    });

    it('returns genre null and color error fixture', async () => {
      const res = await predictArtwork(validFile, {
        forceMock: true,
        mockScenario: 'genreNullColorError',
        simulatedLatencyMs: 0,
      });
      expect(res.genre).toBeNull();
      expect(res.color).toBeNull();
      expect(res.color_error).toBeDefined();
    });

    it('throws network error when scenario is errorNetwork', async () => {
      await expect(
        predictArtwork(validFile, { forceMock: true, mockScenario: 'errorNetwork', simulatedLatencyMs: 0 })
      ).rejects.toThrow(/Network error/i);
    });

    it('throws 500 error when scenario is error500', async () => {
      await expect(
        predictArtwork(validFile, { forceMock: true, mockScenario: 'error500', simulatedLatencyMs: 0 })
      ).rejects.toThrow(/HTTP 500/i);
    });

    it('throws 422 error when scenario is errorInvalidFile', async () => {
      await expect(
        predictArtwork(validFile, { forceMock: true, mockScenario: 'errorInvalidFile', simulatedLatencyMs: 0 })
      ).rejects.toThrow(/HTTP 422/i);
    });
  });

  describe('Configuration Helpers', () => {
    it('defaults to mock mode disabled when VITE_USE_MOCK=false', () => {
      expect(isMockMode()).toBe(false);
    });

    it('returns configured API base URL or localhost:8000', () => {
      expect(getApiBaseUrl()).toMatch(/http:\/\//);
    });

    it('resolves relative static URLs to full backend URLs', () => {
      const res = resolveApiStaticUrl('/static/images/wikiart_000001.jpg');
      expect(res).toBe('http://localhost:8000/static/images/wikiart_000001.jpg');
    });
  });
});
