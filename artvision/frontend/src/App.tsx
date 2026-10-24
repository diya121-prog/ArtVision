import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/common/Header';
import { UploadArea } from './components/upload/UploadArea';
import { ResultsDashboard } from './components/dashboard/ResultsDashboard';
import { DashboardSkeleton } from './components/common/Skeleton';
import { PredictResponse } from './api/types';
import {
  predictArtwork,
  fetchExplanationForHead,
  getApiBaseUrl,
} from './api/client';
import { AlertTriangle, RefreshCw, Image as ImageIcon } from 'lucide-react';

export const App: React.FC = () => {
  // Theme state
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('artvision_theme');
      if (saved) return saved === 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  // Analysis state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHead, setIsLoadingHead] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionData, setPredictionData] = useState<PredictResponse | null>(null);

  // Sync dark mode class
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('artvision_theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('artvision_theme', 'light');
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  // Perform multi-task live prediction
  const runPrediction = useCallback(
    async (file: File) => {
      setIsLoading(true);
      setError(null);

      try {
        const result = await predictArtwork(file, {
          forceMock: false,
          explain: true,
          explainTask: 'style',
        });
        setPredictionData(result);
      } catch (err) {
        setError((err as Error).message || 'An unexpected analysis error occurred.');
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  // Handle switching Grad-CAM head against live backend
  const handleSelectHead = async (head: string) => {
    if (!selectedFile || !predictionData) return;

    setIsLoadingHead(true);
    try {
      const exp = await fetchExplanationForHead(selectedFile, head);
      if (exp) {
        setPredictionData((prev) => (prev ? { ...prev, explanation: exp } : prev));
      }
    } catch (err) {
      console.warn('Could not load Grad-CAM for head:', head, err);
    } finally {
      setIsLoadingHead(false);
    }
  };

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    runPrediction(file);
  };

  const handleClear = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setPredictionData(null);
    setError(null);
  };

  const handleRetry = () => {
    if (selectedFile) {
      runPrediction(selectedFile);
    }
  };

  // Initialize state based on optional ?state= query param for testing
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const stateParam = params.get('state');

    if (stateParam === 'loading') {
      setIsLoading(true);
      return;
    }

    if (stateParam === 'error') {
      setError('HTTP 500: Internal server error occurred during PyTorch inference.');
      return;
    }

    if (stateParam === 'sample') {
      const liveSampleUrl = `${getApiBaseUrl()}/static/images/wikiart_000001.jpg`;
      fetch(liveSampleUrl)
        .then((res) => {
          if (!res.ok) throw new Error('Live sample not reachable');
          return res.blob();
        })
        .then((blob) => {
          const file = new File([blob], 'van_gogh_realism.jpg', { type: 'image/jpeg' });
          setSelectedFile(file);
          setPreviewUrl(liveSampleUrl);
          runPrediction(file);
        })
        .catch(() => {});
    }
    // Default: remain in clean landing page state (ready for user upload or sample selection)
  }, [runPrediction]);

  return (
    <div className="min-h-screen flex flex-col bg-[#fcfaf7] dark:bg-[#121316] text-stone-900 dark:text-stone-100 transition-colors duration-200">
      <Header
        darkMode={darkMode}
        onToggleDarkMode={toggleDarkMode}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-7 space-y-7">
        {/* Upload Zone */}
        <UploadArea
          onFileSelected={handleFileSelected}
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onClear={handleClear}
          isLoading={isLoading}
        />

        {/* Loading State: Skeleton */}
        {isLoading && <DashboardSkeleton />}

        {/* Error State with Retry Button */}
        {error && !isLoading && (
          <div
            role="alert"
            className="rounded-3xl border border-rose-200 dark:border-rose-900/60 bg-rose-50/80 dark:bg-rose-950/30 p-6 sm:p-8 text-center space-y-4 shadow-sm"
          >
            <div className="mx-auto h-12 w-12 rounded-2xl bg-rose-100 dark:bg-rose-900/50 text-rose-600 dark:text-rose-400 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6" />
            </div>
            <div>
              <h3 className="font-serif text-xl font-bold text-rose-900 dark:text-rose-200">
                Inference or Communication Error
              </h3>
              <p className="text-xs text-rose-700 dark:text-rose-300/90 max-w-lg mx-auto mt-1 leading-relaxed">
                {error}
              </p>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-2xl text-xs font-bold bg-amber-800 hover:bg-amber-900 text-white shadow-md shadow-amber-900/20 transition-all focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Retry Analysis</span>
            </button>
          </div>
        )}

        {/* Loaded Results Dashboard */}
        {!isLoading && !error && predictionData && previewUrl && (
          <ResultsDashboard
            data={predictionData}
            previewImage={previewUrl}
            onSelectHead={handleSelectHead}
            isLoadingHead={isLoadingHead}
          />
        )}

        {/* Empty State when cleared */}
        {!isLoading && !error && !predictionData && (
          <div className="rounded-3xl border border-dashed border-stone-300 dark:border-stone-800 bg-white/40 dark:bg-stone-900/20 p-12 text-center space-y-3">
            <div className="mx-auto h-12 w-12 rounded-2xl bg-amber-100/70 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 flex items-center justify-center">
              <ImageIcon className="h-6 w-6" />
            </div>
            <h3 className="font-serif text-xl font-bold text-stone-800 dark:text-stone-200">
              No Artwork Loaded
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400 max-w-sm mx-auto">
              Upload an artwork image above or select a curated sample preset to start analysis.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-200/90 dark:border-stone-800/80 bg-white/60 dark:bg-[#121316] py-6 text-center text-xs text-stone-500 dark:text-stone-400">
        <p>ArtVision · Multi-Task Deep Learning Framework for Artwork Analysis</p>
      </footer>
    </div>
  );
};
