import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, AlertCircle, Palette, ZoomIn, ExternalLink, Image as ImageIcon } from 'lucide-react';
import { validateArtworkFile, isMockMode, getApiBaseUrl } from '../../api/client';
import { createMockArtworkSvg } from '../../api/mocks/generator';

export interface UploadAreaProps {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
  previewUrl: string | null;
  onClear: () => void;
  isLoading: boolean;
}

export const UploadArea: React.FC<UploadAreaProps> = ({
  onFileSelected,
  selectedFile,
  previewUrl,
  onClear,
  isLoading,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isPreviewModalOpen, setIsPreviewModalOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Close preview modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsPreviewModalOpen(false);
      }
    };
    if (isPreviewModalOpen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isPreviewModalOpen]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!isLoading) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isLoading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    setValidationError(null);
    const result = validateArtworkFile(file);
    if (!result.valid) {
      setValidationError(result.error || 'Invalid file.');
      return;
    }
    onFileSelected(file);
  };

  // Helper for 1-click sample artwork selection (works in both Live and Mock mode)
  const handleSelectSample = async (type: 'painting' | 'ai' | 'non-art') => {
    const isMock = isMockMode();

    if (!isMock) {
      // Live API mode: fetch real samples from backend static mounts
      try {
        let imageUrl = '';
        let fileName = '';

        if (type === 'painting') {
          imageUrl = `${getApiBaseUrl()}/static/images/wikiart_000001.jpg`;
          fileName = 'van_gogh_realism.jpg';
        } else if (type === 'ai') {
          imageUrl = `${getApiBaseUrl()}/static/images/ai_artbench_0-114513684-761761.jpg`;
          fileName = 'ai_artbench_diffusion.jpg';
        } else {
          // Generate realistic non-art photo SVG (architectural street grid)
          const nonArtSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
            <defs>
              <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#38bdf8"/>
                <stop offset="60%" stop-color="#bae6fd"/>
                <stop offset="100%" stop-color="#e0f2fe"/>
              </linearGradient>
            </defs>
            <rect width="600" height="450" fill="url(#sky)"/>
            <!-- Office building glass windows and concrete -->
            <rect x="80" y="90" width="190" height="360" fill="#64748b"/>
            <rect x="310" y="40" width="220" height="410" fill="#475569"/>
            <!-- Grid of glass windows -->
            <g fill="#93c5fd" opacity="0.8">
              <rect x="100" y="120" width="30" height="40"/>
              <rect x="145" y="120" width="30" height="40"/>
              <rect x="190" y="120" width="30" height="40"/>
              <rect x="235" y="120" width="25" height="40"/>
              <rect x="100" y="180" width="30" height="40"/>
              <rect x="145" y="180" width="30" height="40"/>
              <rect x="190" y="180" width="30" height="40"/>
              <rect x="235" y="180" width="25" height="40"/>
              <rect x="330" y="70" width="35" height="45"/>
              <rect x="380" y="70" width="35" height="45"/>
              <rect x="430" y="70" width="35" height="45"/>
              <rect x="480" y="70" width="35" height="45"/>
              <rect x="330" y="135" width="35" height="45"/>
              <rect x="380" y="135" width="35" height="45"/>
              <rect x="430" y="135" width="35" height="45"/>
              <rect x="480" y="135" width="35" height="45"/>
            </g>
            <rect x="0" y="380" width="600" height="70" fill="#334155"/>
            <text x="20" y="425" fill="#f8fafc" font-family="sans-serif" font-size="14" font-weight="bold">City Architecture Photo (Non-Art Image)</text>
          </svg>`;
          imageUrl = `data:image/svg+xml;utf8,${encodeURIComponent(nonArtSvg)}`;
          fileName = 'architecture_photo.svg';
        }

        const res = await fetch(imageUrl);
        const blob = await res.blob();
        const file = new File([blob], fileName, { type: blob.type || 'image/jpeg' });
        processFile(file);
        return;
      } catch (err) {
        console.warn('Failed to fetch live sample, falling back to procedural mock:', err);
      }
    }

    // Mock Mode fallback
    let svgData = '';
    let sampleName = '';
    if (type === 'painting') {
      svgData = createMockArtworkSvg('The Night Watch', 'Baroque', ['#0f0b08', '#3d2714', '#b47d34', '#e6c37a']);
      sampleName = 'baroque_portrait.webp';
    } else if (type === 'ai') {
      svgData = createMockArtworkSvg('Synthetic Drift', 'Surrealism', ['#180033', '#6b21a8', '#ec4899', '#06b6d4']);
      sampleName = 'synthetic_ai.webp';
    } else {
      svgData = createMockArtworkSvg('Modern Skyline', 'Architecture', ['#0f172a', '#334155', '#64748b', '#cbd5e1']);
      sampleName = 'architecture_photo.webp';
    }

    const res = await fetch(svgData);
    const blob = await res.blob();
    const file = new File([blob], sampleName, { type: 'image/webp' });
    processFile(file);
  };

  return (
    <section aria-labelledby="upload-section-title" className="space-y-4">
      <h2 id="upload-section-title" className="sr-only">Artwork Upload Section</h2>
      
      {/* Upload Zone */}
      {!previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          tabIndex={0}
          role="button"
          aria-label="Upload artwork image. Drag and drop or press Enter to browse files."
          className={`relative group cursor-pointer rounded-3xl border-2 border-dashed p-8 sm:p-12 text-center transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-amber-500/20 ${
            isDragOver
              ? 'border-amber-600 bg-amber-50/60 dark:bg-amber-950/30 scale-[1.005]'
              : 'border-stone-300 dark:border-stone-700/80 bg-white/70 dark:bg-stone-900/40 hover:border-amber-600 dark:hover:border-amber-500 hover:bg-stone-50/60 dark:hover:bg-stone-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handleFileChange}
            disabled={isLoading}
          />

          <div className="mx-auto flex max-w-md flex-col items-center">
            <div className="h-16 w-16 rounded-2xl bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-400 flex items-center justify-center mb-4 transition-transform group-hover:scale-110 shadow-inner">
              <Upload className="h-8 w-8" />
            </div>

            <p className="text-base sm:text-lg font-semibold text-stone-800 dark:text-stone-100">
              Drag and drop an artwork, or{' '}
              <span className="text-amber-700 dark:text-amber-400 underline decoration-amber-300 underline-offset-4 font-bold">
                browse files
              </span>
            </p>

            <p className="text-xs text-stone-500 dark:text-stone-400 mt-2">
              Supports JPG, PNG, WebP · Max file size 15 MB
            </p>

            {/* Quick Sample Artworks Bar */}
            <div
              className="mt-6 pt-5 border-t border-stone-200/80 dark:border-stone-800/80 w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-center gap-1.5 text-xs text-stone-500 dark:text-stone-400 mb-2.5 font-medium">
                <Palette className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400" />
                <span>Or evaluate a curated sample artwork:</span>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={() => handleSelectSample('painting')}
                  className="px-3 py-1 rounded-xl text-xs font-semibold bg-stone-100 hover:bg-amber-100 dark:bg-stone-800 dark:hover:bg-amber-950/80 text-stone-700 dark:text-stone-200 hover:text-amber-800 dark:hover:text-amber-300 transition-colors border border-stone-200 dark:border-stone-700"
                >
                  Real Painting (Van Gogh)
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectSample('ai')}
                  className="px-3 py-1 rounded-xl text-xs font-semibold bg-stone-100 hover:bg-amber-100 dark:bg-stone-800 dark:hover:bg-amber-950/80 text-stone-700 dark:text-stone-200 hover:text-amber-800 dark:hover:text-amber-300 transition-colors border border-stone-200 dark:border-stone-700"
                >
                  AI-Generated (ArtBench)
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectSample('non-art')}
                  className="px-3 py-1 rounded-xl text-xs font-semibold bg-stone-100 hover:bg-amber-100 dark:bg-stone-800 dark:hover:bg-amber-950/80 text-stone-700 dark:text-stone-200 hover:text-amber-800 dark:hover:text-amber-300 transition-colors border border-stone-200 dark:border-stone-700"
                >
                  Non-Art Photo
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Preview Card */
        <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-4 sm:p-5 shadow-sm transition-all">
          <div className="flex flex-col sm:flex-row items-center gap-5">
            {/* Interactive Preview Box */}
            <div
              role="button"
              tabIndex={0}
              onClick={() => setIsPreviewModalOpen(true)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setIsPreviewModalOpen(true);
                }
              }}
              title="Click to expand full resolution preview"
              className="group relative h-36 w-48 shrink-0 overflow-hidden rounded-2xl border border-stone-200 dark:border-stone-800 bg-stone-100 dark:bg-stone-950 cursor-pointer focus:outline-none focus:ring-2 focus:ring-amber-500 transition-all hover:shadow-lg hover:border-amber-400 select-none"
            >
              <img
                src={previewUrl}
                alt={selectedFile ? selectedFile.name : 'Uploaded artwork preview'}
                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
              />
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded-md bg-stone-950/70 backdrop-blur-sm text-[10px] font-bold text-white tracking-wide">
                Preview
              </div>
              <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-stone-950/80 backdrop-blur-sm text-white text-xs font-semibold shadow">
                  <ZoomIn className="h-3.5 w-3.5" />
                  Full Preview
                </span>
              </div>
            </div>

            <div className="flex-1 w-full text-center sm:text-left">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-stone-900 dark:text-stone-100 truncate max-w-sm">
                    {selectedFile?.name || 'Artwork Image'}
                  </h3>
                  <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">
                    {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB · ${selectedFile.type}` : 'Ready for analysis'}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={onClear}
                  disabled={isLoading}
                  aria-label="Remove image and reset analysis"
                  className="p-2 rounded-xl text-stone-400 hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/30 transition-colors disabled:opacity-50"
                  title="Remove image and reset"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="mt-4 flex items-center gap-3">
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                  <span className="h-2 w-2 rounded-full bg-emerald-600 animate-pulse" />
                  Loaded & Validated
                </span>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-xs font-semibold text-amber-700 dark:text-amber-400 hover:underline"
                >
                  Upload different image
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Validation Alert */}
      {validationError && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-2xl border border-rose-200 dark:border-rose-900/60 bg-rose-50 dark:bg-rose-950/30 p-4 text-sm text-rose-800 dark:text-rose-300"
        >
          <AlertCircle className="h-5 w-5 shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold">Upload Validation Notice</p>
            <p className="text-xs mt-0.5 text-rose-700 dark:text-rose-300/90">{validationError}</p>
          </div>
          <button
            type="button"
            onClick={() => setValidationError(null)}
            className="text-rose-600 hover:text-rose-800 text-xs font-bold"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Artwork Fullscreen / Lightbox Preview Modal */}
      {isPreviewModalOpen && previewUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-stone-950/80 backdrop-blur-md animate-in fade-in duration-200"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setIsPreviewModalOpen(false);
            }
          }}
          role="dialog"
          aria-modal="true"
          aria-label="Artwork Full Preview"
        >
          <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col rounded-3xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-stone-100 dark:border-stone-800">
              <div className="flex items-center gap-2.5">
                <div className="h-8 w-8 rounded-xl bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-400 flex items-center justify-center">
                  <ImageIcon className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="font-serif text-base font-bold text-stone-900 dark:text-stone-100 truncate max-w-md">
                    {selectedFile?.name || 'Artwork Preview'}
                  </h3>
                  <p className="text-xs text-stone-500 dark:text-stone-400">
                    {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB · ${selectedFile.type}` : 'Active analysis artwork'}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsPreviewModalOpen(false)}
                className="rounded-full p-2 text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
                aria-label="Close preview modal"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Image View in Full Ratio */}
            <div className="relative flex-1 min-h-[300px] max-h-[65vh] bg-stone-950 flex items-center justify-center p-4 overflow-hidden">
              <img
                src={previewUrl}
                alt={selectedFile?.name || 'Artwork full preview'}
                className="max-h-full max-w-full object-contain rounded-lg shadow-2xl"
              />
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-4 bg-stone-50/70 dark:bg-stone-950/60 border-t border-stone-100 dark:border-stone-800 flex items-center justify-between gap-4">
              <span className="text-xs text-stone-500 dark:text-stone-400">
                Displaying full source resolution for neural analysis
              </span>
              <div className="flex items-center gap-3">
                <a
                  href={previewUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-stone-700 dark:text-stone-200 bg-white dark:bg-stone-800 hover:bg-stone-100 dark:hover:bg-stone-700 border border-stone-200 dark:border-stone-700 shadow-sm transition-colors"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open in New Tab
                </a>
                <button
                  type="button"
                  onClick={() => setIsPreviewModalOpen(false)}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-amber-700 hover:bg-amber-800 shadow-sm transition-colors"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
