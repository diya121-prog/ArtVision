import React, { useState } from 'react';
import { Eye, Layers, Sliders } from 'lucide-react';
import { UnifiedExplanation, HeadExplanation, ExplanationHeadKey } from '../../api/types';
import { Badge } from '../common/Badge';

export interface ExplanationViewerProps {
  explanation: UnifiedExplanation | null | undefined;
  originalImage: string;
  onSelectHead?: (head: string) => void;
  isLoadingHead?: boolean;
}

type ViewMode = 'original' | 'overlay' | 'heatmap';

export const ExplanationViewer: React.FC<ExplanationViewerProps> = ({
  explanation,
  originalImage,
  onSelectHead,
  isLoadingHead = false,
}) => {
  // Determine if explanation is a single API response or a multi-head dictionary
  const isDirectApiExplanation = Boolean(
    explanation &&
      typeof explanation === 'object' &&
      ('overlay_url' in explanation || 'task' in explanation)
  );

  const availableHeads: ExplanationHeadKey[] = isDirectApiExplanation
    ? ['style', 'artist', 'mood', 'authenticity', 'genre']
    : ((explanation ? Object.keys(explanation) : []) as ExplanationHeadKey[]);

  const defaultHead = isDirectApiExplanation
    ? ((explanation as HeadExplanation).task as ExplanationHeadKey) || 'style'
    : availableHeads[0] || 'style';

  const [activeHead, setActiveHead] = useState<ExplanationHeadKey>(defaultHead);
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [opacity, setOpacity] = useState<number>(0.85);

  if (!explanation) {
    return (
      <div className="rounded-3xl border border-dashed border-stone-300 dark:border-stone-800 bg-white/50 dark:bg-stone-900/40 p-8 text-center">
        <div className="mx-auto h-12 w-12 rounded-2xl bg-stone-100 dark:bg-stone-800 text-stone-400 flex items-center justify-center mb-3">
          <Eye className="h-6 w-6" />
        </div>
        <h3 className="font-serif text-lg font-bold text-stone-800 dark:text-stone-200">
          Grad-CAM Visual Explanations
        </h3>
        <p className="text-xs text-stone-500 dark:text-stone-400 max-w-md mx-auto mt-1 leading-relaxed">
          Saliency heatmaps were not computed for this inference pass. Explanations can be requested by passing the explainability flag to the backend.
        </p>
      </div>
    );
  }

  // Extract explanation data for active head
  let currentOverlayUrl: string | undefined;
  let currentHeatmapUrl: string | undefined;
  let currentTargetClass: string = activeHead;
  let currentBmf: number | undefined;

  if (isDirectApiExplanation) {
    const direct = explanation as HeadExplanation;
    currentOverlayUrl = direct.overlay_url;
    currentHeatmapUrl = direct.heatmap_url || direct.overlay_url;
    currentTargetClass = direct.target_prediction || direct.target_class || activeHead;
    currentBmf = direct.border_mass_fraction;
  } else {
    const headObj = (explanation as Record<string, HeadExplanation>)?.[activeHead];
    if (headObj) {
      currentOverlayUrl = headObj.overlay_url;
      currentHeatmapUrl = headObj.heatmap_url || headObj.overlay_url;
      currentTargetClass = headObj.target_class || headObj.target_prediction || activeHead;
      currentBmf = headObj.border_mass_fraction;
    }
  }

  const handleHeadClick = (head: ExplanationHeadKey) => {
    setActiveHead(head);
    if (onSelectHead) {
      onSelectHead(head);
    }
  };

  return (
    <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 sm:p-7 shadow-sm space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
            <Eye className="h-4 w-4" />
            <span>Explainability & Attention (Grad-CAM)</span>
          </div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-white mt-1">
            Visual Saliency Analysis
          </h3>
          <p className="text-xs text-stone-500 dark:text-stone-400 mt-1">
            Visualizing spatial activations from the final convolutional layer of the EfficientNet backbone.
          </p>
        </div>

        {/* Head Selector Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-2xl bg-stone-100 dark:bg-stone-800 border border-stone-200/80 dark:border-stone-700/60 self-start sm:self-auto flex-wrap">
          {availableHeads.map((head) => (
            <button
              key={head}
              type="button"
              disabled={isLoadingHead}
              onClick={() => handleHeadClick(head)}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold capitalize transition-all ${
                activeHead === head
                  ? 'bg-amber-700 text-white shadow-sm'
                  : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-white'
              } disabled:opacity-50`}
            >
              {head}
            </button>
          ))}
        </div>
      </div>

      {/* Main Visualizer Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Visualizer Display Canvas */}
        <div className="lg:col-span-8 relative aspect-[4/3] rounded-2xl overflow-hidden bg-stone-950 border border-stone-200 dark:border-stone-800 shadow-inner flex items-center justify-center">
          {/* Base image */}
          <img
            src={originalImage}
            alt="Original artwork"
            className="absolute inset-0 h-full w-full object-contain"
          />

          {/* Overlay / Heatmap Layer */}
          {currentOverlayUrl && viewMode !== 'original' && (
            <img
              src={viewMode === 'heatmap' ? currentHeatmapUrl : currentOverlayUrl}
              alt={`Grad-CAM visualization for ${activeHead}`}
              style={{
                opacity: viewMode === 'heatmap' ? 1.0 : opacity,
              }}
              className="absolute inset-0 h-full w-full object-contain pointer-events-none transition-opacity duration-200"
            />
          )}

          {/* Mode Pill Indicator */}
          <div className="absolute top-3 left-3">
            <Badge variant="primary" size="sm">
              {viewMode.toUpperCase()} VIEW
            </Badge>
          </div>

          {isLoadingHead && (
            <div className="absolute inset-0 bg-black/50 backdrop-blur-xs flex items-center justify-center text-white text-xs font-semibold">
              Generating Grad-CAM for {activeHead}...
            </div>
          )}
        </div>

        {/* Visualizer Controls Sidebar */}
        <div className="lg:col-span-4 space-y-5">
          {/* View Mode Toggle */}
          <div className="space-y-2">
            <label className="text-xs font-bold uppercase tracking-wider text-stone-500 dark:text-stone-400 flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5" />
              Display Mode
            </label>
            <div className="grid grid-cols-3 gap-1.5 p-1 rounded-2xl bg-stone-100 dark:bg-stone-800/80 border border-stone-200 dark:border-stone-700/60">
              {(['original', 'overlay', 'heatmap'] as ViewMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setViewMode(mode)}
                  className={`py-1.5 rounded-xl text-xs font-semibold capitalize transition-colors ${
                    viewMode === mode
                      ? 'bg-white dark:bg-stone-900 text-stone-900 dark:text-white shadow-sm font-bold'
                      : 'text-stone-500 hover:text-stone-900 dark:hover:text-white'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {/* Heatmap Opacity Slider (enabled in overlay mode) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <label
                htmlFor="heatmap-opacity"
                className="font-bold uppercase tracking-wider text-stone-500 dark:text-stone-400 flex items-center gap-1.5"
              >
                <Sliders className="h-3.5 w-3.5" />
                Overlay Opacity
              </label>
              <span className="font-mono font-semibold text-stone-700 dark:text-stone-300">
                {Math.round(opacity * 100)}%
              </span>
            </div>
            <input
              id="heatmap-opacity"
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              disabled={viewMode !== 'overlay'}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              aria-label="Heatmap overlay opacity"
              className="w-full accent-amber-700 dark:accent-amber-500 disabled:opacity-40 cursor-pointer"
            />
          </div>

          {/* Target Class Info */}
          <div className="rounded-2xl bg-stone-50 dark:bg-stone-800/50 p-4 border border-stone-200/80 dark:border-stone-800 space-y-1.5">
            <span className="text-[10px] font-bold uppercase tracking-wider text-stone-400">
              Target Head Prediction
            </span>
            <p className="text-sm font-bold text-stone-800 dark:text-stone-200 capitalize">
              {currentTargetClass.replace(/_/g, ' ')}
            </p>
            {currentBmf !== undefined && (
              <div className="flex items-center gap-2 text-[11px] font-mono text-stone-500 dark:text-stone-400 pt-1">
                <span>Border-mass fraction: {currentBmf.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
