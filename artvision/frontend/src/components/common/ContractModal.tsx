import React from 'react';
import { X, FileCode, CheckCircle2 } from 'lucide-react';

export interface ContractModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ContractModal: React.FC<ContractModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="contract-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-950/70 backdrop-blur-sm animate-fadeIn"
    >
      <div className="relative w-full max-w-3xl max-h-[85vh] flex flex-col rounded-3xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-5 sm:p-6 border-b border-stone-200 dark:border-stone-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-amber-100 dark:bg-amber-950/80 text-amber-800 dark:text-amber-400 flex items-center justify-center">
              <FileCode className="h-5 w-5" />
            </div>
            <div>
              <h3 id="contract-modal-title" className="font-serif text-xl font-bold text-stone-900 dark:text-white">
                API Contract & Specification
              </h3>
              <p className="text-xs text-stone-500 dark:text-stone-400">
                POST /predict · schema_version: &quot;1.0.0&quot; · Location: frontend/src/api/contract.md
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close API Contract modal"
            className="p-2 rounded-xl text-stone-400 hover:text-stone-600 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content body */}
        <div className="flex-1 overflow-y-auto p-5 sm:p-6 space-y-4 text-xs font-mono bg-stone-50 dark:bg-stone-950/50 text-stone-700 dark:text-stone-300">
          <div className="rounded-2xl bg-white dark:bg-stone-900 p-4 border border-stone-200 dark:border-stone-800 space-y-2">
            <div className="flex items-center gap-1.5 font-bold text-amber-800 dark:text-amber-400 font-sans text-sm">
              <CheckCircle2 className="h-4 w-4 text-amber-700" />
              <span>Multi-Task Output Schema (Envelope)</span>
            </div>
            <pre className="overflow-x-auto text-[11px] leading-relaxed">
{`{
  "schema_version": "1.0.0",
  "style": {
    "top3": [{ "style": string, "confidence": number }],
    "prediction": string,
    "confidence": number,
    "uncertain": boolean
  },
  "movement": {
    "derived_from_style": string,
    "movement": string
  } | null,
  "artist": {
    "top3": [{ "artist": string, "confidence": number }],
    "prediction": string,
    "confidence": number,
    "uncertain": boolean
  } | null,
  "genre": {
    "top3": [{ "genre": string, "confidence": number }],
    "prediction": string,
    "confidence": number,
    "uncertain": boolean
  } | null,
  "mood": {
    "prediction": string,
    "confidence": number,
    "distribution": Record<string, number>,
    "uncertain": boolean
  } | null,
  "authenticity": {
    "is_ai_generated": boolean,
    "label": "ai_generated" | "human_made",
    "probability_ai": number,
    "confidence": number,
    "uncertain": boolean
  } | null,
  "era": null,
  "technique": null,
  "color": {
    "palette": [{ "hex": string, "rgb": [number, number, number], "percent": number }],
    "color_count": number,
    "temperature": "warm" | "cool" | "neutral-mixed" | "monochrome",
    "warm_score": number,
    "cool_score": number,
    "brightness": "dark" | "mid" | "light",
    "saturation": "muted" | "balanced" | "vivid",
    "is_pastel": boolean
  } | null,
  "color_error": string | null,
  "explanation": {
    [head: string]: {
      "target_class": string,
      "heatmap_url": string,
      "overlay_url": string,
      "stats": { "mean_activation": number, "max_activation": number }
    }
  } | null,
  "similar": [
    {
      "image_url": string,
      "title": string,
      "artist": string,
      "style": string,
      "similarity": number
    }
  ] | null
}`}
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-amber-700 hover:bg-amber-800 text-white transition-colors"
          >
            Close Specification
          </button>
        </div>
      </div>
    </div>
  );
};
