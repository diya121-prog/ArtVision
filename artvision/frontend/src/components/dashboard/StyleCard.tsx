import React from 'react';
import { Palette, Compass } from 'lucide-react';
import { StylePrediction, MovementPrediction } from '../../api/types';
import { ConfidenceBar } from '../common/ConfidenceBar';

export interface StyleCardProps {
  styleData: StylePrediction | null;
  movementData: MovementPrediction | null;
}

export const StyleCard: React.FC<StyleCardProps> = ({ styleData, movementData }) => {
  if (!styleData) {
    return (
      <div className="rounded-3xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-2xs">
        <h3 className="font-serif text-lg font-bold text-stone-900 dark:text-white flex items-center gap-2">
          <Palette className="h-5 w-5 text-amber-700 dark:text-amber-400" />
          <span>Art Style & Movement</span>
        </h3>
        <p className="text-sm text-stone-500 dark:text-stone-400 mt-4">
          Style classification is not available for this artwork.
        </p>
      </div>
    );
  }

  const { top3, prediction, uncertain } = styleData;
  const movement = movementData?.movement || 'Unknown Movement';

  return (
    <div
      className={`rounded-3xl border p-6 shadow-2xs transition-all duration-200 hover:shadow-md ${
        uncertain
          ? 'border-amber-300 dark:border-amber-800/80 bg-amber-50/40 dark:bg-amber-950/20'
          : 'border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
            <Palette className="h-4 w-4" />
            <span>Primary Art Style</span>
          </div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
            {prediction}
          </h3>
        </div>
      </div>

      {/* Derived Movement Subcard */}
      <div className="mt-4 rounded-2xl bg-stone-50/80 dark:bg-stone-800/60 p-3.5 border border-stone-200/80 dark:border-stone-700/60">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 font-semibold text-stone-700 dark:text-stone-300">
            <Compass className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400" />
            Historical Movement
          </span>
          <span className="text-[10px] font-medium text-stone-400 dark:text-stone-500">
            (Derived from style)
          </span>
        </div>
        <p className="mt-1 text-sm font-bold text-stone-800 dark:text-stone-100">
          {movement}
        </p>
      </div>

      {/* Top 3 Confidence Bars */}
      <div className="mt-5 space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500">
          Top Style Candidates
        </p>
        <div className="space-y-2.5">
          {top3.map((item, idx) => (
            <ConfidenceBar
              key={item.style}
              label={item.style.replace(/_/g, ' ')}
              confidence={item.confidence}
              isTop={idx === 0}
              isUncertain={uncertain}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
