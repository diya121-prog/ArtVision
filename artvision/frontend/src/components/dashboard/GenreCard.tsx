import React from 'react';
import { Layers, HelpCircle } from 'lucide-react';
import { GenrePrediction } from '../../api/types';
import { ConfidenceBar } from '../common/ConfidenceBar';

export interface GenreCardProps {
  genreData: GenrePrediction | null;
}

export const GenreCard: React.FC<GenreCardProps> = ({ genreData }) => {
  if (!genreData) {
    return (
      <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-stone-400 dark:text-stone-500">
            <Layers className="h-4 w-4" />
            <span>Genre & Subject</span>
          </div>
          <h3 className="font-serif text-xl font-bold text-stone-400 dark:text-stone-500 mt-1">
            Not Available
          </h3>
        </div>

        <div className="mt-6 rounded-2xl bg-stone-50 dark:bg-stone-800/40 p-4 border border-dashed border-stone-200 dark:border-stone-700/60 text-center">
          <HelpCircle className="h-6 w-6 text-stone-400 mx-auto mb-1.5" />
          <p className="text-xs text-stone-500 dark:text-stone-400 leading-relaxed">
            Genre head unannotated or disabled in current model checkpoint.
          </p>
        </div>
      </div>
    );
  }

  const { top3, prediction, uncertain } = genreData;

  return (
    <div
      className={`rounded-3xl border p-6 shadow-sm transition-all duration-200 ${
        uncertain
          ? 'border-amber-300 dark:border-amber-800/80 bg-amber-50/30 dark:bg-amber-950/20'
          : 'border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
            <Layers className="h-4 w-4" />
            <span>Genre & Subject</span>
          </div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
            {prediction}
          </h3>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500">
          Top Genre Candidates
        </p>
        <div className="space-y-2.5">
          {top3.map((item, idx) => (
            <ConfidenceBar
              key={item.genre}
              label={item.genre}
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
