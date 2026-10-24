import React from 'react';
import { User, HelpCircle } from 'lucide-react';
import { ArtistPrediction } from '../../api/types';
import { ConfidenceBar } from '../common/ConfidenceBar';

export interface ArtistCardProps {
  artistData: ArtistPrediction | null;
}

export const ArtistCard: React.FC<ArtistCardProps> = ({ artistData }) => {
  if (!artistData) {
    return (
      <div className="rounded-3xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-2xs flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-stone-400 dark:text-stone-500">
            <User className="h-4 w-4" />
            <span>Artist Attribution</span>
          </div>
          <h3 className="font-serif text-xl font-bold text-stone-400 dark:text-stone-500 mt-1">
            Artist Not Identified
          </h3>
        </div>
        <div className="mt-6 rounded-2xl bg-stone-50/70 dark:bg-stone-800/40 p-4 border border-dashed border-stone-200 dark:border-stone-700/60 text-center">
          <HelpCircle className="h-6 w-6 text-stone-400 mx-auto mb-1.5" />
          <p className="text-xs text-stone-500 dark:text-stone-400 leading-relaxed">
            Artist prediction was masked or not available for this artwork (e.g. unindexed artist or synthetic piece).
          </p>
        </div>
      </div>
    );
  }

  const { top3, prediction, uncertain } = artistData;

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
            <User className="h-4 w-4" />
            <span>Artist Attribution</span>
          </div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
            {prediction}
          </h3>
        </div>
      </div>

      {/* Top 3 Artists */}
      <div className="mt-5 space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500">
          Top Artist Candidates
        </p>
        <div className="space-y-2.5">
          {top3.map((item, idx) => (
            <ConfidenceBar
              key={item.artist}
              label={item.artist}
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
