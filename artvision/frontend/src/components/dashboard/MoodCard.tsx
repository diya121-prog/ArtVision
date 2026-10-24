import React from 'react';
import { Smile } from 'lucide-react';
import { MoodPrediction } from '../../api/types';
import { ConfidenceBar } from '../common/ConfidenceBar';

export interface MoodCardProps {
  moodData: MoodPrediction | null;
}

export const MoodCard: React.FC<MoodCardProps> = ({ moodData }) => {
  if (!moodData) {
    return (
      <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm">
        <h3 className="font-serif text-lg font-bold text-stone-400 dark:text-stone-500 flex items-center gap-2">
          <Smile className="h-5 w-5 text-stone-400" />
          <span>Mood & Emotion</span>
        </h3>
        <p className="text-sm text-stone-500 dark:text-stone-400 mt-4">
          Mood analysis unavailable for this piece.
        </p>
      </div>
    );
  }

  const { prediction, distribution } = moodData;

  // Sort distribution entries by probability descending
  const sortedEntries = Object.entries(distribution).sort(([, a], [, b]) => b - a);

  return (
    <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
              <Smile className="h-4 w-4" />
              <span>Affective Tone & Mood</span>
            </div>
            <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
              {prediction}
            </h3>
          </div>
        </div>

        <p className="text-xs text-stone-500 dark:text-stone-400 mt-2">
          Calibrated soft probability distribution across emotional categories:
        </p>

        {/* Distribution Bars */}
        <div className="mt-4 space-y-2.5">
          {sortedEntries.map(([label, prob]) => {
            const isTop = label === prediction;
            return (
              <ConfidenceBar
                key={label}
                label={label}
                confidence={prob}
                isTop={isTop}
                barColor={
                  isTop
                    ? 'bg-gradient-to-r from-amber-600 via-amber-700 to-amber-800'
                    : 'bg-stone-200 dark:bg-stone-700'
                }
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};
