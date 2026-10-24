import React from 'react';
import { Calendar, Brush, History } from 'lucide-react';
import { EraPrediction, TechniquePrediction } from '../../api/types';

export interface EraTechniqueCardProps {
  eraData?: EraPrediction | null;
  techniqueData?: TechniquePrediction | null;
}

export const EraTechniqueCard: React.FC<EraTechniqueCardProps> = ({ eraData, techniqueData }) => {
  const period = eraData?.period || 'Historical Timeline & Medium';
  const timeline = eraData?.timeline || 'Historical timeline estimation';
  const medium = techniqueData?.medium || 'Oil / mixed media on canvas';
  const method = techniqueData?.method || 'Classical artistic method';
  const derivedFrom = eraData?.derived_from || techniqueData?.derived_from;

  return (
    <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm flex flex-col justify-between transition-all duration-200">
      <div>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
              <Calendar className="h-4 w-4" />
              <span>Era & Technique</span>
            </div>
            <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
              {period}
            </h3>
          </div>
        </div>

        <div className="mt-5 space-y-3">
          {/* Timeline / Epoch */}
          <div className="flex items-start gap-3 p-3.5 rounded-2xl bg-stone-50 dark:bg-stone-800/60 border border-stone-200/80 dark:border-stone-700/60">
            <div className="h-8 w-8 rounded-xl bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-400 flex items-center justify-center shrink-0 mt-0.5">
              <Calendar className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-bold text-stone-900 dark:text-stone-100">
                Timeline & Epoch
              </p>
              <p className="text-xs font-medium text-stone-600 dark:text-stone-300 mt-0.5">
                {timeline}
              </p>
            </div>
          </div>

          {/* Medium & Technique */}
          <div className="flex items-start gap-3 p-3.5 rounded-2xl bg-stone-50 dark:bg-stone-800/60 border border-stone-200/80 dark:border-stone-700/60">
            <div className="h-8 w-8 rounded-xl bg-stone-200/80 dark:bg-stone-700/80 text-stone-800 dark:text-stone-200 flex items-center justify-center shrink-0 mt-0.5">
              <Brush className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-bold text-stone-900 dark:text-stone-100">
                {medium}
              </p>
              <p className="text-[11px] text-stone-500 dark:text-stone-400 mt-0.5 leading-snug">
                {method}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-stone-100 dark:border-stone-800 flex items-center gap-1.5 text-[11px] text-stone-400 dark:text-stone-500">
        <History className="h-3.5 w-3.5 text-stone-400 shrink-0" />
        <span>
          {derivedFrom
            ? `Derived via historical analysis of ${derivedFrom.replace(/_/g, ' ')}.`
            : 'Derived via historical and physical art taxonomy.'}
        </span>
      </div>
    </div>
  );
};
