import React from 'react';
import { ShieldCheck, Cpu } from 'lucide-react';
import { AuthenticityPrediction } from '../../api/types';
import { Gauge } from '../common/Gauge';

export interface AuthenticityCardProps {
  authenticityData: AuthenticityPrediction | null;
}

export const AuthenticityCard: React.FC<AuthenticityCardProps> = ({ authenticityData }) => {
  if (!authenticityData) {
    return (
      <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm">
        <h3 className="font-serif text-lg font-bold text-stone-400 dark:text-stone-500 flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-stone-400" />
          <span>Authenticity Analysis</span>
        </h3>
        <p className="text-sm text-stone-500 dark:text-stone-400 mt-4">
          Authenticity evaluation is currently unavailable.
        </p>
      </div>
    );
  }

  const { is_ai_generated, probability_ai } = authenticityData;

  return (
    <div
      className={`rounded-3xl border p-6 shadow-sm transition-all duration-200 flex flex-col justify-between ${
        is_ai_generated
          ? 'border-rose-300/80 dark:border-rose-900/60 bg-gradient-to-b from-rose-50/60 via-stone-50/20 to-transparent dark:from-rose-950/20 dark:via-stone-900'
          : 'border-emerald-300/80 dark:border-emerald-900/60 bg-gradient-to-b from-emerald-50/50 via-stone-50/20 to-transparent dark:from-emerald-950/20 dark:via-stone-900'
      }`}
    >
      <div>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
              {is_ai_generated ? <Cpu className="h-4 w-4 text-rose-600 dark:text-rose-400" /> : <ShieldCheck className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />}
              <span>Art Authenticity</span>
            </div>
            <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
              {is_ai_generated ? 'AI-Generated' : 'Human-Crafted'}
            </h3>
          </div>
        </div>

        {/* Semi-circular Speedometer Gauge */}
        <div className="my-5">
          <Gauge
            value={probability_ai}
            isAi={is_ai_generated}
            uncertain={false}
            label="Authenticity probability gauge"
          />
        </div>
      </div>
    </div>
  );
};
