import React, { useState } from 'react';
import { HelpCircle, ChevronDown, ChevronUp, Scale, HeartHandshake, ShieldAlert } from 'lucide-react';

export const AboutPredictions: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="rounded-3xl border border-stone-200/80 dark:border-stone-800 bg-stone-50/70 dark:bg-stone-900/40 overflow-hidden transition-all">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-5 sm:p-6 flex items-center justify-between text-left hover:bg-stone-100/50 dark:hover:bg-stone-800/40 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-500"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-400 flex items-center justify-center shrink-0">
            <HelpCircle className="h-5 w-5" />
          </div>
          <div>
            <h4 className="text-sm sm:text-base font-bold text-stone-900 dark:text-white">
              About These Predictions & Calibrated Estimates
            </h4>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              Understanding uncertainty scaling, dataset boundaries, and model limitations
            </p>
          </div>
        </div>

        <div className="text-stone-400 hover:text-stone-600 dark:hover:text-stone-200">
          {isOpen ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
        </div>
      </button>

      {isOpen && (
        <div className="px-5 pb-6 sm:px-6 pt-2 border-t border-stone-200/60 dark:border-stone-800/60 grid grid-cols-1 md:grid-cols-3 gap-5 text-xs text-stone-600 dark:text-stone-300 leading-relaxed">
          <div className="space-y-1.5 p-3.5 rounded-2xl bg-white dark:bg-stone-900/60 border border-stone-200/80 dark:border-stone-800">
            <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-white">
              <Scale className="h-4 w-4 text-amber-600" />
              <span>Calibrated Confidence</span>
            </div>
            <p>
              Reported percentages reflect temperature-scaled probability estimates, not mathematical guarantees. When confidence is low or margin is narrow, the model flags the prediction as <em>Uncertain</em>.
            </p>
          </div>

          <div className="space-y-1.5 p-3.5 rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800">
            <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white">
              <HeartHandshake className="h-4 w-4 text-pink-500" />
              <span>Affective Subjectivity</span>
            </div>
            <p>
              Artistic mood and emotional tone are inherently subjective and dependent on cultural context. The 5-class distribution represents soft consensus patterns from human annotations.
            </p>
          </div>

          <div className="space-y-1.5 p-3.5 rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800">
            <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white">
              <ShieldAlert className="h-4 w-4 text-amber-500" />
              <span>Artist & Genre Reliability</span>
            </div>
            <p>
              Artist identification is constrained to top cataloged painters with sufficient training density. For unrepresented masters or AI art, artist predictions are intentionally masked or marked uncertain.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
