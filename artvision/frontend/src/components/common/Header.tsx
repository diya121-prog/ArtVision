import React from 'react';
import { Palette, Moon, Sun, Printer } from 'lucide-react';
import { MockScenarioKey } from '../../api/types';

export interface HeaderProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
  mockMode?: boolean;
  onToggleMockMode?: () => void;
  activeScenario?: MockScenarioKey;
  onSelectScenario?: (scenario: MockScenarioKey) => void;
  latencyMs?: number;
  onSelectLatency?: (ms: number) => void;
  onOpenContract?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  onToggleDarkMode,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-stone-200/90 dark:border-stone-800/90 bg-[#fcfaf7]/90 dark:bg-[#121316]/90 backdrop-blur-md transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between gap-4">
        {/* Academic / Curatorial Branding */}
        <div className="flex items-center gap-3.5">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-amber-700 via-amber-600 to-amber-500 flex items-center justify-center text-white shadow-md shadow-amber-900/10 ring-1 ring-amber-400/30 shrink-0">
            <Palette className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-serif text-xl sm:text-2xl font-bold tracking-tight text-stone-900 dark:text-stone-100">
                ArtVision
              </span>
            </div>
            <p className="text-xs text-stone-500 dark:text-stone-400 font-medium">
              Explainable Multi-Task Artwork Analysis
            </p>
          </div>
        </div>

        {/* Controls & Actions */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          {/* Live Engine Status Indicator */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/80 text-xs font-semibold shadow-2xs">
            <span className="h-2 w-2 rounded-full bg-emerald-600 dark:bg-emerald-400 animate-pulse" />
            <span className="hidden sm:inline">Engine:</span>
            <span>Live Analysis</span>
          </div>

          {/* Quick Print/Export Curatorial Dossier for Presentations */}
          <button
            type="button"
            onClick={() => window.print()}
            title="Export / Print Curatorial Dossier for evaluation"
            className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-stone-200 dark:border-stone-800 bg-white/80 dark:bg-stone-900/80 hover:bg-stone-100 dark:hover:bg-stone-800 text-xs font-semibold text-stone-700 dark:text-stone-300 transition-colors shadow-2xs focus:outline-none focus:ring-2 focus:ring-amber-500"
          >
            <Printer className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400" />
            <span>Export Dossier</span>
          </button>

          {/* Dark Mode Toggle */}
          <button
            type="button"
            onClick={onToggleDarkMode}
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            className="p-2 rounded-xl border border-stone-200 dark:border-stone-800 bg-white/60 dark:bg-stone-900/60 hover:bg-stone-100 dark:hover:bg-stone-800 text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-amber-500 shadow-2xs"
          >
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </header>
  );
};
