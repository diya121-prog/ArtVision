import React from 'react';

export const SkeletonCard: React.FC<{ className?: string; height?: string }> = ({
  className = '',
  height = 'h-48',
}) => {
  return (
    <div
      className={`rounded-2xl border border-stone-200/80 dark:border-stone-800 bg-white dark:bg-stone-900/60 p-6 shadow-sm animate-pulse space-y-4 ${className} ${height}`}
    >
      <div className="flex items-center justify-between">
        <div className="h-4 w-1/3 rounded-md bg-stone-200 dark:bg-stone-800" />
        <div className="h-5 w-16 rounded-full bg-stone-200 dark:bg-stone-800" />
      </div>
      <div className="space-y-3 pt-2">
        <div className="h-3 w-full rounded-md bg-stone-200/80 dark:bg-stone-800/80" />
        <div className="h-3 w-5/6 rounded-md bg-stone-200/60 dark:bg-stone-800/60" />
        <div className="h-3 w-4/6 rounded-md bg-stone-200/40 dark:bg-stone-800/40" />
      </div>
    </div>
  );
};

export const DashboardSkeleton: React.FC = () => {
  return (
    <div
      className="space-y-6 animate-pulse"
      role="status"
      aria-label="Analyzing artwork... please wait"
    >
      <div className="rounded-2xl border border-amber-300 dark:border-amber-900/40 bg-amber-50/50 dark:bg-amber-950/20 p-4 text-center">
        <div className="inline-flex items-center gap-2 text-sm font-medium text-amber-800 dark:text-amber-300">
          <svg
            className="animate-spin h-4 w-4 text-amber-700 dark:text-amber-400"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span>Extracting deep multi-task features and computing neural saliency...</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <SkeletonCard height="h-64" />
        <SkeletonCard height="h-64" />
        <SkeletonCard height="h-64" />
        <SkeletonCard height="h-56" />
        <SkeletonCard height="h-56" />
        <SkeletonCard height="h-56" />
      </div>

      <div className="rounded-2xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-6 shadow-sm h-80" />
    </div>
  );
};
