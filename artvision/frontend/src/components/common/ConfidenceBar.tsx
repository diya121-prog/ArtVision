import React from 'react';

export interface ConfidenceBarProps {
  label: string;
  confidence: number; // 0.0 to 1.0
  isTop?: boolean;
  isUncertain?: boolean;
  barColor?: string;
  className?: string;
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  label,
  confidence,
  isTop = false,
  isUncertain = false,
  barColor,
  className = '',
}) => {
  const percentage = Math.round(confidence * 100);
  const formattedPct = (confidence * 100).toFixed(1);

  // Accessible color thresholds with curatorial palette
  const getDefaultBarColor = () => {
    if (isUncertain) return 'bg-amber-600 dark:bg-amber-500';
    if (isTop) return 'bg-gradient-to-r from-amber-600 to-amber-700 dark:from-amber-500 dark:to-amber-600';
    if (percentage > 50) return 'bg-stone-700 dark:bg-stone-400';
    return 'bg-stone-300 dark:bg-stone-600';
  };

  const activeColor = barColor || getDefaultBarColor();

  return (
    <div className={`space-y-1.5 ${className}`}>
      <div className="flex items-center justify-between text-xs sm:text-sm">
        <span
          className={`truncate font-medium ${
            isTop
              ? 'text-stone-900 dark:text-stone-100 font-semibold'
              : 'text-stone-600 dark:text-stone-400'
          }`}
          title={label}
        >
          {label}
        </span>
        <span
          className={`tabular-nums font-mono font-medium ${
            isTop
              ? 'text-stone-900 dark:text-stone-100 font-bold'
              : 'text-stone-500 dark:text-stone-400'
          }`}
        >
          {formattedPct}%
        </span>
      </div>

      <div
        role="meter"
        aria-label={`${label} confidence`}
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 w-full overflow-hidden rounded-full bg-stone-100 dark:bg-stone-800"
      >
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${activeColor}`}
          style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
        />
      </div>
    </div>
  );
};
