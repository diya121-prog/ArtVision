import React from 'react';

export interface GaugeProps {
  value: number; // 0.0 (Human) to 1.0 (AI)
  label?: string;
  isAi: boolean;
  uncertain?: boolean;
}

export const Gauge: React.FC<GaugeProps> = ({
  value,
  label = 'AI Probability Gauge',
  isAi,
  uncertain = false,
}) => {
  const percentage = Math.min(100, Math.max(0, Math.round(value * 100)));
  
  // Angle for semi-circle: 180 degrees from -180 to 0 (or -90 to +90)
  // Let angle sweep from -180 deg (left = 0% AI) to 0 deg (right = 100% AI)
  const angle = -180 + (percentage / 100) * 180;
  
  // Needle coordinates for radius = 70, center at (100, 95)
  const cx = 100;
  const cy = 95;
  const r = 70;
  const rad = (angle * Math.PI) / 180;
  const needleLength = 58;
  const nx = cx + needleLength * Math.cos(rad);
  const ny = cy + needleLength * Math.sin(rad);

  // SVG arc path for radius 70 from -180 to 0 deg:
  // start: (30, 95), end: (170, 95)
  const arcPath = `M 30,95 A 70,70 0 0,1 170,95`;

  // Perimeter of semi-circle arc: PI * r ≈ 219.9
  const arcLength = Math.PI * r;
  const strokeDashoffset = arcLength * (1 - percentage / 100);

  return (
    <div
      className="flex flex-col items-center justify-center"
      role="meter"
      aria-label={label}
      aria-valuenow={percentage}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="relative w-48 h-28 flex items-center justify-center">
        <svg
          viewBox="0 0 200 115"
          className="w-full h-full overflow-visible"
        >
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2d6a4f" />
              <stop offset="50%" stopColor="#d97706" />
              <stop offset="100%" stopColor="#991b1b" />
            </linearGradient>
          </defs>

          {/* Background track */}
          <path
            d={arcPath}
            fill="none"
            stroke="currentColor"
            strokeWidth="14"
            strokeLinecap="round"
            className="text-stone-200 dark:text-stone-800"
          />

          {/* Active colored arc */}
          <path
            d={arcPath}
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={arcLength}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-700 ease-out"
          />

          {/* Needle pivot circle */}
          <circle cx={cx} cy={cy} r="6" className="fill-stone-700 dark:fill-stone-300" />
          
          {/* Indicator needle line */}
          <line
            x1={cx}
            y1={cy}
            x2={nx}
            y2={ny}
            stroke="currentColor"
            strokeWidth="3.5"
            strokeLinecap="round"
            className="text-stone-800 dark:text-stone-100 transition-all duration-700 ease-out"
          />
          <circle cx={nx} cy={ny} r="3" className="fill-stone-900 dark:fill-white" />
        </svg>
      </div>

      {/* Numerical percentage badge */}
      <div className="flex items-center gap-1.5 -mt-3">
        <span className="text-2xl font-extrabold tracking-tight text-stone-900 dark:text-stone-100">
          {(value * 100).toFixed(1)}%
        </span>
        <span className="text-xs text-stone-500 dark:text-stone-400 font-medium">
          synthetic probability
        </span>
      </div>

      {/* Axis labels */}
      <div className="w-full max-w-[210px] flex justify-between text-[11px] font-semibold text-stone-500 dark:text-stone-400 mt-1 px-1">
        <span className={!isAi && !uncertain ? 'text-emerald-700 dark:text-emerald-400 font-bold' : ''}>
          Human-made
        </span>
        <span className={isAi && !uncertain ? 'text-rose-700 dark:text-rose-400 font-bold' : ''}>
          AI-generated
        </span>
      </div>
    </div>
  );
};
