import React from 'react';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'warning' | 'danger' | 'success' | 'neutral' | 'subtle';
  size?: 'sm' | 'md';
  className?: string;
  icon?: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'secondary',
  size = 'md',
  className = '',
  icon,
}) => {
  const variantStyles = {
    primary: 'bg-amber-50 text-amber-900 dark:bg-amber-950/60 dark:text-amber-200 border-amber-200/90 dark:border-amber-800/80',
    secondary: 'bg-stone-100 text-stone-800 dark:bg-stone-800/80 dark:text-stone-200 border-stone-200/90 dark:border-stone-700/80',
    warning: 'bg-amber-50 text-amber-900 dark:bg-amber-950/70 dark:text-amber-300 border-amber-300 dark:border-amber-700/80',
    danger: 'bg-rose-50 text-rose-900 dark:bg-rose-950/70 dark:text-rose-200 border-rose-200 dark:border-rose-900/60',
    success: 'bg-emerald-50 text-emerald-900 dark:bg-emerald-950/70 dark:text-emerald-200 border-emerald-200 dark:border-emerald-800/80',
    neutral: 'bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300 border-stone-200 dark:border-stone-700',
    subtle: 'bg-transparent text-stone-600 dark:text-stone-400 border-stone-200 dark:border-stone-800',
  };

  const sizeStyles = {
    sm: 'text-xs px-2 py-0.5 font-medium',
    md: 'text-xs px-2.5 py-1 font-semibold tracking-wide',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border transition-colors ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      <span>{children}</span>
    </span>
  );
};
