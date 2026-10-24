import React, { useState } from 'react';
import { Pipette, Copy, Check, Thermometer, SunMedium, AlertCircle } from 'lucide-react';
import { ColorAnalysis } from '../../api/types';
import { Badge } from '../common/Badge';

export interface ColorPaletteCardProps {
  colorData: ColorAnalysis | null;
  colorError?: string | null;
}

export const ColorPaletteCard: React.FC<ColorPaletteCardProps> = ({
  colorData,
  colorError,
}) => {
  const [copiedHex, setCopiedHex] = useState<string | null>(null);

  const handleCopy = (hex: string) => {
    navigator.clipboard.writeText(hex);
    setCopiedHex(hex);
    setTimeout(() => {
      setCopiedHex((current) => (current === hex ? null : current));
    }, 2000);
  };

  if (!colorData) {
    return (
      <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-stone-400 dark:text-stone-500">
          <Pipette className="h-4 w-4" />
          <span>Color & Tone Analysis</span>
        </div>
        <h3 className="font-serif text-xl font-bold text-stone-400 dark:text-stone-500 mt-1">
          Extraction Failed
        </h3>

        <div className="mt-4 rounded-2xl bg-amber-50 dark:bg-amber-950/30 p-4 border border-amber-200 dark:border-amber-900/50 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-800 dark:text-amber-300">
            <p className="font-semibold">Color clustering error</p>
            <p className="mt-1 opacity-90">{colorError || 'Unable to extract color palette.'}</p>
          </div>
        </div>
      </div>
    );
  }

  const {
    palette,
    temperature,
    warm_score,
    cool_score,
    brightness,
    saturation,
    is_pastel,
  } = colorData;

  const tempBadgeVariant =
    temperature === 'warm'
      ? 'warning'
      : temperature === 'cool'
      ? 'primary'
      : 'neutral';

  return (
    <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 shadow-sm space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
            <Pipette className="h-4 w-4" />
            <span>Dominant Palette & Tone</span>
          </div>
          <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1 capitalize">
            {temperature} Harmony
          </h3>
        </div>

        <Badge variant={tempBadgeVariant} className="capitalize">
          {temperature}
        </Badge>
      </div>

      {/* Swatches Grid */}
      <div className="space-y-2">
        <div className="flex h-12 w-full overflow-hidden rounded-2xl shadow-inner ring-1 ring-black/5 dark:ring-white/10">
          {palette.map((swatch) => (
            <div
              key={swatch.hex}
              style={{
                backgroundColor: swatch.hex,
                width: `${Math.max(5, swatch.percent * 100)}%`,
              }}
              title={`${swatch.hex} (${(swatch.percent * 100).toFixed(0)}%)`}
              className="h-full transition-all hover:opacity-95"
            />
          ))}
        </div>

        {/* Swatch detail chips */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 pt-1">
          {palette.map((swatch) => {
            const isCopied = copiedHex === swatch.hex;
            return (
              <button
                key={swatch.hex}
                type="button"
                onClick={() => handleCopy(swatch.hex)}
                className="group relative flex flex-col items-center justify-center p-2 rounded-xl bg-stone-50 dark:bg-stone-800/60 border border-stone-200/80 dark:border-stone-700/60 hover:border-amber-500 dark:hover:border-amber-400 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-500"
                aria-label={`Copy hex code ${swatch.hex}`}
              >
                <div className="flex items-center gap-1.5 w-full justify-between">
                  <span
                    className="h-3.5 w-3.5 rounded-full ring-1 ring-black/10 shrink-0"
                    style={{ backgroundColor: swatch.hex }}
                  />
                  <span className="font-mono text-xs font-semibold text-stone-800 dark:text-stone-200">
                    {swatch.hex.toUpperCase()}
                  </span>
                  <span className="text-stone-400 group-hover:text-amber-700 dark:group-hover:text-amber-400 shrink-0">
                    {isCopied ? (
                      <Check className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </span>
                </div>
                <div className="w-full text-right mt-1 text-[10px] font-medium text-stone-400 dark:text-stone-500">
                  {(swatch.percent * 100).toFixed(0)}%
                </div>

                {isCopied && (
                  <span className="absolute -top-6 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded bg-stone-900 text-white text-[10px] shadow">
                    Copied!
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Temperature Warm vs Cool Score Bar */}
      <div className="rounded-2xl bg-stone-50 dark:bg-stone-800/50 p-3.5 border border-stone-200/80 dark:border-stone-800 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1 font-semibold text-stone-700 dark:text-stone-300">
            <Thermometer className="h-3.5 w-3.5 text-amber-600" />
            Color Temperature Balance
          </span>
          <span className="font-mono text-stone-500 text-[11px]">
            Warm: {(warm_score * 100).toFixed(0)}% · Cool: {(cool_score * 100).toFixed(0)}%
          </span>
        </div>

        <div className="h-2.5 w-full rounded-full bg-stone-200 dark:bg-stone-700 overflow-hidden flex">
          <div
            className="h-full bg-gradient-to-r from-amber-500 to-orange-600 transition-all duration-500"
            style={{ width: `${warm_score * 100}%` }}
          />
          <div
            className="h-full bg-gradient-to-r from-teal-500 to-cyan-700 transition-all duration-500"
            style={{ width: `${cool_score * 100}%` }}
          />
        </div>
      </div>

      {/* Tone Descriptors */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <span className="text-xs font-semibold text-stone-400 dark:text-stone-500 flex items-center gap-1">
          <SunMedium className="h-3.5 w-3.5" />
          Descriptors:
        </span>
        <Badge variant="secondary" size="sm">
          Brightness: <span className="font-bold capitalize ml-1">{brightness}</span>
        </Badge>
        <Badge variant="secondary" size="sm">
          Saturation: <span className="font-bold capitalize ml-1">{saturation}</span>
        </Badge>
        {is_pastel && (
          <Badge variant="success" size="sm">
            Pastel Palette
          </Badge>
        )}
      </div>
    </div>
  );
};
