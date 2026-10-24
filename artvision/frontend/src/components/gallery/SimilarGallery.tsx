import React, { useState, useEffect } from 'react';
import { Images, Image as ImageIcon, X, ExternalLink, ZoomIn } from 'lucide-react';
import { SimilarArtwork } from '../../api/types';
import { Badge } from '../common/Badge';

export interface SimilarGalleryProps {
  similarArtworks: SimilarArtwork[] | null | undefined;
}

export const SimilarGallery: React.FC<SimilarGalleryProps> = ({ similarArtworks }) => {
  const [selectedArtwork, setSelectedArtwork] = useState<SimilarArtwork | null>(null);

  const hasArtworks = Boolean(similarArtworks && similarArtworks.length > 0);

  // Close modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedArtwork(null);
      }
    };
    if (selectedArtwork) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [selectedArtwork]);

  if (!hasArtworks) {
    return (
      <div className="rounded-3xl border border-dashed border-stone-300 dark:border-stone-800 bg-white/50 dark:bg-stone-900/40 p-8 text-center">
        <div className="mx-auto h-12 w-12 rounded-2xl bg-stone-100 dark:bg-stone-800 text-stone-400 flex items-center justify-center mb-3">
          <ImageIcon className="h-6 w-6" />
        </div>
        <h3 className="font-serif text-lg font-bold text-stone-800 dark:text-stone-200">
          Visually Similar Artworks
        </h3>
        <p className="text-xs text-stone-500 dark:text-stone-400 max-w-md mx-auto mt-1 leading-relaxed">
          Vector similarity search (k-NN visual indexing) is not active in this inference pass.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="rounded-3xl border border-stone-200/90 dark:border-stone-800 bg-white dark:bg-stone-900 p-6 sm:p-7 shadow-sm space-y-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
              <Images className="h-4 w-4" />
              <span>Visual Similarity Search</span>
            </div>
            <h3 className="font-serif text-2xl font-bold text-stone-900 dark:text-stone-100 mt-1">
              Related Artworks in Corpus
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">
              Click any artwork to inspect full resolution and catalog details
            </p>
          </div>

          <Badge variant="secondary">
            {similarArtworks?.length} Matches Found
          </Badge>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {similarArtworks?.map((art, idx) => (
            <div
              key={idx}
              role="button"
              tabIndex={0}
              onClick={() => setSelectedArtwork(art)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedArtwork(art);
                }
              }}
              className="group overflow-hidden rounded-2xl border border-stone-200/80 dark:border-stone-800 bg-stone-50/50 dark:bg-stone-950/40 transition-all duration-200 hover:-translate-y-1 hover:shadow-lg hover:border-amber-400 dark:hover:border-amber-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-amber-500 select-none"
            >
              <div className="relative aspect-[4/3] w-full overflow-hidden bg-stone-900">
                <img
                  src={art.image_url}
                  alt={art.title || 'Related visual'}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/70 backdrop-blur-sm text-white text-xs font-medium shadow">
                    <ZoomIn className="h-3.5 w-3.5" />
                    Click to Open
                  </span>
                </div>
                <div className="absolute top-2.5 right-2.5">
                  <Badge variant="primary" size="sm">
                    {(art.similarity * 100).toFixed(1)}% match
                  </Badge>
                </div>
              </div>

              <div className="p-3.5 space-y-1">
                <h4 className="font-serif text-sm font-bold text-stone-900 dark:text-stone-100 truncate">
                  {art.title || 'Untitled Artwork'}
                </h4>
                <p className="text-xs text-stone-600 dark:text-stone-300 truncate">
                  {art.artist || 'Unknown Artist'}
                </p>
                <p className="text-[11px] text-stone-400 dark:text-stone-500 truncate">
                  {(art.style || 'Classic').replace(/_/g, ' ')}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox / Modal for inspecting selected artwork */}
      {selectedArtwork && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-stone-950/80 backdrop-blur-md animate-in fade-in duration-200"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setSelectedArtwork(null);
            }
          }}
          role="dialog"
          aria-modal="true"
          aria-label={selectedArtwork.title || 'Artwork Details'}
        >
          <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col rounded-3xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-stone-100 dark:border-stone-800">
              <div className="flex items-center gap-2">
                <Images className="h-5 w-5 text-amber-700 dark:text-amber-400" />
                <h3 className="font-serif text-lg font-bold text-stone-900 dark:text-stone-100">
                  Visual Similarity Match
                </h3>
                <Badge variant="primary" size="sm" className="ml-2">
                  {(selectedArtwork.similarity * 100).toFixed(1)}% match
                </Badge>
              </div>
              <button
                type="button"
                onClick={() => setSelectedArtwork(null)}
                className="rounded-full p-2 text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
                aria-label="Close modal"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Image View */}
            <div className="relative flex-1 min-h-[280px] max-h-[58vh] bg-stone-950 flex items-center justify-center p-2 overflow-hidden">
              <img
                src={selectedArtwork.image_url}
                alt={selectedArtwork.title || 'Selected artwork'}
                className="max-h-full max-w-full object-contain rounded-lg shadow-lg"
              />
            </div>

            {/* Modal Footer / Details */}
            <div className="p-5 sm:p-6 bg-stone-50/60 dark:bg-stone-950/50 border-t border-stone-100 dark:border-stone-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="space-y-1 min-w-0">
                <h4 className="font-serif text-lg font-bold text-stone-900 dark:text-stone-100 truncate">
                  {selectedArtwork.title || 'Untitled Artwork'}
                </h4>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-stone-600 dark:text-stone-400">
                  <span>
                    <strong className="text-stone-800 dark:text-stone-200">Artist:</strong>{' '}
                    {selectedArtwork.artist || 'Unknown'}
                  </span>
                  <span>
                    <strong className="text-stone-800 dark:text-stone-200">Style:</strong>{' '}
                    {(selectedArtwork.style || 'Classic').replace(/_/g, ' ')}
                  </span>
                  {selectedArtwork.genre && (
                    <span>
                      <strong className="text-stone-800 dark:text-stone-200">Genre:</strong>{' '}
                      {selectedArtwork.genre.replace(/_/g, ' ')}
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-3 shrink-0 w-full sm:w-auto">
                <a
                  href={selectedArtwork.image_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-stone-700 dark:text-stone-200 bg-white dark:bg-stone-800 hover:bg-stone-100 dark:hover:bg-stone-700 border border-stone-200 dark:border-stone-700 shadow-sm transition-colors w-full sm:w-auto"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open Full Resolution
                </a>
                <button
                  type="button"
                  onClick={() => setSelectedArtwork(null)}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-amber-700 hover:bg-amber-800 shadow-sm transition-colors w-full sm:w-auto"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
