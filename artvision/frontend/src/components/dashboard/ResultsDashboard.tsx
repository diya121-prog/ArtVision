import React from 'react';
import { PredictResponse } from '../../api/types';
import { StyleCard } from './StyleCard';
import { ArtistCard } from './ArtistCard';
import { GenreCard } from './GenreCard';
import { MoodCard } from './MoodCard';
import { AuthenticityCard } from './AuthenticityCard';
import { EraTechniqueCard } from './EraTechniqueCard';
import { ColorPaletteCard } from './ColorPaletteCard';
import { ExplanationViewer } from '../explanation/ExplanationViewer';
import { SimilarGallery } from '../gallery/SimilarGallery';

export interface ResultsDashboardProps {
  data: PredictResponse;
  previewImage: string;
  onSelectHead?: (head: string) => void;
  isLoadingHead?: boolean;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  data,
  previewImage,
  onSelectHead,
  isLoadingHead = false,
}) => {
  return (
    <div className="space-y-8 animate-fadeIn" role="region" aria-label="Artwork Analysis Results">
      {/* Primary: Core Attribution (Style & Movement, Artist, Authenticity) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <StyleCard styleData={data.style} movementData={data.movement} />
        <ArtistCard artistData={data.artist} />
        <AuthenticityCard authenticityData={data.authenticity} />
      </div>

      {/* Secondary: Historical & Contextual (Genre, Mood, Era & Technique) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <GenreCard genreData={data.genre} />
        <MoodCard moodData={data.mood} />
        <EraTechniqueCard eraData={data.era} techniqueData={data.technique} />
      </div>

      {/* Tertiary: Color & Tone Analysis */}
      <ColorPaletteCard
        colorData={data.color}
        colorError={data.color_error}
      />

      {/* Quaternary: Explainability & Attention (Grad-CAM Saliency) */}
      <ExplanationViewer
        explanation={data.explanation}
        originalImage={previewImage}
        onSelectHead={onSelectHead}
        isLoadingHead={isLoadingHead}
      />

      {/* Quinary: Visually Similar Artworks Gallery */}
      <SimilarGallery similarArtworks={data.similar} />
    </div>
  );
};
