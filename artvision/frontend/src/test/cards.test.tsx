import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StyleCard } from '../components/dashboard/StyleCard';
import { ArtistCard } from '../components/dashboard/ArtistCard';
import { GenreCard } from '../components/dashboard/GenreCard';
import { MoodCard } from '../components/dashboard/MoodCard';
import { AuthenticityCard } from '../components/dashboard/AuthenticityCard';
import { EraTechniqueCard } from '../components/dashboard/EraTechniqueCard';
import { ColorPaletteCard } from '../components/dashboard/ColorPaletteCard';
import { ExplanationViewer } from '../components/explanation/ExplanationViewer';
import { SimilarGallery } from '../components/gallery/SimilarGallery';
import {
  confidentBaroqueFixture,
  uncertainArtistFixture,
  aiGeneratedFixture,
  genreNullColorErrorFixture,
} from '../api/mocks/fixtures';

describe('Results Dashboard Cards Component Suite', () => {
  describe('StyleCard', () => {
    it('renders confident style and derived movement', () => {
      render(
        <StyleCard
          styleData={confidentBaroqueFixture.style}
          movementData={confidentBaroqueFixture.movement}
        />
      );
      expect(screen.getByRole('heading', { name: 'Baroque' })).toBeInTheDocument();
      expect(screen.getByText('Baroque & Rococo')).toBeInTheDocument();
      expect(screen.getByText('Top Style Candidates')).toBeInTheDocument();
      expect(screen.queryByText('94% Match')).not.toBeInTheDocument();
      expect(screen.queryByText('Low Confidence')).not.toBeInTheDocument();
    });

    it('renders uncertain style cleanly without low confidence badge', () => {
      const uncertainStyle = {
        top3: [
          { style: 'Abstract_Expressionism', confidence: 0.31 },
          { style: 'Expressionism', confidence: 0.29 },
        ],
        prediction: 'Abstract_Expressionism',
        confidence: 0.31,
        uncertain: true,
      };
      render(<StyleCard styleData={uncertainStyle} movementData={null} />);
      expect(screen.getByRole('heading', { name: 'Abstract_Expressionism' })).toBeInTheDocument();
      expect(screen.queryByText('Low Confidence')).not.toBeInTheDocument();
    });

    it('renders fallback when styleData is null', () => {
      render(<StyleCard styleData={null} movementData={null} />);
      expect(screen.getByText(/Style classification is not available/i)).toBeInTheDocument();
    });
  });

  describe('ArtistCard', () => {
    it('renders artist attribution cleanly without confidence tags', () => {
      render(<ArtistCard artistData={confidentBaroqueFixture.artist} />);
      expect(screen.getByRole('heading', { name: 'Rembrandt van Rijn' })).toBeInTheDocument();
      expect(screen.queryByText(/Among the 23 artists/i)).not.toBeInTheDocument();
      expect(screen.queryByText('Low Confidence')).not.toBeInTheDocument();
      expect(screen.queryByText('90% Match')).not.toBeInTheDocument();
    });

    it('renders artist attribution without warning badge or alert box', () => {
      render(<ArtistCard artistData={uncertainArtistFixture.artist} />);
      expect(screen.getByRole('heading', { name: 'Claude Monet' })).toBeInTheDocument();
      expect(screen.queryByText(/Among the 23 artists/i)).not.toBeInTheDocument();
      expect(screen.queryByText('Low Confidence')).not.toBeInTheDocument();
      expect(screen.queryByText(/Uncertain attribution/i)).not.toBeInTheDocument();
    });

    it('renders graceful state when artist is null (e.g. AI-generated artwork)', () => {
      render(<ArtistCard artistData={null} />);
      expect(screen.getByText('Artist Not Identified')).toBeInTheDocument();
      expect(screen.getByText(/Artist prediction was masked/i)).toBeInTheDocument();
    });
  });

  describe('GenreCard', () => {
    it('renders genre data when present without % match badge', () => {
      render(<GenreCard genreData={confidentBaroqueFixture.genre} />);
      expect(screen.getByRole('heading', { name: 'Portrait' })).toBeInTheDocument();
      expect(screen.queryByText('82% Match')).not.toBeInTheDocument();
    });

    it('renders neutral not-available state when genre is null', () => {
      render(<GenreCard genreData={null} />);
      expect(screen.getByText('Not Available')).toBeInTheDocument();
      expect(screen.getByText(/Genre head unannotated or disabled/i)).toBeInTheDocument();
    });
  });

  describe('MoodCard', () => {
    it('renders mood distribution bars without experimental or dominant badges', () => {
      render(<MoodCard moodData={confidentBaroqueFixture.mood} />);
      expect(screen.getByRole('heading', { name: 'Tense / Dark' })).toBeInTheDocument();
      expect(screen.queryByText('Experimental')).not.toBeInTheDocument();
      expect(screen.queryByText('71% Dominant')).not.toBeInTheDocument();
      expect(screen.getByText('Awe / Sublime')).toBeInTheDocument();
      expect(screen.getByText('Calm / Content')).toBeInTheDocument();
    });

    it('renders fallback when mood is null', () => {
      render(<MoodCard moodData={null} />);
      expect(screen.getByText(/Mood analysis unavailable/i)).toBeInTheDocument();
    });
  });

  describe('AuthenticityCard', () => {
    it('renders human-crafted prediction with gauge and disclaimer without confident tag', () => {
      render(<AuthenticityCard authenticityData={confidentBaroqueFixture.authenticity} />);
      expect(screen.getByRole('heading', { name: 'Human-Crafted' })).toBeInTheDocument();
      expect(screen.queryByText(/% Confident/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/Trained on specific generators/i)).not.toBeInTheDocument();
    });

    it('renders AI-generated prediction without confident tag', () => {
      render(<AuthenticityCard authenticityData={aiGeneratedFixture.authenticity} />);
      expect(screen.getByRole('heading', { name: 'AI-Generated' })).toBeInTheDocument();
      expect(screen.queryByText(/% Confident/i)).not.toBeInTheDocument();
    });

    it('renders fallback when authenticity is null', () => {
      render(<AuthenticityCard authenticityData={null} />);
      expect(screen.getByText(/Authenticity evaluation is currently unavailable/i)).toBeInTheDocument();
    });
  });

  describe('EraTechniqueCard', () => {
    it('renders derived era timeline and physical technique', () => {
      render(
        <EraTechniqueCard
          eraData={confidentBaroqueFixture.era}
          techniqueData={confidentBaroqueFixture.technique}
        />
      );
      expect(screen.getByText('Baroque Era')).toBeInTheDocument();
      expect(screen.getByText('17th – Early 18th Century (c. 1600–1750)')).toBeInTheDocument();
      expect(screen.getByText('Oil on canvas')).toBeInTheDocument();
      expect(screen.getByText(/Tenebrism, dramatic directional lighting/i)).toBeInTheDocument();
    });

    it('renders fallback defaults when data is null', () => {
      render(<EraTechniqueCard eraData={null} techniqueData={null} />);
      expect(screen.getByText('Historical Timeline & Medium')).toBeInTheDocument();
    });
  });

  describe('ColorPaletteCard', () => {
    it('renders color swatches, temperature scores, and tone descriptors', () => {
      render(
        <ColorPaletteCard
          colorData={confidentBaroqueFixture.color}
          colorError={confidentBaroqueFixture.color_error}
        />
      );
      expect(screen.getByText('warm Harmony')).toBeInTheDocument();
      expect(screen.getByText(/Warm: 88%/i)).toBeInTheDocument();
      expect(screen.getByText(/Cool: 12%/i)).toBeInTheDocument();
      expect(screen.getByText('#16110B')).toBeInTheDocument();
    });

    it('renders error notice when color extraction fails', () => {
      render(
        <ColorPaletteCard
          colorData={genreNullColorErrorFixture.color}
          colorError={genreNullColorErrorFixture.color_error}
        />
      );
      expect(screen.getByText('Extraction Failed')).toBeInTheDocument();
      expect(screen.getByText(/CIELAB clustering failed/i)).toBeInTheDocument();
    });
  });

  describe('ExplanationViewer (Grad-CAM)', () => {
    it('renders visualizer with non-causal disclaimer wording when explanation exists', () => {
      render(
        <ExplanationViewer
          explanation={confidentBaroqueFixture.explanation}
          originalImage="data:image/svg+xml;utf8,<svg></svg>"
        />
      );
      expect(screen.getByText('Visual Saliency Analysis')).toBeInTheDocument();
      expect(screen.getByText(/The model focused mostly on these regions/i)).toBeInTheDocument();
      expect(screen.queryByText(/causal proof/i)).toBeInTheDocument();
    });

    it('renders tidy placeholder when explanation is null or empty', () => {
      render(
        <ExplanationViewer
          explanation={null}
          originalImage="data:image/svg+xml;utf8,<svg></svg>"
        />
      );
      expect(screen.getByText(/Grad-CAM Visual Explanations/i)).toBeInTheDocument();
      expect(screen.getByText(/Saliency heatmaps were not computed/i)).toBeInTheDocument();
    });
  });

  describe('SimilarGallery', () => {
    it('renders similar artwork cards with similarity percentages', () => {
      render(<SimilarGallery similarArtworks={confidentBaroqueFixture.similar} />);
      expect(screen.getByText('Related Artworks in Corpus')).toBeInTheDocument();
      expect(screen.getByText('The Anatomy Lesson of Dr. Nicolaes Tulp')).toBeInTheDocument();
      expect(screen.getByText('92.4% match')).toBeInTheDocument();
    });

    it('renders placeholder when similar artworks are null', () => {
      render(<SimilarGallery similarArtworks={null} />);
      expect(screen.getByText('Visually Similar Artworks')).toBeInTheDocument();
      expect(screen.getByText(/Vector similarity search/i)).toBeInTheDocument();
    });
  });
});
