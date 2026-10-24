/**
 * ArtVision API TypeScript Types
 * Corresponds to predict.py multi-task output schema & FastAPI endpoints (schema_version: "1.0")
 */

export interface ClassConfidence {
  [key: string]: string | number;
  confidence: number;
}

export interface StyleItem {
  style: string;
  confidence: number;
}

export interface ArtistItem {
  artist: string;
  confidence: number;
}

export interface GenreItem {
  genre: string;
  confidence: number;
}

export interface StylePrediction {
  top3: StyleItem[];
  prediction: string;
  confidence: number;
  uncertain: boolean;
}

/**
 * Historical art movement derived as a coarse period group from predicted style:
 * 1. Renaissance & Mannerism
 * 2. Baroque & Rococo
 * 3. 19th-Century Realism & Romanticism
 * 4. Late 19th-Century Avant-Garde
 * 5. Early 20th-Century Modernism
 * 6. Post-War & Contemporary Art
 * 7. Traditional East Asian Art
 */
export interface MovementPrediction {
  derived_from_style: string;
  movement: string;
}

export interface ArtistPrediction {
  top3: ArtistItem[];
  prediction: string;
  confidence: number;
  uncertain: boolean;
}

export interface GenrePrediction {
  top3: GenreItem[];
  prediction: string;
  confidence: number;
  uncertain: boolean;
}

export interface MoodPrediction {
  prediction: string;
  confidence: number;
  distribution: Record<string, number>;
  uncertain: boolean;
  experimental?: boolean; // Real API schema includes experimental: true
}

export interface AuthenticityPrediction {
  is_ai_generated: boolean;
  label: 'ai_generated' | 'human_made';
  probability_ai: number;
  confidence: number;
  uncertain: boolean;
  caveat?: string; // Real API schema caveat message
}

export interface ColorSwatch {
  hex: string;
  rgb: [number, number, number];
  percent: number;
}

export interface ColorAnalysis {
  palette: ColorSwatch[];
  color_count: number;
  temperature: 'warm' | 'cool' | 'neutral-mixed' | 'monochrome' | string;
  warm_score: number;
  cool_score: number;
  brightness: 'dark' | 'mid' | 'light' | string;
  saturation: 'muted' | 'balanced' | 'vivid' | string;
  is_pastel: boolean;
}

export interface HeadExplanation {
  target_class?: string;
  heatmap_url?: string;
  overlay_url: string;
  task?: string;
  target_prediction?: string;
  border_mass_fraction?: number;
  heatmap_dimensions?: number[];
  stats?: {
    mean_activation?: number;
    max_activation?: number;
    [key: string]: number | undefined;
  };
  error?: string;
}

export type ExplanationHeadKey = 'style' | 'mood' | 'authenticity' | 'artist' | 'genre';

export type ExplanationMap = {
  [head in ExplanationHeadKey]?: HeadExplanation;
};

// Real API returns either a single explanation object or a map
export type UnifiedExplanation = HeadExplanation | ExplanationMap;

export interface SimilarArtwork {
  image_id?: string;
  image_url: string;
  title?: string;
  artist: string;
  style: string;
  genre?: string;
  similarity: number;
  is_ai_generated?: boolean;
}

export interface EraPrediction {
  period: string;
  timeline: string;
  derived_from?: string;
}

export interface TechniquePrediction {
  medium: string;
  method: string;
  derived_from?: string;
}

export interface PredictResponse {
  schema_version: '1.0' | '1.0.0' | string;
  style: StylePrediction;
  movement: MovementPrediction | null;
  artist: ArtistPrediction | null;
  genre: GenrePrediction | null;
  mood: MoodPrediction | null;
  authenticity: AuthenticityPrediction | null;
  era: EraPrediction | null;
  technique: TechniquePrediction | null;
  color: ColorAnalysis | null;
  color_error?: string | null;
  explanation?: UnifiedExplanation | null;
  similar?: SimilarArtwork[] | null;
}

export interface ApiError {
  error: string;
  details?: string;
  status?: number;
}

export type MockScenarioKey =
  | 'confidentBaroque'
  | 'uncertainArtist'
  | 'aiGenerated'
  | 'genreNullColorError'
  | 'highUncertaintyAll'
  | 'error500'
  | 'errorNetwork'
  | 'errorInvalidFile';
