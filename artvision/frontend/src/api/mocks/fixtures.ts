import { PredictResponse } from '../types';
import {
  createMockArtworkSvg,
  createMockHeatmapSvg,
  createMockOverlaySvg,
} from './generator';

// Pre-generate SVG placeholders
const baroqueArtSvg = createMockArtworkSvg('The Night Watch', 'Baroque', [
  '#0f0b08',
  '#3d2714',
  '#b47d34',
  '#e6c37a',
]);
const aiArtSvg = createMockArtworkSvg('Synthetic Dreamscape', 'Surrealism', [
  '#180033',
  '#6b21a8',
  '#ec4899',
  '#06b6d4',
]);

// 1. Confident Baroque Result
export const confidentBaroqueFixture: PredictResponse = {
  schema_version: '1.0.0',
  style: {
    top3: [
      { style: 'Baroque', confidence: 0.9421 },
      { style: 'Renaissance', confidence: 0.0384 },
      { style: 'Rococo', confidence: 0.0125 },
    ],
    prediction: 'Baroque',
    confidence: 0.9421,
    uncertain: false,
  },
  movement: {
    derived_from_style: 'Baroque',
    movement: 'Baroque & Rococo',
  },
  artist: {
    top3: [
      { artist: 'Rembrandt van Rijn', confidence: 0.8954 },
      { artist: 'Peter Paul Rubens', confidence: 0.0612 },
      { artist: 'Caravaggio', confidence: 0.0289 },
    ],
    prediction: 'Rembrandt van Rijn',
    confidence: 0.8954,
    uncertain: false,
  },
  genre: {
    top3: [
      { genre: 'Portrait', confidence: 0.8214 },
      { genre: 'History Painting', confidence: 0.1245 },
      { genre: 'Genre Art', confidence: 0.0351 },
    ],
    prediction: 'Portrait',
    confidence: 0.8214,
    uncertain: false,
  },
  mood: {
    prediction: 'Tense / Dark',
    confidence: 0.7134,
    distribution: {
      'Tense / Dark': 0.7134,
      'Awe / Sublime': 0.1652,
      'Sad / Melancholic': 0.0781,
      'Calm / Content': 0.0298,
      'Joyful / Excited': 0.0135,
    },
    uncertain: false,
  },
  authenticity: {
    is_ai_generated: false,
    label: 'human_made',
    probability_ai: 0.0162,
    confidence: 0.9838,
    uncertain: false,
  },
  era: {
    period: 'Baroque Era',
    timeline: '17th – Early 18th Century (c. 1600–1750)',
    derived_from: 'Baroque',
  },
  technique: {
    medium: 'Oil on canvas',
    method: 'Tenebrism, dramatic directional lighting, bold impasto highlights',
    derived_from: 'Baroque',
  },
  color: {
    palette: [
      { hex: '#16110b', rgb: [22, 17, 11], percent: 0.44 },
      { hex: '#3d2514', rgb: [61, 37, 20], percent: 0.26 },
      { hex: '#7b4c21', rgb: [123, 76, 33], percent: 0.16 },
      { hex: '#c59546', rgb: [197, 149, 70], percent: 0.09 },
      { hex: '#ecd8ab', rgb: [236, 216, 171], percent: 0.05 },
    ],
    color_count: 5,
    temperature: 'warm',
    warm_score: 0.88,
    cool_score: 0.12,
    brightness: 'dark',
    saturation: 'vivid',
    is_pastel: false,
  },
  explanation: {
    style: {
      target_class: 'Baroque',
      heatmap_url: createMockHeatmapSvg('style', '#f59e0b'),
      overlay_url: createMockOverlaySvg(baroqueArtSvg, createMockHeatmapSvg('style', '#f59e0b')),
      stats: { mean_activation: 0.482, max_activation: 0.964 },
    },
    mood: {
      target_class: 'Tense / Dark',
      heatmap_url: createMockHeatmapSvg('mood', '#ef4444'),
      overlay_url: createMockOverlaySvg(baroqueArtSvg, createMockHeatmapSvg('mood', '#ef4444')),
      stats: { mean_activation: 0.514, max_activation: 0.991 },
    },
    authenticity: {
      target_class: 'human_made',
      heatmap_url: createMockHeatmapSvg('authenticity', '#10b981'),
      overlay_url: createMockOverlaySvg(baroqueArtSvg, createMockHeatmapSvg('authenticity', '#10b981')),
      stats: { mean_activation: 0.389, max_activation: 0.871 },
    },
    artist: {
      target_class: 'Rembrandt van Rijn',
      heatmap_url: createMockHeatmapSvg('artist', '#8b5cf6'),
      overlay_url: createMockOverlaySvg(baroqueArtSvg, createMockHeatmapSvg('artist', '#8b5cf6')),
      stats: { mean_activation: 0.445, max_activation: 0.923 },
    },
  },
  similar: [
    {
      image_url: createMockArtworkSvg('The Anatomy Lesson', 'Baroque', ['#111', '#422', '#c94']),
      title: 'The Anatomy Lesson of Dr. Nicolaes Tulp',
      artist: 'Rembrandt van Rijn',
      style: 'Baroque',
      similarity: 0.924,
    },
    {
      image_url: createMockArtworkSvg('Self-Portrait with Two Circles', 'Baroque', ['#1a120b', '#5c3a21', '#d4af37']),
      title: 'Self-Portrait with Two Circles',
      artist: 'Rembrandt van Rijn',
      style: 'Baroque',
      similarity: 0.887,
    },
    {
      image_url: createMockArtworkSvg('The Calling of Saint Matthew', 'Baroque', ['#0f0d14', '#553c25', '#e09f3e']),
      title: 'The Calling of Saint Matthew',
      artist: 'Caravaggio',
      style: 'Baroque',
      similarity: 0.841,
    },
    {
      image_url: createMockArtworkSvg('The Elevation of the Cross', 'Baroque', ['#1e140d', '#61381b', '#d9a74a']),
      title: 'The Elevation of the Cross',
      artist: 'Peter Paul Rubens',
      style: 'Baroque',
      similarity: 0.796,
    },
  ],
};

// 2. Uncertain Artist (Low confidence badge, softened styling, null genre, cool pastel palette)
export const uncertainArtistFixture: PredictResponse = {
  schema_version: '1.0.0',
  style: {
    top3: [
      { style: 'Impressionism', confidence: 0.7842 },
      { style: 'Post_Impressionism', confidence: 0.1423 },
      { style: 'Realism', confidence: 0.0435 },
    ],
    prediction: 'Impressionism',
    confidence: 0.7842,
    uncertain: false,
  },
  movement: {
    derived_from_style: 'Impressionism',
    movement: 'Late 19th-Century Avant-Garde',
  },
  artist: {
    top3: [
      { artist: 'Claude Monet', confidence: 0.3421 },
      { artist: 'Pierre-Auguste Renoir', confidence: 0.3189 },
      { artist: 'Camille Pissarro', confidence: 0.1874 },
    ],
    prediction: 'Claude Monet',
    confidence: 0.3421,
    uncertain: true, // Marked uncertain: small margin between Monet and Renoir
  },
  genre: null, // Null genre (simulates unannotated / absent genre head)
  mood: {
    prediction: 'Calm / Content',
    confidence: 0.6452,
    distribution: {
      'Calm / Content': 0.6452,
      'Joyful / Excited': 0.2145,
      'Awe / Sublime': 0.0892,
      'Sad / Melancholic': 0.0341,
      'Tense / Dark': 0.0170,
    },
    uncertain: false,
  },
  authenticity: {
    is_ai_generated: false,
    label: 'human_made',
    probability_ai: 0.0789,
    confidence: 0.9211,
    uncertain: false,
  },
  era: {
    period: 'Impressionist Belle Époque',
    timeline: 'Late 19th Century (c. 1870–1890)',
    derived_from: 'Impressionism',
  },
  technique: {
    medium: 'Oil on primed canvas (plein-air)',
    method: 'Broken brushstrokes, unblended pure pigment, transient optical mixing',
    derived_from: 'Impressionism',
  },
  color: {
    palette: [
      { hex: '#7eaec7', rgb: [126, 174, 199], percent: 0.38 },
      { hex: '#a8d5ba', rgb: [168, 213, 186], percent: 0.32 },
      { hex: '#e8d5b5', rgb: [232, 213, 181], percent: 0.18 },
      { hex: '#f0c2d3', rgb: [240, 194, 211], percent: 0.12 },
    ],
    color_count: 4,
    temperature: 'cool',
    warm_score: 0.21,
    cool_score: 0.79,
    brightness: 'light',
    saturation: 'muted',
    is_pastel: true,
  },
  explanation: null, // Test graceful placeholder when explanation is null
  similar: null, // Test graceful placeholder when similar is null
};

// 3. AI Generated Artwork
export const aiGeneratedFixture: PredictResponse = {
  schema_version: '1.0.0',
  style: {
    top3: [
      { style: 'Surrealism', confidence: 0.6214 },
      { style: 'Expressionism', confidence: 0.2415 },
      { style: 'Art_Nouveau', confidence: 0.0872 },
    ],
    prediction: 'Surrealism',
    confidence: 0.6214,
    uncertain: false,
  },
  movement: {
    derived_from_style: 'Surrealism',
    movement: 'Late 19th-Century Avant-Garde',
  },
  artist: null, // Masked artist (AI generated has no human artist)
  genre: {
    top3: [
      { genre: 'Landscape', confidence: 0.7412 },
      { genre: 'Mythological Painting', confidence: 0.1824 },
      { genre: 'Abstract', confidence: 0.0451 },
    ],
    prediction: 'Landscape',
    confidence: 0.7412,
    uncertain: false,
  },
  mood: {
    prediction: 'Joyful / Excited',
    confidence: 0.5841,
    distribution: {
      'Joyful / Excited': 0.5841,
      'Awe / Sublime': 0.2814,
      'Calm / Content': 0.0712,
      'Tense / Dark': 0.0421,
      'Sad / Melancholic': 0.0212,
    },
    uncertain: false,
  },
  authenticity: {
    is_ai_generated: true,
    label: 'ai_generated',
    probability_ai: 0.9423,
    confidence: 0.9423,
    uncertain: false,
  },
  era: {
    period: 'Contemporary Digital / Synthetic Synthesis',
    timeline: '2020s Generative AI Era',
    derived_from: 'AI Authenticity Head',
  },
  technique: {
    medium: 'Latent Diffusion / Deep Neural Synthesis',
    method: 'Iterative denoising in latent feature space (text-to-image synthesis)',
    derived_from: 'AI Authenticity Head',
  },
  color: {
    palette: [
      { hex: '#6b21a8', rgb: [107, 33, 168], percent: 0.35 },
      { hex: '#ec4899', rgb: [236, 72, 153], percent: 0.30 },
      { hex: '#06b6d4', rgb: [6, 182, 212], percent: 0.22 },
      { hex: '#facc15', rgb: [250, 204, 21], percent: 0.13 },
    ],
    color_count: 4,
    temperature: 'neutral-mixed',
    warm_score: 0.52,
    cool_score: 0.48,
    brightness: 'mid',
    saturation: 'vivid',
    is_pastel: false,
  },
  explanation: {
    authenticity: {
      target_class: 'ai_generated',
      heatmap_url: createMockHeatmapSvg('authenticity', '#ef4444'),
      overlay_url: createMockOverlaySvg(aiArtSvg, createMockHeatmapSvg('authenticity', '#ef4444')),
      stats: { mean_activation: 0.631, max_activation: 0.988 },
    },
  },
  similar: [
    {
      image_url: createMockArtworkSvg('Diffusion Dream 01', 'Surrealism', ['#3b0764', '#d946ef', '#06b6d4']),
      title: 'Latent Space Drift',
      artist: 'Midjourney v6',
      style: 'Surrealism',
      similarity: 0.884,
    },
  ],
};

// 4. Genre Null + Color Error (Tests error boundary and fallback states)
export const genreNullColorErrorFixture: PredictResponse = {
  schema_version: '1.0.0',
  style: {
    top3: [
      { style: 'Realism', confidence: 0.8124 },
      { style: 'Romanticism', confidence: 0.1142 },
      { style: 'Impressionism', confidence: 0.0451 },
    ],
    prediction: 'Realism',
    confidence: 0.8124,
    uncertain: false,
  },
  movement: {
    derived_from_style: 'Realism',
    movement: '19th-Century Realism & Romanticism',
  },
  artist: {
    top3: [
      { artist: 'Gustave Courbet', confidence: 0.7541 },
      { artist: 'Jean-François Millet', confidence: 0.1428 },
      { artist: 'Honoré Daumier', confidence: 0.0612 },
    ],
    prediction: 'Gustave Courbet',
    confidence: 0.7541,
    uncertain: false,
  },
  genre: null, // Null genre
  mood: {
    prediction: 'Sad / Melancholic',
    confidence: 0.6721,
    distribution: {
      'Sad / Melancholic': 0.6721,
      'Tense / Dark': 0.1845,
      'Calm / Content': 0.0912,
      'Awe / Sublime': 0.0381,
      'Joyful / Excited': 0.0141,
    },
    uncertain: false,
  },
  authenticity: {
    is_ai_generated: false,
    label: 'human_made',
    probability_ai: 0.0349,
    confidence: 0.9651,
    uncertain: false,
  },
  era: {
    period: 'Mid-19th Century Realism',
    timeline: 'Mid-19th Century (c. 1840–1880)',
    derived_from: 'Realism',
  },
  technique: {
    medium: 'Oil on canvas',
    method: 'Direct tonal painting (alla prima), earthy mineral pigments, textural truth',
    derived_from: 'Realism',
  },
  color: null, // Color extraction returned null
  color_error: 'CIELAB clustering failed: image dimensions too small or corrupted pixel buffer.',
  explanation: null,
  similar: null,
};

// 5. High Uncertainty across all prediction heads
export const highUncertaintyAllFixture: PredictResponse = {
  schema_version: '1.0.0',
  style: {
    top3: [
      { style: 'Abstract_Expressionism', confidence: 0.3152 },
      { style: 'Expressionism', confidence: 0.2984 },
      { style: 'Cubism', confidence: 0.2641 },
    ],
    prediction: 'Abstract_Expressionism',
    confidence: 0.3152,
    uncertain: true, // Margin < 0.15 and confidence < 0.40
  },
  movement: {
    derived_from_style: 'Abstract_Expressionism',
    movement: 'Post-War & Contemporary Art',
  },
  artist: {
    top3: [
      { artist: 'Wassily Kandinsky', confidence: 0.2841 },
      { artist: 'Paul Klee', confidence: 0.2619 },
      { artist: 'Joan Miró', confidence: 0.2312 },
    ],
    prediction: 'Wassily Kandinsky',
    confidence: 0.2841,
    uncertain: true,
  },
  genre: null,
  mood: {
    prediction: 'Awe / Sublime',
    confidence: 0.2541,
    distribution: {
      'Awe / Sublime': 0.2541,
      'Joyful / Excited': 0.2214,
      'Calm / Content': 0.2015,
      'Tense / Dark': 0.1812,
      'Sad / Melancholic': 0.1418,
    },
    uncertain: true, // Near uniform distribution
  },
  authenticity: {
    is_ai_generated: true,
    label: 'ai_generated',
    probability_ai: 0.5124, // Near 50/50 decision boundary
    confidence: 0.5124,
    uncertain: true,
  },
  era: {
    period: 'New York School / Post-WWII',
    timeline: 'c. 1940s–1960s',
    derived_from: 'Abstract_Expressionism',
  },
  technique: {
    medium: 'Enamel, house paint, oil on raw cotton duck',
    method: 'All-over composition, unstretched canvas handling, dynamic physical gesture',
    derived_from: 'Abstract_Expressionism',
  },
  color: {
    palette: [
      { hex: '#111827', rgb: [17, 24, 39], percent: 0.40 },
      { hex: '#374151', rgb: [55, 65, 81], percent: 0.30 },
      { hex: '#6b7280', rgb: [107, 114, 128], percent: 0.20 },
      { hex: '#9ca3af', rgb: [156, 163, 175], percent: 0.10 },
    ],
    color_count: 4,
    temperature: 'monochrome',
    warm_score: 0.05,
    cool_score: 0.05,
    brightness: 'dark',
    saturation: 'muted',
    is_pastel: false,
  },
  explanation: null,
  similar: null,
};

export const mockFixtures: Record<string, PredictResponse> = {
  confidentBaroque: confidentBaroqueFixture,
  uncertainArtist: uncertainArtistFixture,
  aiGenerated: aiGeneratedFixture,
  genreNullColorError: genreNullColorErrorFixture,
  highUncertaintyAll: highUncertaintyAllFixture,
};
