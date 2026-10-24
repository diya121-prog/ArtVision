/**
 * Pure SVG Procedural Image Generator
 * Generates self-contained SVG Data URIs for mock artwork, heatmaps, and overlays.
 * No external network calls, no copyrighted imagery.
 */

export function createMockArtworkSvg(title: string, style: string, colors: string[]): string {
  const [c1 = '#1e1b4b', c2 = '#7c3aed', c3 = '#f59e0b', c4 = '#10b981'] = colors;
  
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
    <defs>
      <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c1}" />
        <stop offset="50%" stop-color="${c2}" />
        <stop offset="100%" stop-color="${c3}" />
      </linearGradient>
      <radialGradient id="glow" cx="45%" cy="40%" r="55%">
        <stop offset="0%" stop-color="${c4}" stop-opacity="0.7" />
        <stop offset="100%" stop-color="${c1}" stop-opacity="0" />
      </radialGradient>
      <filter id="noise">
        <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="3" result="noise" />
        <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.15 0"/>
        <feBlend in="SourceGraphic" mode="multiply" />
      </filter>
    </defs>
    <rect width="600" height="450" fill="url(#bg)" />
    <circle cx="280" cy="220" r="160" fill="url(#glow)" />
    
    <!-- Abstract compositional shapes -->
    <path d="M 80,380 Q 240,160 420,320 T 560,200" fill="none" stroke="${c3}" stroke-width="14" opacity="0.65" stroke-linecap="round"/>
    <polygon points="180,90 320,130 250,270" fill="${c4}" opacity="0.35" />
    <circle cx="430" cy="140" r="55" fill="${c3}" opacity="0.5" />
    <circle cx="160" cy="290" r="40" fill="${c2}" opacity="0.4" />
    <rect x="0" y="0" width="600" height="450" filter="url(#noise)" opacity="0.8" />
    
    <!-- Art label overlay badge -->
    <g transform="translate(20, 390)">
      <rect width="260" height="42" rx="8" fill="#0f172a" opacity="0.75" />
      <text x="16" y="26" font-family="system-ui, sans-serif" font-size="14" font-weight="600" fill="#f8fafc">
        ${escapeXml(title)} · ${escapeXml(style)}
      </text>
    </g>
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function createMockHeatmapSvg(head: string, primaryColor: string = '#ef4444'): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
    <defs>
      <radialGradient id="hotspot1" cx="42%" cy="46%" r="35%">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="1" />
        <stop offset="30%" stop-color="#ef4444" stop-opacity="0.9" />
        <stop offset="65%" stop-color="#f59e0b" stop-opacity="0.6" />
        <stop offset="85%" stop-color="#3b82f6" stop-opacity="0.3" />
        <stop offset="100%" stop-color="#1e1b4b" stop-opacity="0" />
      </radialGradient>
      <radialGradient id="hotspot2" cx="70%" cy="30%" r="28%">
        <stop offset="0%" stop-color="#fef08a" stop-opacity="0.9" />
        <stop offset="40%" stop-color="#f97316" stop-opacity="0.7" />
        <stop offset="80%" stop-color="#ef4444" stop-opacity="0.3" />
        <stop offset="100%" stop-color="#000000" stop-opacity="0" />
      </radialGradient>
    </defs>
    <rect width="600" height="450" fill="#0f172a" />
    <rect width="600" height="450" fill="url(#hotspot1)" />
    <rect width="600" height="450" fill="url(#hotspot2)" />
    <g transform="translate(20, 28)">
      <rect width="180" height="30" rx="6" fill="#000000" opacity="0.75" />
      <text x="12" y="20" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="${primaryColor}">
        Grad-CAM: ${escapeXml(head.toUpperCase())}
      </text>
    </g>
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function createMockOverlaySvg(artworkSvg: string, heatmapSvg: string): string {
  // Composite by embedding artwork and heatmap in an SVG group with mix-blend-mode
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
    <image href="${artworkSvg}" width="600" height="450" />
    <image href="${heatmapSvg}" width="600" height="450" opacity="0.6" style="mix-blend-mode: screen;" />
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function escapeXml(unsafe: string): string {
  return unsafe.replace(/[<>&'"]/g, (c) => {
    switch (c) {
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '&': return '&amp;';
      case '\'': return '&apos;';
      case '"': return '&quot;';
      default: return c;
    }
  });
}
