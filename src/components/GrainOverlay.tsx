import React from 'react';

/**
 * Procedural SVG <feTurbulence> micro-texture and subtle studio cyclorama
 * surface falloff giving flat obsidian surfaces authentic physical grain
 * without external image assets.
 */
export const GrainOverlay: React.FC = () => {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-10"
    >
      <svg
        className="h-full w-full opacity-[0.042] mix-blend-screen"
        xmlns="http://www.w3.org/2000/svg"
      >
        <filter id="studio-micro-grain">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.78"
            numOctaves={4}
            seed={7}
            stitchTiles="stitch"
          />
          <feColorMatrix
            type="matrix"
            values="0.33 0.33 0.33 0 0  0.33 0.33 0.33 0 0  0.33 0.33 0.33 0 0  0 0 0 0.85 0"
          />
          <feComponentTransfer>
            <feFuncR type="gamma" amplitude="1" exponent="1.15" offset="0" />
            <feFuncG type="gamma" amplitude="1" exponent="1.15" offset="0" />
            <feFuncB type="gamma" amplitude="1" exponent="1.15" offset="0" />
          </feComponentTransfer>
        </filter>
        <rect width="100%" height="100%" filter="url(#studio-micro-grain)" />
      </svg>
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(circle at 38% 48%, transparent 48%, rgba(6, 6, 8, 0.38) 100%)',
        }}
      />
    </div>
  );
};
