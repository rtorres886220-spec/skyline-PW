import React, { useEffect, useRef, useState } from 'react';
import { LightingPresetId } from '../data/skylineLayout';
import {
  SkylineController,
  createSanDiegoSkylineScene,
} from '../three/SanDiegoSkylineScene';

interface SkylineCanvasProps {
  lightingPreset: LightingPresetId;
  resetSignal: number;
}

export const SkylineCanvas: React.FC<SkylineCanvasProps> = ({
  lightingPreset,
  resetSignal,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<SkylineController | null>(null);
  const [ready, setReady] = useState(false);
  const [contextLost, setContextLost] = useState(false);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const controller = createSanDiegoSkylineScene(container, {
      onReady: () => setReady(true),
      onContextLost: () => setContextLost(true),
      onContextRestored: () => setContextLost(false),
    });

    controllerRef.current = controller;

    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, []);

  useEffect(() => {
    controllerRef.current?.setLightingPreset(lightingPreset);
  }, [lightingPreset]);

  useEffect(() => {
    if (resetSignal > 0) {
      controllerRef.current?.resetCamera();
    }
  }, [resetSignal]);

  return (
    <div
      className="relative h-full w-full select-none"
      data-testid="skyline-canvas-wrapper"
      data-webgl-ready={ready ? 'true' : 'false'}
      data-lighting-preset={lightingPreset}
    >
      {/* Isolated WebGL mount container with seamless alpha compositing */}
      <div
        ref={mountRef}
        data-testid="skyline-canvas"
        aria-label="Interactive 3D monochrome diorama of the San Diego Bay skyline and Convention Center"
        role="img"
        className="h-full w-full cursor-grab active:cursor-grabbing"
      />

      {!ready && !contextLost && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
        >
          <div className="h-[64%] w-[64%] rounded-full border border-white/10 bg-radial from-white/[0.04] via-transparent to-transparent" />
        </div>
      )}

      {contextLost && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#111113]/95 p-6 text-center">
          <div className="max-w-xs space-y-2 border border-white/15 bg-[#161619] p-5">
            <p className="text-sm font-medium text-[#F4F4F5]">
              3D Viewport Paused
            </p>
            <p className="text-xs leading-relaxed text-[#A1A1AA]">
              WebGL2 context was suspended by the browser. Reload or restore GPU resources to resume the San Diego skyline diorama.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
