import React from 'react';
import { RotateCcw, ArrowUp } from 'lucide-react';
import { LIGHTING_PRESETS, LightingPresetId } from '../data/skylineLayout';
import { CRAFT_SPECS } from '../data/studioData';
import { useTactileGroup, useTactileSurface } from '../physics/useTactileSurface';

interface StudioCraftSectionProps {
  lightingPreset: LightingPresetId;
  onSelectLightingPreset: (id: LightingPresetId) => void;
  onResetCamera: () => void;
  onReturnToHero: () => void;
}

const PRESET_ORDER: LightingPresetId[] = [
  'overhead-studio',
  'grazing-rim',
  'harbor-dusk',
];

export const StudioCraftSection: React.FC<StudioCraftSectionProps> = ({
  lightingPreset,
  onSelectLightingPreset,
  onResetCamera,
  onReturnToHero,
}) => {
  const getPresetProps = useTactileGroup<HTMLButtonElement>('toggle');
  const getSpecCardProps = useTactileGroup<HTMLDivElement>('card');
  const inspectKey = useTactileSurface<HTMLButtonElement>({ variant: 'button' });
  const backKey = useTactileSurface<HTMLButtonElement>({ variant: 'button' });

  return (
    <section
      id="studio-craft"
      className="relative z-20 border-t border-white/[0.08] bg-[#121215] py-20 lg:py-24"
    >
      <div className="mx-auto w-full max-w-[1440px] px-6 sm:px-10 lg:px-12">
        <div className="grid grid-cols-1 gap-10 border-b border-white/[0.08] pb-12 lg:grid-cols-12 lg:items-end">
          <div className="lg:col-span-6">
            <p className="text-[12px] tracking-[0.06em] text-[#8E8E93] uppercase">
              Skyline Rendering Architecture
            </p>
            <h2
              className="mt-3 text-[clamp(1.65rem,2.4vw,2.25rem)] font-light leading-[1.18] tracking-[-0.02em] text-[#F4F4F5]"
              style={{ textWrap: 'balance' }}
            >
              Procedural virtual softbox studio and WebGL2 GLSL3 material craft.
            </h2>
          </div>

          <div className="flex flex-col gap-3 lg:col-span-6 lg:items-end">
            <div className="flex flex-wrap items-center gap-2">
              <span className="mr-2 text-[12px] text-[#8E8E93]">
                Studio Softbox Rig:
              </span>
              {PRESET_ORDER.map((id) => {
                const preset = LIGHTING_PRESETS[id];
                const isActive = lightingPreset === id;
                return (
                  <button
                    key={id}
                    {...getPresetProps(id, isActive)}
                    type="button"
                    onClick={() => onSelectLightingPreset(id)}
                    aria-pressed={isActive}
                    className={`hw-control whitespace-nowrap rounded-full border px-4 py-2 text-[12px] font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
                      isActive
                        ? 'border-white bg-[#F4F4F5] text-[#111113]'
                        : 'border-white/20 bg-transparent text-[#D4D4D8] hover:border-white/50'
                    }`}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center gap-4 text-[12px] text-[#8E8E93]">
              <span>{LIGHTING_PRESETS[lightingPreset].subtitle}</span>
              <button
                ref={inspectKey.ref}
                {...inspectKey.tactileProps}
                type="button"
                onClick={() => {
                  onResetCamera();
                  onReturnToHero();
                }}
                className="hw-text-trigger inline-flex items-center gap-1.5 whitespace-nowrap text-[#F4F4F5] underline decoration-white/30 underline-offset-4 hover:decoration-white focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Inspect Skyline</span>
              </button>
            </div>
          </div>
        </div>

        <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-3">
          {CRAFT_SPECS.map((spec) => (
            <div
              key={spec.id}
              {...getSpecCardProps(spec.id, false)}
              className="hw-card flex flex-col justify-between border border-white/[0.08] bg-[#151518] p-6"
            >
              <div>
                <p className="text-[12px] tabular-nums text-[#71717A]">
                  {spec.index}. {spec.subtitle}
                </p>
                <h3 className="mt-3 text-lg font-normal text-[#F4F4F5]">
                  {spec.title}
                </h3>
                <p className="mt-3 text-[14px] leading-relaxed text-[#A1A1AA]">
                  {spec.description}
                </p>
              </div>
              <p className="mt-6 border-t border-white/[0.06] pt-4 text-[12px] leading-relaxed text-[#71717A]">
                {spec.technicalNote}
              </p>
            </div>
          ))}
        </div>

        <footer className="mt-16 flex flex-col items-start justify-between gap-6 border-t border-white/[0.08] pt-8 text-[12px] text-[#8E8E93] sm:flex-row sm:items-center">
          <div className="flex flex-wrap items-center gap-4 tabular-nums">
            <span className="text-[#E4E4E7]">Promptworks Studio™</span>
            <span aria-hidden="true">·</span>
            <span>San Diego, California</span>
            <span aria-hidden="true">·</span>
            <span>32.7157° N, 117.1611° W</span>
          </div>

          <button
            ref={backKey.ref}
            {...backKey.tactileProps}
            type="button"
            onClick={onReturnToHero}
            className="hw-text-trigger inline-flex items-center gap-2 whitespace-nowrap text-[#F4F4F5] hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white"
          >
            <span>Back to Skyline</span>
            <ArrowUp className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </footer>
      </div>
    </section>
  );
};
