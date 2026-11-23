import React from 'react';
import { ArrowDown } from 'lucide-react';
import { LightingPresetId } from '../data/skylineLayout';
import { useTactileSurface } from '../physics/useTactileSurface';
import { SkylineCanvas } from './SkylineCanvas';
import { StudioHeader } from './StudioHeader';

interface HeroSectionProps {
  lightingPreset: LightingPresetId;
  resetSignal: number;
  onOpenContact: () => void;
  onScrollToWork: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  lightingPreset,
  resetSignal,
  onOpenContact,
  onScrollToWork,
}) => {
  const scrollTrigger = useTactileSurface<HTMLButtonElement>({ variant: 'button' });

  return (
    <section
      id="hero"
      className="relative flex min-h-screen w-full flex-col justify-between overflow-hidden bg-radial-[at_36%_52%] from-[#1B1B1E] via-[#131315] to-[#0E0E10]"
    >
      <StudioHeader onOpenContact={onOpenContact} />

      <div className="relative z-20 mx-auto grid w-full max-w-[1440px] flex-1 grid-cols-1 items-center gap-6 px-4 pb-4 sm:px-8 lg:grid-cols-12 lg:gap-4 lg:px-10">
        {/* Left 3D Procedural San Diego Skyline Diorama */}
        <div className="relative h-[400px] w-full sm:h-[500px] lg:col-span-7 lg:h-[670px] xl:h-[710px]">
          <SkylineCanvas
            lightingPreset={lightingPreset}
            resetSignal={resetSignal}
          />
        </div>

        {/* Right Editorial Studio Copy */}
        <div className="flex flex-col items-start justify-center px-2 sm:px-2 lg:col-span-5 lg:pl-6 xl:pl-10">
          <h1
            className="max-w-[500px] text-[clamp(2.25rem,3.6vw,3.55rem)] font-light leading-[1.09] tracking-[-0.025em] text-[#F4F4F5]"
            style={{ textWrap: 'balance' }}
          >
            Independent web design and art direction based in San Diego, CA.
          </h1>

          <p className="mt-7 max-w-[400px] text-[15px] leading-[1.6] font-light text-[#9A9AA3]">
            Promptworks Studio is an independent creative practice crafting digital experiences and solving problems through thoughtful, functional design.
          </p>

          <button
            ref={scrollTrigger.ref}
            {...scrollTrigger.tactileProps}
            type="button"
            onClick={onScrollToWork}
            className="hw-control mt-9 inline-flex items-center gap-3 whitespace-nowrap rounded-full border border-white/25 bg-white/[0.02] px-6 py-3 text-[11px] font-medium tracking-[0.09em] text-[#F4F4F5] uppercase hover:border-white/70 hover:bg-white/[0.07] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            <span>Scroll for More</span>
            <ArrowDown className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Bottom Hero Metadata Bar aligned with right column */}
      <div className="relative z-20 mx-auto grid w-full max-w-[1440px] grid-cols-1 px-6 pb-7 sm:px-10 lg:grid-cols-12 lg:gap-8 lg:px-12 lg:pb-9">
        <div className="hidden lg:col-span-7 lg:block" />
        <div className="flex items-center justify-between border-t border-white/[0.06] pt-4 text-[12px] tabular-nums tracking-[0.03em] text-[#8E8E93] lg:col-span-5 lg:border-t-0 lg:pt-0 lg:pl-4 xl:pl-8">
          <span>32.7157° N, 117.1611° W</span>
          <span>© 2026</span>
        </div>
      </div>
    </section>
  );
};
