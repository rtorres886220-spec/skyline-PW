import React from 'react';
import { useTactileSurface } from '../physics/useTactileSurface';

interface StudioHeaderProps {
  onOpenContact: () => void;
}

export const StudioHeader: React.FC<StudioHeaderProps> = ({ onOpenContact }) => {
  const talkKey = useTactileSurface<HTMLButtonElement>({ variant: 'button' });

  return (
    <header className="relative z-20 mx-auto flex w-full max-w-[1440px] items-center justify-between gap-8 px-6 py-6 sm:px-10 lg:px-12 lg:py-8">
      <a
        href="#hero"
        className="shrink-0 whitespace-nowrap text-[17px] font-normal tracking-[-0.015em] text-[#F4F4F5] hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white"
      >
        Promptworks Studio™
      </a>

      <div className="hidden items-center gap-16 md:flex lg:gap-24">
        <div className="flex flex-col text-[11px] leading-[1.45] tracking-[0.06em] uppercase">
          <span className="whitespace-nowrap font-medium text-[#E4E4E7]">
            Web Design / Art Direction
          </span>
          <span className="whitespace-nowrap tabular-nums text-[#71717A]">
            Portfolio / 2021 — 2026
          </span>
        </div>

        <div className="hidden flex-col text-[11px] leading-[1.45] tracking-[0.06em] uppercase lg:flex">
          <span className="whitespace-nowrap font-medium text-[#E4E4E7]">
            Available for Select
          </span>
          <span className="whitespace-nowrap tabular-nums text-[#71717A]">
            Projects in 2026
          </span>
        </div>
      </div>

      <button
        ref={talkKey.ref}
        {...talkKey.tactileProps}
        type="button"
        onClick={onOpenContact}
        className="hw-control shrink-0 whitespace-nowrap rounded-full border border-white/25 bg-white/[0.03] px-6 py-2.5 text-[11px] font-medium tracking-[0.08em] text-[#F4F4F5] uppercase hover:border-white/70 hover:bg-white/[0.08] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
      >
        Let&apos;s Talk
      </button>
    </header>
  );
};
