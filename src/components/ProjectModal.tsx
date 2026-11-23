import React, { useEffect } from 'react';
import { X } from 'lucide-react';
import { StudioProject } from '../data/studioData';
import { useTactileSurface } from '../physics/useTactileSurface';

interface ProjectModalProps {
  project: StudioProject | null;
  onClose: () => void;
  onInquire: () => void;
}

export const ProjectModal: React.FC<ProjectModalProps> = ({
  project,
  onClose,
  onInquire,
}) => {
  const panelSurface = useTactileSurface<HTMLDivElement>({
    variant: 'panel',
    active: Boolean(project),
  });
  const closeIconBtn = useTactileSurface<HTMLButtonElement>({ variant: 'button' });
  const closeBtn = useTactileSurface<HTMLButtonElement>({ variant: 'button' });
  const inquireBtn = useTactileSurface<HTMLButtonElement>({
    variant: 'button',
    active: true,
  });

  useEffect(() => {
    if (!project) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [project, onClose]);

  if (!project) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="project-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 sm:p-6"
    >
      <div
        ref={panelSurface.ref}
        {...panelSurface.tactileProps}
        className="hw-panel w-full max-w-xl border border-white/15 bg-[#141417] p-6 sm:p-8"
      >
        <div className="flex items-start justify-between gap-4 border-b border-white/[0.08] pb-5">
          <div>
            <p className="text-[12px] tabular-nums text-[#8E8E93]">
              {project.index}. {project.discipline} · {project.year}
            </p>
            <h2
              id="project-modal-title"
              className="mt-2 text-2xl font-light tracking-[-0.02em] text-[#F4F4F5]"
            >
              {project.title}
            </h2>
          </div>
          <button
            ref={closeIconBtn.ref}
            {...closeIconBtn.tactileProps}
            type="button"
            onClick={onClose}
            aria-label="Close case study"
            className="hw-control flex h-10 w-10 items-center justify-center rounded-full border border-white/15 text-[#A1A1AA] hover:border-white/50 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="mt-6 space-y-5 text-[14px] leading-relaxed text-[#A1A1AA]">
          <p>{project.summary}</p>

          <div className="grid grid-cols-1 gap-4 border-y border-white/[0.08] py-4 sm:grid-cols-2">
            <div>
              <p className="text-[11px] tracking-[0.06em] text-[#71717A] uppercase">
                Client
              </p>
              <p className="mt-1 text-[#F4F4F5]">{project.client}</p>
            </div>
            <div>
              <p className="text-[11px] tracking-[0.06em] text-[#71717A] uppercase">
                Deliverables
              </p>
              <p className="mt-1 text-[#F4F4F5]">
                {project.deliverables.join(' · ')}
              </p>
            </div>
          </div>

          <div>
            <p className="text-[11px] tracking-[0.06em] text-[#71717A] uppercase">
              Outcome
            </p>
            <p className="mt-1 text-[#E4E4E7]">{project.outcome}</p>
          </div>
        </div>

        <div className="mt-7 flex items-center justify-end gap-3 border-t border-white/[0.08] pt-5">
          <button
            ref={closeBtn.ref}
            {...closeBtn.tactileProps}
            type="button"
            onClick={onClose}
            className="hw-control rounded-full border border-white/20 px-5 py-2.5 text-[12px] font-medium text-[#D4D4D8] hover:border-white/50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            Close
          </button>
          <button
            ref={inquireBtn.ref}
            {...inquireBtn.tactileProps}
            type="button"
            onClick={() => {
              onClose();
              onInquire();
            }}
            className="hw-control rounded-full bg-[#F4F4F5] px-6 py-2.5 text-[12px] font-medium tracking-[0.06em] text-[#111113] uppercase hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            Request Similar Scope
          </button>
        </div>
      </div>
    </div>
  );
};
