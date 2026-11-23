import React from 'react';
import { ArrowUpRight } from 'lucide-react';
import {
  ProjectDiscipline,
  STUDIO_PROJECTS,
  StudioProject,
  filterProjectsByDiscipline,
} from '../data/studioData';
import { useTactileGroup } from '../physics/useTactileSurface';

interface SelectedWorksSectionProps {
  activeDiscipline: ProjectDiscipline;
  onSelectDiscipline: (discipline: ProjectDiscipline) => void;
  onInspectProject: (project: StudioProject) => void;
}

const DISCIPLINES: ProjectDiscipline[] = [
  'All',
  'Web Design',
  'Art Direction',
  'WebGL & 3D',
];

const ProjectArchitecturalMotif: React.FC<{ motif: StudioProject['motif'] }> = ({
  motif,
}) => {
  return (
    <svg
      viewBox="0 0 320 140"
      aria-hidden="true"
      className="h-28 w-full border-b border-white/[0.08] bg-[#141417] text-[#E4E4E7]"
    >
      {motif === 'architecture' && (
        <g fill="none" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1">
          <path d="M24 118 H296" strokeOpacity="0.2" />
          <rect x="44" y="54" width="38" height="64" />
          <rect x="92" y="28" width="44" height="90" strokeOpacity="0.7" />
          <path d="M92 28 L114 14 L136 28" strokeOpacity="0.8" />
          <rect x="148" y="44" width="42" height="74" />
          <path d="M204 118 Q224 70 244 118 M234 118 Q254 70 274 118" strokeOpacity="0.65" />
        </g>
      )}
      {motif === 'acoustic' && (
        <g fill="none" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1">
          <circle cx="160" cy="70" r="46" strokeOpacity="0.65" />
          <circle cx="160" cy="70" r="30" strokeOpacity="0.35" />
          <circle cx="160" cy="70" r="14" strokeOpacity="0.8" />
          <line x1="36" y1="70" x2="284" y2="70" strokeOpacity="0.18" />
        </g>
      )}
      {motif === 'optics' && (
        <g fill="none" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1">
          <path d="M20 94 Q90 66 160 88 T300 76" strokeOpacity="0.6" />
          <path d="M20 108 Q90 80 160 102 T300 90" strokeOpacity="0.3" />
          <polygon points="154,68 154,28 176,68" strokeOpacity="0.85" />
          <polygon points="214,76 214,44 230,76" strokeOpacity="0.65" />
        </g>
      )}
      {motif === 'typography' && (
        <g fill="none" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1">
          <rect x="42" y="24" width="72" height="92" strokeOpacity="0.6" />
          <rect x="124" y="24" width="72" height="92" strokeOpacity="0.35" />
          <rect x="206" y="24" width="72" height="92" strokeOpacity="0.6" />
          <line x1="54" y1="44" x2="100" y2="44" strokeOpacity="0.8" />
          <line x1="136" y1="68" x2="184" y2="68" strokeOpacity="0.55" />
          <line x1="218" y1="54" x2="264" y2="54" strokeOpacity="0.8" />
        </g>
      )}
    </svg>
  );
};

export const SelectedWorksSection: React.FC<SelectedWorksSectionProps> = ({
  activeDiscipline,
  onSelectDiscipline,
  onInspectProject,
}) => {
  const visibleProjects = filterProjectsByDiscipline(
    STUDIO_PROJECTS,
    activeDiscipline,
  );
  const getToggleProps = useTactileGroup<HTMLButtonElement>('toggle');
  const getCardProps = useTactileGroup<HTMLElement>('card');
  const getButtonProps = useTactileGroup<HTMLButtonElement>('button');

  return (
    <section
      id="selected-works"
      className="relative z-20 border-t border-white/[0.08] bg-[#101012] py-20 lg:py-28"
    >
      <div className="mx-auto w-full max-w-[1440px] px-6 sm:px-10 lg:px-12">
        <div className="flex flex-col justify-between gap-8 border-b border-white/[0.08] pb-10 lg:flex-row lg:items-end">
          <div className="max-w-xl">
            <p className="text-[12px] tracking-[0.06em] text-[#8E8E93] uppercase">
              Selected Commissions
            </p>
            <h2
              className="mt-3 text-[clamp(1.75rem,2.6vw,2.5rem)] font-light leading-[1.15] tracking-[-0.02em] text-[#F4F4F5]"
              style={{ textWrap: 'balance' }}
            >
              Digital architecture, spatial web systems, and tactile monochrome direction.
            </h2>
          </div>

          <div
            role="group"
            aria-label="Filter projects by discipline"
            className="flex flex-wrap items-center gap-1.5 rounded-lg border border-white/[0.08] bg-[#151518] p-1"
          >
            {DISCIPLINES.map((discipline) => {
              const isActive = activeDiscipline === discipline;
              return (
                <button
                  key={discipline}
                  {...getToggleProps(discipline, isActive)}
                  type="button"
                  onClick={() => onSelectDiscipline(discipline)}
                  aria-pressed={isActive}
                  className={`hw-control whitespace-nowrap rounded-md px-3.5 py-2 text-[12px] font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
                    isActive
                      ? 'bg-[#F4F4F5] text-[#111113]'
                      : 'text-[#A1A1AA] hover:text-[#F4F4F5]'
                  }`}
                >
                  {discipline}
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-2">
          {visibleProjects.map((project) => (
            <article
              key={project.id}
              {...getCardProps(project.id, false)}
              className="hw-card group flex flex-col justify-between border border-white/[0.08] bg-[#131316] hover:border-white/25"
            >
              <div>
                <ProjectArchitecturalMotif motif={project.motif} />

                <div className="p-6 sm:p-7">
                  <div className="flex items-center justify-between gap-4 text-[12px] tabular-nums text-[#8E8E93]">
                    <span>
                      {project.index}. {project.discipline}
                    </span>
                    <span>{project.year}</span>
                  </div>

                  <h3 className="mt-3 text-xl font-normal tracking-[-0.015em] text-[#F4F4F5]">
                    {project.title}
                  </h3>

                  <p className="mt-2.5 text-[14px] leading-relaxed text-[#A1A1AA]">
                    {project.summary}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between gap-4 border-t border-white/[0.06] px-6 py-4 sm:px-7">
                <span className="truncate text-[12px] text-[#71717A]">
                  {project.deliverables.join(' · ')}
                </span>
                <button
                  {...getButtonProps(`inspect-${project.id}`, false)}
                  type="button"
                  onClick={() => onInspectProject(project)}
                  aria-label={`Inspect ${project.title} case study`}
                  className="hw-text-trigger inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap text-[12px] font-medium text-[#F4F4F5] hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white"
                >
                  <span>Case Study</span>
                  <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
                </button>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
};
