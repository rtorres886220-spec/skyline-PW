import { useState } from 'react';
import { LightingPresetId } from './data/skylineLayout';
import { ProjectDiscipline, StudioProject } from './data/studioData';
import { GrainOverlay } from './components/GrainOverlay';
import { HeroSection } from './components/HeroSection';
import { SelectedWorksSection } from './components/SelectedWorksSection';
import { StudioCraftSection } from './components/StudioCraftSection';
import { ContactModal } from './components/ContactModal';
import { ProjectModal } from './components/ProjectModal';

export default function App() {
  const [lightingPreset, setLightingPreset] =
    useState<LightingPresetId>('overhead-studio');
  const [resetSignal, setResetSignal] = useState(0);
  const [activeDiscipline, setActiveDiscipline] =
    useState<ProjectDiscipline>('All');
  const [contactOpen, setContactOpen] = useState(false);
  const [inspectedProject, setInspectedProject] =
    useState<StudioProject | null>(null);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="relative min-h-screen bg-[#111113] text-[#F4F4F5]">
      <GrainOverlay />

      <main>
        <HeroSection
          lightingPreset={lightingPreset}
          resetSignal={resetSignal}
          onOpenContact={() => setContactOpen(true)}
          onScrollToWork={() => scrollToSection('selected-works')}
        />

        <SelectedWorksSection
          activeDiscipline={activeDiscipline}
          onSelectDiscipline={setActiveDiscipline}
          onInspectProject={setInspectedProject}
        />

        <StudioCraftSection
          lightingPreset={lightingPreset}
          onSelectLightingPreset={setLightingPreset}
          onResetCamera={() => setResetSignal((n) => n + 1)}
          onReturnToHero={() => scrollToSection('hero')}
        />
      </main>

      <ContactModal
        open={contactOpen}
        onClose={() => setContactOpen(false)}
      />

      <ProjectModal
        project={inspectedProject}
        onClose={() => setInspectedProject(null)}
        onInquire={() => setContactOpen(true)}
      />
    </div>
  );
}
