export type ProjectDiscipline = 'All' | 'Web Design' | 'Art Direction' | 'WebGL & 3D';

export interface StudioProject {
  id: string;
  index: string;
  title: string;
  client: string;
  year: string;
  discipline: Exclude<ProjectDiscipline, 'All'>;
  summary: string;
  deliverables: string[];
  outcome: string;
  motif: 'architecture' | 'optics' | 'acoustic' | 'typography';
}

export interface CraftSpec {
  id: string;
  index: string;
  title: string;
  subtitle: string;
  description: string;
  technicalNote: string;
}

export interface InquiryFormData {
  name: string;
  email: string;
  discipline: string;
  timeline: string;
  message: string;
}

export interface InquiryValidationResult {
  valid: boolean;
  errors: Partial<Record<keyof InquiryFormData, string>>;
}

export const STUDIO_PROJECTS: StudioProject[] = [
  {
    id: 'torrey-spatial',
    index: '01',
    title: 'Torrey Pines Spatial Observatory',
    client: 'Salk Coastal Research Initiative',
    year: '2026',
    discipline: 'WebGL & 3D',
    summary:
      'Interactive topographic bay telemetry and procedural shader atlas built for coastal atmospheric researchers in La Jolla.',
    deliverables: ['WebGL2 Shader Pipeline', 'Design System', 'Editorial Direction'],
    outcome: '60 fps software-resilient coastal terrain explorer across 14 research stations.',
    motif: 'architecture',
  },
  {
    id: 'monolith-acoustics',
    index: '02',
    title: 'Monolith Acoustic Instruments',
    client: 'Monolith Audio Lab',
    year: '2025',
    discipline: 'Art Direction',
    summary:
      'Brushed anodized aluminum product monograph and tactile digital showroom anchored in monochrome studio photography.',
    deliverables: ['Brand Identity', 'Anisotropic 3D Finishes', 'E-Commerce Architecture'],
    outcome: '3.2x session dwell time during flagship studio monitor launch.',
    motif: 'acoustic',
  },
  {
    id: 'coronado-maritime',
    index: '03',
    title: 'Coronado Maritime Archive',
    client: 'Pacific Naval Heritage Trust',
    year: '2025',
    discipline: 'Web Design',
    summary:
      'Editorial archive documenting century-old Pacific harbor vessels through high-contrast blueprint typography and crisp bathymetric maps.',
    deliverables: ['Interactive Archive', 'Typographic System', 'Spatial Wayfinding'],
    outcome: '48,000 archival plates indexed with instant spatial search.',
    motif: 'optics',
  },
  {
    id: 'kettner-press',
    index: '04',
    title: 'Kettner Architectural Press',
    client: 'Little Italy Monograph Collective',
    year: '2024',
    discipline: 'Web Design',
    summary:
      'Digital reading room and print catalog for Southern California brutalist and mid-century coastal architecture publications.',
    deliverables: ['Editorial Platform', 'Custom Grid Engine', 'Subscription Flow'],
    outcome: '42% lift in international print-edition reservations.',
    motif: 'typography',
  },
];

export const CRAFT_SPECS: CraftSpec[] = [
  {
    id: 'virtual-softboxes',
    index: '01',
    title: 'Procedural Studio Softboxes',
    subtitle: 'PMREM Environment & RectAreaLight Rig',
    description:
      'Rather than harsh point or flat directional lamps, the skyline sits inside a calibrated dark cyclorama lit by virtual rectangular softbox panels.',
    technicalNote:
      'Custom PMREM environment baking paired with RectAreaLight specular evaluation in GLSL3.',
  },
  {
    id: 'glsl3-sdfs',
    index: '02',
    title: 'WebGL2 GLSL3 SDFs & Anisotropy',
    subtitle: 'Glass Mullions, Harbor Water & Brushed Metal',
    description:
      'Curtain-wall mullions, harbor basin ripples, and brushed titanium fins are evaluated analytically per fragment using screen-space derivatives.',
    technicalNote:
      'fwidth()-antialiased SDF grids, Schlick Fresnel reflections, and Ward anisotropic highlights.',
  },
  {
    id: 'micro-textures',
    index: '03',
    title: 'Procedural Micro-Textures',
    subtitle: 'SVG <feTurbulence> & Offscreen Canvas Maps',
    description:
      'Every flat matte obsidian plane and metallic tower crown carries deterministic physical grain generated entirely in code with zero external image assets.',
    technicalNote:
      'DOM fractalNoise turbulence filter + offscreen Canvas brushed metal & aggregate roughness maps.',
  },
];

export function filterProjectsByDiscipline(
  projects: StudioProject[],
  discipline: ProjectDiscipline,
): StudioProject[] {
  if (discipline === 'All') {
    return projects;
  }
  return projects.filter((project) => project.discipline === discipline);
}

export function validateContactInquiry(data: InquiryFormData): InquiryValidationResult {
  const errors: Partial<Record<keyof InquiryFormData, string>> = {};
  const trimmedName = data.name.trim();
  const trimmedEmail = data.email.trim();
  const trimmedMessage = data.message.trim();

  if (trimmedName.length < 2) {
    errors.name = 'Please enter your name (at least 2 characters).';
  }

  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailPattern.test(trimmedEmail)) {
    errors.email = 'Please enter a valid email address.';
  }

  if (trimmedMessage.length < 12) {
    errors.message = 'Please share a brief overview (at least 12 characters).';
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
}
