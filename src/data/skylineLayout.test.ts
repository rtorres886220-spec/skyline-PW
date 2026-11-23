import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BASIN_RADIUS,
  HERO_LANDMARKS,
  HARBOR_SAILBOATS,
  generateCityGridBlocks,
  generateSeawallLights,
  isInsideBasin,
  isOnBayWater,
  isOnLand,
  seawallZ,
} from './skylineLayout';
import {
  STUDIO_PROJECTS,
  filterProjectsByDiscipline,
  validateContactInquiry,
} from './studioData';

test('all hero landmarks remain inside the circular diorama basin and on land', () => {
  assert.ok(HERO_LANDMARKS.length >= 8);
  for (const landmark of HERO_LANDMARKS) {
    assert.equal(
      isInsideBasin(landmark.x, landmark.z, 0.5),
      true,
      `${landmark.id} must fit inside the diorama basin`,
    );
    assert.equal(
      isOnLand(landmark.x, landmark.z, 0.0),
      true,
      `${landmark.id} must sit behind the Embarcadero seawall`,
    );
  }
});

test('generated city blocks are deterministic, grounded on land, and avoid hero footprints', () => {
  const blocksA = generateCityGridBlocks(2026);
  const blocksB = generateCityGridBlocks(2026);
  assert.equal(blocksA.length, blocksB.length);
  assert.ok(blocksA.length >= 30);

  for (let i = 0; i < blocksA.length; i++) {
    const b = blocksA[i];
    assert.deepEqual(b, blocksB[i]);
    assert.equal(isOnLand(b.x, b.z, 0.8), true, `${b.id} must stay on land`);
    for (const hero of HERO_LANDMARKS) {
      const dist = Math.hypot(hero.x - b.x, hero.z - b.z);
      assert.ok(dist >= 1.5, `${b.id} overlaps hero ${hero.id}`);
    }
  }
});

test('seawall streetlights and harbor sailboats stay strictly inside the circular basin', () => {
  const lights = generateSeawallLights(30);
  assert.ok(lights.length >= 20);
  for (const light of lights) {
    assert.ok(Math.hypot(light.x, light.z) < BASIN_RADIUS);
    assert.ok(Math.abs(light.z - seawallZ(light.x)) < 0.25);
  }

  for (const boat of HARBOR_SAILBOATS) {
    assert.equal(
      isOnBayWater(boat.baseX, boat.baseZ),
      true,
      `${boat.id} must sit in bay water south of seawall`,
    );
  }
});

test('project filtering returns expected subsets', () => {
  assert.equal(filterProjectsByDiscipline(STUDIO_PROJECTS, 'All').length, STUDIO_PROJECTS.length);
  const webglProjects = filterProjectsByDiscipline(STUDIO_PROJECTS, 'WebGL & 3D');
  assert.ok(webglProjects.length >= 1);
  assert.ok(webglProjects.every((p) => p.discipline === 'WebGL & 3D'));
});

test('contact inquiry validation rejects empty/invalid inputs and accepts valid briefs', () => {
  const invalid = validateContactInquiry({
    name: 'A',
    email: 'not-an-email',
    discipline: 'Web Design',
    timeline: 'Q2 2026',
    message: 'Too short',
  });
  assert.equal(invalid.valid, false);
  assert.ok(invalid.errors.name);
  assert.ok(invalid.errors.email);
  assert.ok(invalid.errors.message);

  const valid = validateContactInquiry({
    name: 'Elena Vance',
    email: 'elena@torreycreative.org',
    discipline: 'WebGL & 3D',
    timeline: 'Q2 2026',
    message: 'We need an interactive architectural showcase for our San Diego waterfront exhibition.',
  });
  assert.equal(valid.valid, true);
  assert.deepEqual(valid.errors, {});
});
