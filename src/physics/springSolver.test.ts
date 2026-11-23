import assert from 'node:assert/strict';
import test from 'node:test';
import {
  SPRING_PRESETS,
  createSpringState,
  isSpringSettled,
  stepSpringRK4,
} from './springSolver.ts';

test('RK4 spring solver reaches target press detent with firm bottom-out clamp', () => {
  const state = createSpringState(0);
  state.target = 1.0;

  let maxObserved = 0;
  for (let i = 0; i < 60; i++) {
    stepSpringRK4(state, 1 / 60, SPRING_PRESETS.tactileButton);
    if (state.value > maxObserved) maxObserved = state.value;
    assert.ok(
      state.value <= (SPRING_PRESETS.tactileButton.maxClamp ?? 1.0) + 1e-9,
      `spring exceeded maxClamp bottom-out stop: ${state.value}`,
    );
  }

  assert.ok(Math.abs(state.value - 1.0) < 0.02, `expected ~1.0, got ${state.value}`);
  assert.ok(maxObserved <= 1.0 + 1e-9);
});

test('RK4 underdamped spring exhibits realistic mechanical rebound overshoot on release', () => {
  const state = createSpringState(1.0);
  state.target = 0.0;

  let minObserved = 1.0;
  for (let i = 0; i < 60; i++) {
    stepSpringRK4(state, 1 / 60, SPRING_PRESETS.tactileButton);
    if (state.value < minObserved) minObserved = state.value;
  }

  // Should rebound slightly below 0 (mechanical overshoot) while respecting minClamp
  assert.ok(minObserved < 0, `expected slight mechanical overshoot < 0, got ${minObserved}`);
  assert.ok(
    minObserved >= (SPRING_PRESETS.tactileButton.minClamp ?? -0.1) - 1e-9,
    `overshoot penetrated rebound stop: ${minObserved}`,
  );
  assert.ok(isSpringSettled(state, 0.01));
});

test('reduced-motion spring preset settles without overshoot', () => {
  const state = createSpringState(1.0);
  state.target = 0.0;

  for (let i = 0; i < 60; i++) {
    stepSpringRK4(state, 1 / 60, SPRING_PRESETS.tactileButtonReduced);
    assert.ok(state.value >= 0, `reduced-motion spring went negative: ${state.value}`);
  }
  assert.ok(isSpringSettled(state, 0.005));
});
