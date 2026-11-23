export interface SpringConfig {
  mass: number;
  stiffness: number;
  damping: number;
  minClamp?: number;
  maxClamp?: number;
  restitution?: number;
}

export interface SpringState {
  value: number;
  velocity: number;
  target: number;
}

export const SPRING_PRESETS = {
  /** Snappy anodized keycap / tactile switch with crisp detent overshoot & hard stop */
  tactileButton: {
    mass: 0.72,
    stiffness: 460,
    damping: 22,
    minClamp: -0.10,
    maxClamp: 1.0,
    restitution: 0.14,
  } satisfies SpringConfig,

  /** Reduced-motion critically damped tactile switch (zero overshoot/oscillation) */
  tactileButtonReduced: {
    mass: 0.72,
    stiffness: 520,
    damping: 40,
    minClamp: 0.0,
    maxClamp: 1.0,
    restitution: 0.0,
  } satisfies SpringConfig,

  /** Heavier machined chassis / floating panel / diorama suspension */
  chassisPanel: {
    mass: 1.35,
    stiffness: 260,
    damping: 20,
    minClamp: -0.18,
    maxClamp: 1.15,
    restitution: 0.12,
  } satisfies SpringConfig,

  /** Smooth specular tilt & optical parallax gimbal */
  specularParallax: {
    mass: 1.1,
    stiffness: 185,
    damping: 21,
    minClamp: -1.25,
    maxClamp: 1.25,
    restitution: 0.0,
  } satisfies SpringConfig,
} as const;

export function createSpringState(initial = 0): SpringState {
  return {
    value: initial,
    velocity: 0,
    target: initial,
  };
}

/**
 * 4th-order Runge-Kutta (RK4) damped harmonic spring solver with firm
 * mechanical stops/detents (minClamp / maxClamp + controlled restitution).
 * Accurately integrates stiff tactile switches without high-frequency divergence.
 */
export function stepSpringRK4(
  state: SpringState,
  dtSeconds: number,
  config: SpringConfig,
): SpringState {
  const dt = Math.min(Math.max(dtSeconds, 0.0001), 0.032);
  const { mass, stiffness, damping, minClamp, maxClamp, restitution = 0.12 } = config;
  const invMass = 1 / Math.max(0.05, mass);
  const target = state.target;

  const accel = (x: number, v: number): number =>
    (stiffness * (target - x) - damping * v) * invMass;

  const x0 = state.value;
  const v0 = state.velocity;

  const k1x = v0;
  const k1v = accel(x0, v0);

  const halfDt = dt * 0.5;
  const x1 = x0 + k1x * halfDt;
  const v1 = v0 + k1v * halfDt;
  const k2x = v1;
  const k2v = accel(x1, v1);

  const x2 = x0 + k2x * halfDt;
  const v2 = v0 + k2v * halfDt;
  const k3x = v2;
  const k3v = accel(x2, v2);

  const x3 = x0 + k3x * dt;
  const v3 = v0 + k3v * dt;
  const k4x = v3;
  const k4v = accel(x3, v3);

  let nextValue = x0 + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
  let nextVelocity = v0 + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);

  // Firm mechanical bottom-out / rebound stops (detents)
  if (maxClamp !== undefined && nextValue > maxClamp) {
    nextValue = maxClamp;
    if (nextVelocity > 0) {
      nextVelocity = target >= maxClamp ? -nextVelocity * restitution : 0;
    }
  }
  if (minClamp !== undefined && nextValue < minClamp) {
    nextValue = minClamp;
    if (nextVelocity < 0) {
      nextVelocity = target <= minClamp ? -nextVelocity * restitution : 0;
    }
  }

  state.value = nextValue;
  state.velocity = nextVelocity;
  return state;
}

export function isSpringSettled(
  state: SpringState,
  epsilon = 0.0008,
): boolean {
  return (
    Math.abs(state.target - state.value) < epsilon &&
    Math.abs(state.velocity) < epsilon * 4
  );
}

/**
 * Shared real-time hardware interaction telemetry read directly by WebGL2 shaders
 * and CSS light-coupling without triggering React re-renders.
 */
export interface GlobalHardwareTelemetry {
  activeUiPressDepth: number;
  uiTiltX: number;
  uiTiltY: number;
  pointerNx: number;
  pointerNy: number;
  pointerVelX: number;
  pointerVelY: number;
  deviceTiltX: number;
  deviceTiltY: number;
}

export const globalHardwareTelemetry: GlobalHardwareTelemetry = {
  activeUiPressDepth: 0,
  uiTiltX: 0,
  uiTiltY: 0,
  pointerNx: 0,
  pointerNy: 0,
  pointerVelX: 0,
  pointerVelY: 0,
  deviceTiltX: 0,
  deviceTiltY: 0,
};
