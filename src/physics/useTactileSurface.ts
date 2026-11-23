import { useCallback, useEffect, useRef } from 'react';
import {
  SPRING_PRESETS,
  SpringConfig,
  SpringState,
  createSpringState,
  globalHardwareTelemetry,
  isSpringSettled,
  stepSpringRK4,
} from './springSolver';

export type TactileSurfaceVariant = 'button' | 'toggle' | 'card' | 'panel';

interface RegisteredSurface {
  element: HTMLElement;
  variant: TactileSurfaceVariant;
  active: boolean;
  press: SpringState;
  hover: SpringState;
  tiltX: SpringState;
  tiltY: SpringState;
  pressedByPointer: boolean;
  pressedByKey: boolean;
}

class HardwarePhysicsManager {
  private surfaces = new Set<RegisteredSurface>();
  private rafId = 0;
  private running = false;
  private lastTime = 0;
  private reducedMotion = false;
  private globalListenersAttached = false;

  private rawPointerNx = 0;
  private rawPointerNy = 0;
  private prevPointerNx = 0;
  private prevPointerNy = 0;
  private pointerXSpring = createSpringState(0);
  private pointerYSpring = createSpringState(0);
  private velXSpring = createSpringState(0);
  private velYSpring = createSpringState(0);
  private rawDeviceTiltX = 0;
  private rawDeviceTiltY = 0;
  private pointerResetTimer = 0;

  private ensureGlobalListeners() {
    if (this.globalListenersAttached || typeof window === 'undefined') return;
    this.globalListenersAttached = true;

    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    this.reducedMotion = Boolean(mq?.matches);
    mq?.addEventListener?.('change', (e) => {
      this.reducedMotion = e.matches;
      this.wake();
    });

    window.addEventListener(
      'pointermove',
      (e) => {
        const w = Math.max(window.innerWidth, 1);
        const h = Math.max(window.innerHeight, 1);
        const nx = (e.clientX / w) * 2 - 1;
        const ny = (e.clientY / h) * 2 - 1;
        const dx = nx - this.rawPointerNx;
        const dy = ny - this.rawPointerNy;
        this.rawPointerNx = nx;
        this.rawPointerNy = ny;
        this.pointerXSpring.target = this.reducedMotion ? 0 : nx;
        this.pointerYSpring.target = this.reducedMotion ? 0 : ny;
        this.velXSpring.target = this.reducedMotion
          ? 0
          : Math.max(-1, Math.min(1, dx * 28));
        this.velYSpring.target = this.reducedMotion
          ? 0
          : Math.max(-1, Math.min(1, dy * 28));

        window.clearTimeout(this.pointerResetTimer);
        this.pointerResetTimer = window.setTimeout(() => {
          this.velXSpring.target = 0;
          this.velYSpring.target = 0;
          this.wake();
        }, 70);

        this.wake();
      },
      { passive: true },
    );

    if (typeof DeviceOrientationEvent !== 'undefined') {
      window.addEventListener(
        'deviceorientation',
        (e) => {
          if (this.reducedMotion) return;
          if (e.gamma === null && e.beta === null) return;
          const gx = typeof e.gamma === 'number' ? Math.max(-35, Math.min(35, e.gamma)) / 35 : 0;
          const by = typeof e.beta === 'number' ? Math.max(-35, Math.min(35, e.beta - 35)) / 35 : 0;
          this.rawDeviceTiltX = gx;
          this.rawDeviceTiltY = by;
          globalHardwareTelemetry.deviceTiltX = gx;
          globalHardwareTelemetry.deviceTiltY = by;
          this.wake();
        },
        { passive: true },
      );
    }
  }

  register(surface: RegisteredSurface) {
    this.ensureGlobalListeners();
    this.surfaces.add(surface);
    this.applySurfaceStyles(surface);
    this.wake();
  }

  unregister(surface: RegisteredSurface) {
    this.surfaces.delete(surface);
    this.syncGlobalPressDepth();
  }

  wake() {
    if (this.running || typeof window === 'undefined') return;
    this.running = true;
    this.lastTime = performance.now();
    this.rafId = window.requestAnimationFrame(this.tick);
  }

  private syncGlobalPressDepth() {
    let maxPress = 0;
    let leadTiltX = 0;
    let leadTiltY = 0;
    for (const s of this.surfaces) {
      if (s.press.value > maxPress) {
        maxPress = s.press.value;
        leadTiltX = s.tiltX.value;
        leadTiltY = s.tiltY.value;
      }
    }
    globalHardwareTelemetry.activeUiPressDepth = Math.max(0, Math.min(1, maxPress));
    globalHardwareTelemetry.uiTiltX = leadTiltX;
    globalHardwareTelemetry.uiTiltY = leadTiltY;
  }

  private applySurfaceStyles(s: RegisteredSurface) {
    const press = s.press.value;
    const hover = s.hover.value;
    const tx = this.reducedMotion ? 0 : s.tiltX.value;
    const ty = this.reducedMotion ? 0 : s.tiltY.value;
    const vel = Math.min(
      1,
      Math.hypot(s.press.velocity, s.hover.velocity) * 0.14,
    );

    const travelDown = s.variant === 'card' ? 3.2 : s.variant === 'panel' ? 2.5 : 2.6;
    const hoverLift = s.variant === 'panel' ? 0.0 : s.variant === 'card' ? 2.6 : 1.5;
    const zOffset = hover * hoverLift - Math.max(-0.15, press) * travelDown;

    // Contact AO tightens (smaller Y offset & tighter blur) and darkens on depression
    const baseShadowY = s.variant === 'panel' ? 26 : s.variant === 'card' ? 14 : 7;
    const shadowY = Math.max(
      1,
      baseShadowY * (1 - press * 0.78) + hover * (s.variant === 'panel' ? 0 : 2.5),
    );
    const baseBlur = s.variant === 'panel' ? 48 : s.variant === 'card' ? 26 : 16;
    const shadowBlur = Math.max(
      3,
      baseBlur * (1 - press * 0.74) + hover * (s.variant === 'panel' ? 0 : 4),
    );
    const aoAlpha = Math.min(
      0.86,
      (s.variant === 'panel' ? 0.56 : 0.34) + press * 0.36 + hover * 0.08,
    );

    // Chamfer catchlight rotates dynamically with surface tilt & pointer parallax
    const parallaxShift = this.reducedMotion
      ? 0
      : globalHardwareTelemetry.pointerNx * 14;
    const chamferAngle = 135 + tx * 28 - ty * 18 + parallaxShift + press * 16;

    // Active / pressed indicators emit soft reflected bounce illumination onto surrounding bezel
    const activeGlow = s.active ? 0.18 : 0.0;
    const bounceGlow = Math.min(
      0.52,
      activeGlow + hover * 0.10 + Math.max(0, press) * 0.24 + vel * 0.08,
    );

    const tiltDegX = this.reducedMotion
      ? 0
      : (s.variant === 'panel' ? 1.8 : s.variant === 'card' ? 2.8 : 4.2) * -ty;
    const tiltDegY = this.reducedMotion
      ? 0
      : (s.variant === 'panel' ? 1.8 : s.variant === 'card' ? 2.8 : 4.2) * tx;

    const style = s.element.style;
    style.setProperty('--hw-press', press.toFixed(4));
    style.setProperty('--hw-hover', hover.toFixed(4));
    style.setProperty('--hw-tilt-x', tiltDegX.toFixed(3));
    style.setProperty('--hw-tilt-y', tiltDegY.toFixed(3));
    style.setProperty('--hw-z', `${zOffset.toFixed(3)}px`);
    style.setProperty('--hw-vel', vel.toFixed(4));
    style.setProperty('--hw-shadow-y', `${shadowY.toFixed(2)}px`);
    style.setProperty('--hw-shadow-blur', `${shadowBlur.toFixed(2)}px`);
    style.setProperty('--hw-ao-alpha', aoAlpha.toFixed(3));
    style.setProperty('--hw-chamfer-angle', `${chamferAngle.toFixed(1)}deg`);
    style.setProperty('--hw-bounce-glow', bounceGlow.toFixed(3));
  }

  private tick = (now: number) => {
    const dt = Math.min(Math.max((now - this.lastTime) / 1000, 0.001), 0.032);
    this.lastTime = now;

    // Global specular parallax springs
    stepSpringRK4(
      this.pointerXSpring,
      dt,
      SPRING_PRESETS.specularParallax,
    );
    stepSpringRK4(
      this.pointerYSpring,
      dt,
      SPRING_PRESETS.specularParallax,
    );
    stepSpringRK4(
      this.velXSpring,
      dt,
      SPRING_PRESETS.specularParallax,
    );
    stepSpringRK4(
      this.velYSpring,
      dt,
      SPRING_PRESETS.specularParallax,
    );

    // Decay pointer velocity target smoothly toward zero
    if (
      Math.abs(this.rawPointerNx - this.prevPointerNx) < 1e-4 &&
      Math.abs(this.rawPointerNy - this.prevPointerNy) < 1e-4
    ) {
      this.velXSpring.target *= 0.86;
      this.velYSpring.target *= 0.86;
    }
    this.prevPointerNx = this.rawPointerNx;
    this.prevPointerNy = this.rawPointerNy;

    globalHardwareTelemetry.pointerNx = this.reducedMotion
      ? 0
      : this.pointerXSpring.value;
    globalHardwareTelemetry.pointerNy = this.reducedMotion
      ? 0
      : this.pointerYSpring.value;
    globalHardwareTelemetry.pointerVelX = this.reducedMotion
      ? 0
      : this.velXSpring.value;
    globalHardwareTelemetry.pointerVelY = this.reducedMotion
      ? 0
      : this.velYSpring.value;

    let anyMoving =
      !isSpringSettled(this.pointerXSpring) ||
      !isSpringSettled(this.pointerYSpring) ||
      !isSpringSettled(this.velXSpring) ||
      !isSpringSettled(this.velYSpring);

    const buttonCfg: SpringConfig = this.reducedMotion
      ? SPRING_PRESETS.tactileButtonReduced
      : SPRING_PRESETS.tactileButton;

    for (const s of this.surfaces) {
      const pressCfg =
        s.variant === 'card' || s.variant === 'panel'
          ? this.reducedMotion
            ? SPRING_PRESETS.tactileButtonReduced
            : SPRING_PRESETS.chassisPanel
          : buttonCfg;

      const hoverCfg = this.reducedMotion
        ? SPRING_PRESETS.tactileButtonReduced
        : SPRING_PRESETS.specularParallax;

      stepSpringRK4(s.press, dt, pressCfg);
      stepSpringRK4(s.hover, dt, hoverCfg);
      stepSpringRK4(s.tiltX, dt, SPRING_PRESETS.specularParallax);
      stepSpringRK4(s.tiltY, dt, SPRING_PRESETS.specularParallax);

      this.applySurfaceStyles(s);

      if (
        !isSpringSettled(s.press) ||
        !isSpringSettled(s.hover) ||
        !isSpringSettled(s.tiltX) ||
        !isSpringSettled(s.tiltY)
      ) {
        anyMoving = true;
      }
    }

    this.syncGlobalPressDepth();

    // Update specular CSS angle on root so even non-hovered panels feel ambient parallax
    if (typeof document !== 'undefined') {
      const globalAngle = this.reducedMotion
        ? 135
        : 135 +
          (globalHardwareTelemetry.pointerNx + this.rawDeviceTiltX) * 16 +
          globalHardwareTelemetry.pointerVelX * 10;
      document.documentElement.style.setProperty(
        '--hw-global-chamfer',
        `${globalAngle.toFixed(1)}deg`,
      );
    }

    if (anyMoving) {
      this.rafId = window.requestAnimationFrame(this.tick);
    } else {
      this.running = false;
      this.rafId = 0;
    }
  };
}

const hardwareManager = new HardwarePhysicsManager();

export interface UseTactileSurfaceOptions {
  variant?: TactileSurfaceVariant;
  active?: boolean;
}

export function useTactileSurface<T extends HTMLElement = HTMLButtonElement>(
  options: UseTactileSurfaceOptions = {},
) {
  const { variant = 'button', active = false } = options;
  const surfaceRef = useRef<RegisteredSurface | null>(null);
  const elRef = useRef<T | null>(null);

  const attachRef = useCallback(
    (node: T | null) => {
      if (surfaceRef.current && elRef.current !== node) {
        hardwareManager.unregister(surfaceRef.current);
        surfaceRef.current = null;
      }
      elRef.current = node;
      if (node && !surfaceRef.current) {
        const reg: RegisteredSurface = {
          element: node,
          variant,
          active,
          press: createSpringState(0),
          hover: createSpringState(0),
          tiltX: createSpringState(0),
          tiltY: createSpringState(0),
          pressedByPointer: false,
          pressedByKey: false,
        };
        surfaceRef.current = reg;
        hardwareManager.register(reg);
      }
    },
    [variant],
  );

  useEffect(() => {
    if (surfaceRef.current) {
      surfaceRef.current.active = active;
      hardwareManager.wake();
    }
  }, [active]);

  useEffect(() => {
    return () => {
      if (surfaceRef.current) {
        hardwareManager.unregister(surfaceRef.current);
        surfaceRef.current = null;
      }
    };
  }, []);

  const syncPressTarget = (s: RegisteredSurface) => {
    s.press.target = s.pressedByPointer || s.pressedByKey ? 1.0 : 0.0;
    hardwareManager.wake();
  };

  const onPointerEnter = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.hover.target = 1.0;
    hardwareManager.wake();
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent<T>) => {
      const s = surfaceRef.current;
      if (!s) return;
      const rect = s.element.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        const nx = Math.max(-1, Math.min(1, ((e.clientX - rect.left) / rect.width) * 2 - 1));
        const ny = Math.max(-1, Math.min(1, ((e.clientY - rect.top) / rect.height) * 2 - 1));
        s.tiltX.target = nx;
        s.tiltY.target = ny;
        hardwareManager.wake();
      }
    },
    [],
  );

  const onPointerLeave = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.hover.target = 0.0;
    s.tiltX.target = 0.0;
    s.tiltY.target = 0.0;
    s.pressedByPointer = false;
    syncPressTarget(s);
  }, []);

  const onPointerDown = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.pressedByPointer = true;
    syncPressTarget(s);
  }, []);

  const onPointerUp = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.pressedByPointer = false;
    syncPressTarget(s);
  }, []);

  const onPointerCancel = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.pressedByPointer = false;
    syncPressTarget(s);
  }, []);

  const onKeyDown = useCallback((e: React.KeyboardEvent<T>) => {
    if (e.key !== ' ' && e.key !== 'Enter') return;
    const s = surfaceRef.current;
    if (!s) return;
    s.pressedByKey = true;
    syncPressTarget(s);
  }, []);

  const onKeyUp = useCallback((e: React.KeyboardEvent<T>) => {
    if (e.key !== ' ' && e.key !== 'Enter') return;
    const s = surfaceRef.current;
    if (!s) return;
    s.pressedByKey = false;
    syncPressTarget(s);
  }, []);

  const onFocus = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.hover.target = 1.0;
    hardwareManager.wake();
  }, []);

  const onBlur = useCallback(() => {
    const s = surfaceRef.current;
    if (!s) return;
    s.hover.target = 0.0;
    s.pressedByKey = false;
    s.pressedByPointer = false;
    syncPressTarget(s);
  }, []);

  return {
    ref: attachRef,
    tactileProps: {
      onPointerEnter,
      onPointerMove,
      onPointerLeave,
      onPointerDown,
      onPointerUp,
      onPointerCancel,
      onKeyDown,
      onKeyUp,
      onFocus,
      onBlur,
    },
  };
}

/**
 * Attaches RK4 damped-spring physics directly onto mapped controls/cards by key
 * without wrapping or altering component DOM structure.
 */
export function useTactileGroup<T extends HTMLElement = HTMLElement>(
  variant: TactileSurfaceVariant = 'button',
) {
  const mapRef = useRef<Map<string, RegisteredSurface>>(new Map());

  useEffect(() => {
    const map = mapRef.current;
    return () => {
      for (const s of map.values()) {
        hardwareManager.unregister(s);
      }
      map.clear();
    };
  }, []);

  const getProps = useCallback(
    (key: string, active = false) => {
      const ref = (node: T | null) => {
        const existing = mapRef.current.get(key);
        if (!node) {
          if (existing) {
            hardwareManager.unregister(existing);
            mapRef.current.delete(key);
          }
          return;
        }
        if (existing && existing.element === node) {
          if (existing.active !== active) {
            existing.active = active;
            hardwareManager.wake();
          }
          return;
        }
        if (existing) {
          hardwareManager.unregister(existing);
        }
        const reg: RegisteredSurface = {
          element: node,
          variant,
          active,
          press: createSpringState(0),
          hover: createSpringState(0),
          tiltX: createSpringState(0),
          tiltY: createSpringState(0),
          pressedByPointer: false,
          pressedByKey: false,
        };
        mapRef.current.set(key, reg);
        hardwareManager.register(reg);
      };

      const syncPress = (s: RegisteredSurface) => {
        s.press.target = s.pressedByPointer || s.pressedByKey ? 1.0 : 0.0;
        hardwareManager.wake();
      };

      return {
        ref,
        onPointerEnter: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.hover.target = 1.0;
          hardwareManager.wake();
        },
        onPointerMove: (e: React.PointerEvent<T>) => {
          const s = mapRef.current.get(key);
          if (!s) return;
          const rect = s.element.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) {
            s.tiltX.target = Math.max(-1, Math.min(1, ((e.clientX - rect.left) / rect.width) * 2 - 1));
            s.tiltY.target = Math.max(-1, Math.min(1, ((e.clientY - rect.top) / rect.height) * 2 - 1));
            hardwareManager.wake();
          }
        },
        onPointerLeave: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.hover.target = 0.0;
          s.tiltX.target = 0.0;
          s.tiltY.target = 0.0;
          s.pressedByPointer = false;
          syncPress(s);
        },
        onPointerDown: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.pressedByPointer = true;
          syncPress(s);
        },
        onPointerUp: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.pressedByPointer = false;
          syncPress(s);
        },
        onPointerCancel: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.pressedByPointer = false;
          syncPress(s);
        },
        onKeyDown: (e: React.KeyboardEvent<T>) => {
          if (e.key !== ' ' && e.key !== 'Enter') return;
          const s = mapRef.current.get(key);
          if (!s) return;
          s.pressedByKey = true;
          syncPress(s);
        },
        onKeyUp: (e: React.KeyboardEvent<T>) => {
          if (e.key !== ' ' && e.key !== 'Enter') return;
          const s = mapRef.current.get(key);
          if (!s) return;
          s.pressedByKey = false;
          syncPress(s);
        },
        onFocus: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.hover.target = 1.0;
          hardwareManager.wake();
        },
        onBlur: () => {
          const s = mapRef.current.get(key);
          if (!s) return;
          s.hover.target = 0.0;
          s.pressedByKey = false;
          s.pressedByPointer = false;
          syncPress(s);
        },
      };
    },
    [variant],
  );

  return getProps;
}

