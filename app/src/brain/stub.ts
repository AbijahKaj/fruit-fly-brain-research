/**
 * Layer 3 (brain), milestone 1 stub.
 *
 * HR correlator per eye, summed into one yaw estimate, and a proportional
 * controller that turns the fly with the pattern (optomotor response).
 * No connectome here; it exists so the loop can be proven before the real
 * graph goes in.
 *
 *   turn > 0  fly should yaw right (clockwise seen from above)
 */
import type { Ommatidia } from "../eye/ommatidia";
import { HRCorrelator } from "./hr";
import type { Brain, EyeInput, MotorCommand } from "./types";

export class StubBrain implements Brain {
  readonly name = "stub: HR correlator + P controller";
  readonly lattice = "fibonacci" as const;

  /** Turn command per unit of summed flow. */
  gain = 60;
  /** Hover amplitude for both wings. */
  baseAmp = 0.5;
  /** Maximum wing asymmetry. */
  maxTurn = 0.5;
  /** Smoothing of the turn command, seconds. */
  tauOut = 0.05;

  private readonly hrL: HRCorrelator;
  private readonly hrR: HRCorrelator;
  private turn = 0;

  constructor(ommL: Ommatidia, ommR: Ommatidia) {
    this.hrL = new HRCorrelator(ommL);
    this.hrR = new HRCorrelator(ommR);
  }

  reset(): void {
    this.hrL.reset();
    this.hrR.reset();
    this.turn = 0;
  }

  step(input: EyeInput, dt: number): MotorCommand {
    const rot = this.hrL.update(input.left, dt) + this.hrR.update(input.right, dt);
    const raw = Math.max(-this.maxTurn, Math.min(this.maxTurn, this.gain * rot));
    this.turn += (raw - this.turn) * Math.min(1, dt / this.tauOut);
    // Yaw right = left wing beats harder.
    return {
      left: clamp01(this.baseAmp + this.turn / 2),
      right: clamp01(this.baseAmp - this.turn / 2),
    };
  }

  telemetry(): Record<string, number> {
    return { flowL: this.hrL.value, flowR: this.hrR.value, turn: this.turn };
  }
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
