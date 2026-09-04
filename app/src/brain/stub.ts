/**
 * Layer 3 (brain), milestone 1 stub.
 *
 * A Hassenstein-Reichardt correlator per eye over neighbouring ommatidia,
 * summed into one yaw estimate, and a proportional controller that turns
 * the fly with the pattern (the optomotor response). No connectome here;
 * this exists so the loop can be proven before the real graph goes in.
 *
 * Sign conventions:
 *   flow  > 0  pattern moving toward +azimuth (rightward on the retina)
 *   turn  > 0  fly should yaw right (clockwise seen from above)
 */
import type { Ommatidia } from "../eye/ommatidia";
import type { Brain, EyeInput, MotorCommand } from "./types";

export class StubBrain implements Brain {
  readonly name = "stub: HR correlator + P controller";

  /** Turn command per unit of summed flow. */
  gain = 60;
  /** Correlator delay time constant, seconds. */
  tauDelay = 0.03;
  /** Hover amplitude for both wings. */
  baseAmp = 0.5;
  /** Maximum wing asymmetry. */
  maxTurn = 0.5;
  /** Smoothing of the turn command, seconds. The raw correlator is noisy frame to frame. */
  tauOut = 0.05;

  private readonly delayedL: Float32Array;
  private readonly delayedR: Float32Array;
  private flowL = 0;
  private flowR = 0;
  private turn = 0;

  constructor(
    private readonly ommL: Ommatidia,
    private readonly ommR: Ommatidia,
  ) {
    this.delayedL = new Float32Array(ommL.count);
    this.delayedR = new Float32Array(ommR.count);
  }

  reset(): void {
    this.delayedL.fill(0);
    this.delayedR.fill(0);
    this.flowL = this.flowR = this.turn = 0;
  }

  step(input: EyeInput, dt: number): MotorCommand {
    this.flowL = this.correlate(input.left, this.delayedL, this.ommL, dt);
    this.flowR = this.correlate(input.right, this.delayedR, this.ommR, dt);
    const rot = this.flowL + this.flowR;
    const raw = Math.max(-this.maxTurn, Math.min(this.maxTurn, this.gain * rot));
    this.turn += (raw - this.turn) * Math.min(1, dt / this.tauOut);
    // Yaw right = left wing beats harder.
    return {
      left: clamp01(this.baseAmp + this.turn / 2),
      right: clamp01(this.baseAmp - this.turn / 2),
    };
  }

  telemetry(): Record<string, number> {
    return { flowL: this.flowL, flowR: this.flowR, turn: this.turn };
  }

  private correlate(s: Float32Array, d: Float32Array, omm: Ommatidia, dt: number): number {
    const alpha = Math.min(1, dt / this.tauDelay);
    let sum = 0;
    let pairs = 0;
    for (let i = 0; i < omm.count; i++) {
      d[i] = d[i]! + alpha * (s[i]! - d[i]!);
    }
    for (let i = 0; i < omm.count; i++) {
      const j = omm.azNext[i]!;
      if (j < 0) continue;
      sum += d[i]! * s[j]! - s[i]! * d[j]!;
      pairs++;
    }
    return pairs ? sum / pairs : 0;
  }
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
