/**
 * Hassenstein-Reichardt correlator over azimuthal neighbours.
 *
 * Stands in for T4/T5 until milestone 3. Output > 0 when the pattern moves
 * toward +azimuth (rightward on the retina) for that eye.
 */
import type { Ommatidia } from "../eye/ommatidia";

export class HRCorrelator {
  private readonly delayed: Float32Array;
  value = 0;

  constructor(
    private readonly omm: Ommatidia,
    /** Delay-arm time constant, seconds. */
    public tauDelay = 0.03,
  ) {
    this.delayed = new Float32Array(omm.count);
  }

  reset(): void {
    this.delayed.fill(0);
    this.value = 0;
  }

  update(s: Float32Array, dt: number): number {
    const d = this.delayed;
    const omm = this.omm;
    const alpha = Math.min(1, dt / this.tauDelay);
    for (let i = 0; i < omm.count; i++) d[i] = d[i]! + alpha * (s[i]! - d[i]!);
    let sum = 0;
    let pairs = 0;
    for (let i = 0; i < omm.count; i++) {
      const j = omm.azNext[i]!;
      if (j < 0) continue;
      sum += d[i]! * s[j]! - s[i]! * d[j]!;
      pairs++;
    }
    this.value = pairs ? sum / pairs : 0;
    return this.value;
  }
}
