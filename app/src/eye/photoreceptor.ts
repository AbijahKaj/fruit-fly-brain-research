/**
 * Layer 2 (eye): photoreceptor stage.
 *
 * Log luminance (Weber-like contrast) followed by a temporal high-pass,
 * so a static scene fades to zero and motion is what survives.
 */
export class Photoreceptors {
  readonly logLum: Float32Array;
  readonly out: Float32Array;
  private readonly lowpass: Float32Array;
  private primed = false;

  constructor(
    readonly count: number,
    /** Time constant of the high-pass, seconds. */
    readonly tau = 0.05,
  ) {
    this.logLum = new Float32Array(count);
    this.out = new Float32Array(count);
    this.lowpass = new Float32Array(count);
  }

  reset(): void {
    this.primed = false;
    this.out.fill(0);
  }

  update(lum: Float32Array, dt: number): Float32Array {
    const alpha = Math.min(1, dt / this.tau);
    for (let i = 0; i < this.count; i++) {
      const x = Math.log(lum[i]! + 0.02);
      this.logLum[i] = x;
      if (!this.primed) this.lowpass[i] = x;
      this.lowpass[i] = this.lowpass[i]! + alpha * (x - this.lowpass[i]!);
      this.out[i] = x - this.lowpass[i]!;
    }
    this.primed = true;
    return this.out;
  }
}
