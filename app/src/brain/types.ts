/**
 * Layer 3 (brain): the only interface the rest of the app cares about.
 *
 * Input is the luminance each eye sampled from the scene, one value per
 * optic-lobe column. Output is a wing command.
 */
export interface EyeInput {
  /** Luminance 0..1 per column (the brain drives its own photoreceptors). */
  lumLeft: Float32Array;
  lumRight: Float32Array;
}

export interface MotorCommand {
  /** Left wingbeat amplitude, 0..1. */
  left: number;
  /** Right wingbeat amplitude, 0..1. */
  right: number;
}

export interface Brain {
  readonly name: string;
  step(input: EyeInput, dt: number): MotorCommand;
  reset(): void;
  /** Free-form numbers for the HUD. */
  telemetry(): Record<string, number>;
}
