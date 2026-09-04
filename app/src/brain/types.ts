/**
 * Layer 3 (brain): the only interface the rest of the app cares about.
 *
 * Input is the photoreceptor output of both eyes. Output is a wing command.
 * Milestone 1 uses a hand-written stub; milestone 2 puts real MaleCNS
 * wiring behind this same interface.
 */
export interface EyeInput {
  left: Float32Array;
  right: Float32Array;
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
