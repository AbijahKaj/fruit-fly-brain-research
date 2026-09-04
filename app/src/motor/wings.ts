/**
 * Layer 4 (motor): wing amplitudes to body forces.
 *
 * Deliberately a cartoon. Mean amplitude sets thrust, the left/right
 * difference sets yaw torque and a visual bank angle. No aerodynamics,
 * no wingbeat CPG, no haltere feedback; those are milestone 5.
 */
import type { MotorCommand } from "../brain/types";

export interface BodyForces {
  /** Forward acceleration, units/s^2. */
  thrust: number;
  /** Yaw angular acceleration, rad/s^2. Positive = left (counterclockwise from above). */
  yawTorque: number;
  /** Target bank angle, radians. Positive = right wing down. */
  bank: number;
}

export interface WingParams {
  /** Amplitude at which thrust balances drag at rest. */
  hoverAmp: number;
  thrustGain: number;
  yawGain: number;
  bankGain: number;
}

export const defaultWingParams: WingParams = {
  hoverAmp: 0.5,
  thrustGain: 6,
  yawGain: 16,
  bankGain: 1.0,
};

export function wingsToForces(cmd: MotorCommand, p: WingParams = defaultWingParams): BodyForces {
  const mean = (cmd.left + cmd.right) / 2;
  const diff = cmd.left - cmd.right; // > 0: left wing harder -> yaw right
  return {
    thrust: p.thrustGain * (mean - p.hoverAmp),
    yawTorque: -p.yawGain * diff,
    bank: -p.bankGain * diff,
  };
}
