/**
 * Layer 1 (body): rigid-body state of the fly and its integration.
 *
 * Yaw is a rotation about +Y (positive = left). Altitude is held constant
 * for now; pitch is unused. Roll is cosmetic (bank into turns).
 */
import * as THREE from "three";
import type { BodyForces } from "../motor/wings";

export interface FlyState {
  position: THREE.Vector3;
  yaw: number;
  roll: number;
  yawRate: number;
  speed: number;
}

export interface BodyParams {
  yawDamping: number;
  drag: number;
  altitude: number;
  rollResponse: number;
  maxSpeed: number;
}

export const defaultBodyParams: BodyParams = {
  yawDamping: 4,
  drag: 1.5,
  altitude: 2,
  rollResponse: 8,
  maxSpeed: 6,
};

export class FlyBody {
  readonly state: FlyState = {
    position: new THREE.Vector3(0, 2, 0),
    yaw: 0,
    roll: 0,
    yawRate: 0,
    speed: 0,
  };
  private readonly euler = new THREE.Euler(0, 0, 0, "YXZ");

  constructor(readonly params: BodyParams = defaultBodyParams) {}

  reset(): void {
    this.state.position.set(0, this.params.altitude, 0);
    this.state.yaw = this.state.roll = this.state.yawRate = this.state.speed = 0;
  }

  step(f: BodyForces, dt: number): void {
    const s = this.state;
    const p = this.params;
    s.yawRate += (f.yawTorque - p.yawDamping * s.yawRate) * dt;
    s.yaw += s.yawRate * dt;
    s.speed += (f.thrust - p.drag * s.speed) * dt;
    s.speed = Math.max(0, Math.min(p.maxSpeed, s.speed));
    s.roll += (f.bank - s.roll) * Math.min(1, p.rollResponse * dt);
    // yaw = 0 faces -Z; positive yaw rotates the nose toward -X.
    s.position.x += -Math.sin(s.yaw) * s.speed * dt;
    s.position.z += -Math.cos(s.yaw) * s.speed * dt;
    s.position.y = p.altitude;
  }

  /** Forward unit vector in world space. */
  forward(out: THREE.Vector3): THREE.Vector3 {
    return out.set(-Math.sin(this.state.yaw), 0, -Math.cos(this.state.yaw));
  }

  /** `root` gets position and yaw (the eye lives there, level); `bodyMesh`, if given, gets the bank. */
  applyTo(root: THREE.Object3D, bodyMesh?: THREE.Object3D): void {
    root.position.copy(this.state.position);
    this.euler.set(0, this.state.yaw, 0);
    root.quaternion.setFromEuler(this.euler);
    if (bodyMesh) bodyMesh.rotation.z = this.state.roll;
  }
}
