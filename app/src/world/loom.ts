/**
 * Layer 1 (world): an object that approaches the fly. The classic looming
 * stimulus, here as a real sphere in the scene so the eye sees it the same
 * way it sees a pillar.
 *
 * `retinal` keeps the approach direction fixed in the fly's frame (open-loop
 * assay: the fly cannot escape it). Otherwise the sphere flies a straight
 * world-space line toward where the fly was when launched, so turning away
 * works.
 */
import * as THREE from "three";

export interface LoomParams {
  /** Azimuth of approach in the fly frame, radians, positive = right. */
  az: number;
  el: number;
  /** Start distance and approach speed, world units (fly body ~0.6 long). */
  startDistance: number;
  speed: number;
  radius: number;
  retinal: boolean;
  /** Restart from startDistance after passing the fly. */
  loop: boolean;
}

export const defaultLoomParams: LoomParams = {
  az: Math.PI / 4,
  el: 0,
  startDistance: 12,
  speed: 6,
  radius: 0.6,
  retinal: true,
  loop: true,
};

export class Loomer {
  readonly mesh: THREE.Mesh;
  readonly params: LoomParams = { ...defaultLoomParams };
  active = false;
  /** Current distance from the fly, world units. */
  distance = Infinity;
  /** Passes that came within radius + 0.1 of the fly, and the closest approach of the current pass. */
  hits = 0;
  closest = Infinity;
  private readonly velocity = new THREE.Vector3();
  private readonly tmp = new THREE.Vector3();

  constructor(scene: THREE.Scene) {
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 24, 16),
      new THREE.MeshStandardMaterial({ color: "#151515", roughness: 0.9 }),
    );
    this.mesh.visible = false;
    scene.add(this.mesh);
  }

  /** Start an approach with the given parameter overrides. */
  launch(flyPos: THREE.Vector3, yaw: number, p: Partial<LoomParams> = {}): void {
    Object.assign(this.params, p);
    this.active = true;
    this.mesh.visible = true;
    this.mesh.scale.setScalar(this.params.radius);
    this.distance = this.params.startDistance;
    this.closest = Infinity;
    this.place(flyPos, yaw);
    // world-space velocity toward the fly's position at launch
    this.velocity.copy(flyPos).sub(this.mesh.position).normalize().multiplyScalar(this.params.speed);
  }

  stop(): void {
    this.active = false;
    this.mesh.visible = false;
    this.distance = Infinity;
  }

  /** Fly-frame direction (az right-positive, el up) to world, given the fly's yaw. */
  private direction(yaw: number, out: THREE.Vector3): THREE.Vector3 {
    const { az, el } = this.params;
    out.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el));
    return out.applyAxisAngle(THREE.Object3D.DEFAULT_UP, yaw);
  }

  private place(flyPos: THREE.Vector3, yaw: number): void {
    this.direction(yaw, this.tmp);
    this.mesh.position.copy(flyPos).addScaledVector(this.tmp, this.distance);
  }

  update(dt: number, flyPos: THREE.Vector3, yaw: number): void {
    if (!this.active) return;
    const p = this.params;
    if (p.retinal) {
      this.distance -= p.speed * dt;
      this.place(flyPos, yaw);
    } else {
      this.mesh.position.addScaledVector(this.velocity, dt);
      this.distance = this.mesh.position.distanceTo(flyPos);
    }
    this.closest = Math.min(this.closest, this.distance);
    const passed = p.retinal ? this.distance <= p.radius : this.tmp.copy(flyPos).sub(this.mesh.position).dot(this.velocity) < 0;
    if (passed) {
      if (this.closest <= p.radius + 0.1) this.hits++;
      if (p.loop) this.launch(flyPos, yaw);
      else this.stop();
    }
  }
}
