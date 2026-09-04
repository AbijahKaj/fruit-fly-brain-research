/**
 * Layer 2 (eye): render what the fly sees and sample it at the ommatidia.
 *
 * Six 90-degree cameras are parented to the fly and render into small
 * offscreen targets. Each ommatidium direction is mapped once to a
 * (face, pixel) pair, so per frame we only read back six tiny buffers
 * and gather. No cube-map conventions are involved: the cameras are ours.
 *
 * Later this can move to a single GPU gather pass; for milestone 1 the
 * CPU path is simple and fast enough (6 x 48 x 48 pixels).
 */
import * as THREE from "three";
import type { Ommatidia } from "./ommatidia";

interface Face {
  dir: [number, number, number];
  up: [number, number, number];
}

const FACES: Face[] = [
  { dir: [0, 0, -1], up: [0, 1, 0] }, // front
  { dir: [0, 0, 1], up: [0, 1, 0] }, // back
  { dir: [-1, 0, 0], up: [0, 1, 0] }, // left
  { dir: [1, 0, 0], up: [0, 1, 0] }, // right
  { dir: [0, 1, 0], up: [0, 0, 1] }, // up
  { dir: [0, -1, 0], up: [0, 0, -1] }, // down
];

interface Lookup {
  face: Int32Array;
  offset: Int32Array; // byte offset into the face's RGBA buffer
}

export class CompoundEye {
  readonly size: number;
  private readonly cams: THREE.PerspectiveCamera[] = [];
  private readonly targets: THREE.WebGLRenderTarget[] = [];
  private readonly pixels: Uint8Array[] = [];
  private readonly lookups = new Map<Ommatidia, Lookup>();

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly scene: THREE.Scene,
    parent: THREE.Object3D,
    size = 48,
  ) {
    this.size = size;
    const zero = new THREE.Vector3();
    for (const f of FACES) {
      const cam = new THREE.PerspectiveCamera(90, 1, 0.05, 500);
      const m = new THREE.Matrix4().lookAt(zero, new THREE.Vector3(...f.dir), new THREE.Vector3(...f.up));
      cam.quaternion.setFromRotationMatrix(m);
      cam.layers.set(0); // the fly's own body lives on layer 1 and is invisible to its eyes
      parent.add(cam);
      this.cams.push(cam);
      this.targets.push(
        new THREE.WebGLRenderTarget(size, size, {
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter,
          depthBuffer: true,
        }),
      );
      this.pixels.push(new Uint8Array(size * size * 4));
    }
  }

  /** Precompute the (face, pixel) for each ommatidium of a lattice. */
  register(omm: Ommatidia): void {
    const face = new Int32Array(omm.count).fill(-1);
    const offset = new Int32Array(omm.count);
    const d = new THREE.Vector3();
    const inv = this.cams.map((c) => c.quaternion.clone().invert());
    for (let i = 0; i < omm.count; i++) {
      d.set(omm.dirs[i * 3]!, omm.dirs[i * 3 + 1]!, omm.dirs[i * 3 + 2]!);
      for (let k = 0; k < this.cams.length; k++) {
        const c = d.clone().applyQuaternion(inv[k]!);
        if (c.z >= 0) continue;
        const x = c.x / -c.z;
        const y = c.y / -c.z;
        if (Math.abs(x) > 1 || Math.abs(y) > 1) continue;
        const px = Math.min(this.size - 1, Math.floor(((x + 1) / 2) * this.size));
        const py = Math.min(this.size - 1, Math.floor(((y + 1) / 2) * this.size));
        face[i] = k;
        offset[i] = (py * this.size + px) * 4;
        break;
      }
    }
    this.lookups.set(omm, { face, offset });
  }

  /** Render all six faces from the fly's current pose and read them back. */
  render(): void {
    const r = this.renderer;
    for (let k = 0; k < this.cams.length; k++) {
      r.setRenderTarget(this.targets[k]!);
      r.render(this.scene, this.cams[k]!);
      r.readRenderTargetPixels(this.targets[k]!, 0, 0, this.size, this.size, this.pixels[k]!);
    }
    r.setRenderTarget(null);
  }

  /** Gather luminance (0..1) per ommatidium from the last render. */
  sample(omm: Ommatidia, out: Float32Array): void {
    const lk = this.lookups.get(omm);
    if (!lk) throw new Error("ommatidia not registered with this eye");
    for (let i = 0; i < omm.count; i++) {
      const f = lk.face[i]!;
      if (f < 0) {
        out[i] = 0;
        continue;
      }
      const p = this.pixels[f]!;
      const o = lk.offset[i]!;
      out[i] = (0.2126 * p[o]! + 0.7152 * p[o + 1]! + 0.0722 * p[o + 2]!) / 255;
    }
  }
}
