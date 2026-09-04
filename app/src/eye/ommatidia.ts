/**
 * Layer 2 (eye): the sampling lattice.
 *
 * One point per ommatidium, in the fly-local frame:
 *   x = right, y = up, -z = forward.
 * Azimuth is 0 straight ahead and positive to the right for both eyes,
 * so the left eye lives at negative azimuth. Elevation is positive up.
 *
 * Drosophila has ~750 ommatidia per eye with ~5 deg spacing. A Fibonacci
 * lattice on the sphere, clipped to one eye's field, gives that within a
 * few percent and is deterministic.
 */

export type EyeSide = "left" | "right";

export interface Ommatidia {
  side: EyeSide;
  count: number;
  /** Unit direction vectors, fly-local, 3 floats per ommatidium. */
  dirs: Float32Array;
  /** Azimuth in radians (0 = forward, + = right). */
  az: Float32Array;
  /** Elevation in radians (+ = up). */
  el: Float32Array;
  /** Index of the neighbour one step toward +azimuth, or -1 if none. */
  azNext: Int32Array;
  /** Mean angular spacing between neighbours, radians. */
  spacing: number;
}

const DEG = Math.PI / 180;

// Field of one eye, expressed for the right eye and mirrored for the left.
const AZ_MIN = -15 * DEG; // 15 deg of binocular overlap past the midline
const AZ_MAX = 165 * DEG;
const EL_MIN = -60 * DEG;
const EL_MAX = 75 * DEG;

export function buildOmmatidia(side: EyeSide, target = 750): Ommatidia {
  // Fraction of the sphere covered by one eye, used to size the lattice.
  const frac = ((AZ_MAX - AZ_MIN) / (2 * Math.PI)) * ((Math.sin(EL_MAX) - Math.sin(EL_MIN)) / 2);
  const n = Math.round(target / frac);
  const golden = Math.PI * (3 - Math.sqrt(5));
  const spacing = Math.sqrt((4 * Math.PI) / n);

  const azList: number[] = [];
  const elList: number[] = [];
  for (let i = 0; i < n; i++) {
    const sinEl = 1 - (2 * (i + 0.5)) / n;
    const el = Math.asin(sinEl);
    let az = (i * golden) % (2 * Math.PI);
    if (az > Math.PI) az -= 2 * Math.PI;
    const mirrored = side === "right" ? az : -az;
    if (mirrored >= AZ_MIN && mirrored <= AZ_MAX && el >= EL_MIN && el <= EL_MAX) {
      azList.push(az);
      elList.push(el);
    }
  }

  const count = azList.length;
  const az = Float32Array.from(azList);
  const el = Float32Array.from(elList);
  const dirs = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const a = az[i]!;
    const e = el[i]!;
    dirs[i * 3 + 0] = Math.sin(a) * Math.cos(e);
    dirs[i * 3 + 1] = Math.sin(e);
    dirs[i * 3 + 2] = -Math.cos(a) * Math.cos(e);
  }

  // Nearest neighbour toward +azimuth, for the stub motion detector.
  const azNext = new Int32Array(count).fill(-1);
  const maxAngle = spacing * 0.9;
  for (let i = 0; i < count; i++) {
    const ta = az[i]! + spacing / Math.max(0.2, Math.cos(el[i]!));
    const te = el[i]!;
    const tx = Math.sin(ta) * Math.cos(te);
    const ty = Math.sin(te);
    const tz = -Math.cos(ta) * Math.cos(te);
    let best = -1;
    let bestDot = Math.cos(maxAngle);
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const dot = dirs[j * 3]! * tx + dirs[j * 3 + 1]! * ty + dirs[j * 3 + 2]! * tz;
      if (dot > bestDot) {
        bestDot = dot;
        best = j;
      }
    }
    azNext[i] = best;
  }

  return { side, count, dirs, az, el, azNext, spacing };
}
