/**
 * Layer 2 (eye): the sampling lattice.
 *
 * One point per ommatidium, in the fly-local frame:
 *   x = right, y = up, -z = forward.
 * Azimuth is 0 straight ahead and positive to the right for both eyes,
 * so the left eye lives at negative azimuth. Elevation is positive up.
 *
 * The lattice is the connectome's own: one ommatidium per optic-lobe column,
 * at the direction the extractor calibrated from T4 preferred directions
 * (about 885 columns per eye, 5 deg spacing).
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
  /** Mean angular spacing between neighbours, radians. */
  spacing: number;
  /** Connectome column index per ommatidium. */
  col: Int32Array;
}

const DEG = Math.PI / 180;

/**
 * Lattice from connectome columns: one ommatidium per optic-lobe column,
 * at the direction the extractor calibrated from T4 preferred directions.
 */
export function ommatidiaFromColumns(
  side: EyeSide,
  columns: { count: number; side: Int8Array; az: Float32Array; el: Float32Array },
  spacingDeg = 5,
): Ommatidia {
  const want = side === "left" ? 0 : 1;
  const idx: number[] = [];
  for (let c = 0; c < columns.count; c++) if (columns.side[c] === want) idx.push(c);
  const az = Float32Array.from(idx, (c) => columns.az[c]!);
  const el = Float32Array.from(idx, (c) => columns.el[c]!);
  const omm = finish(side, az, el, spacingDeg * DEG);
  omm.col = Int32Array.from(idx);
  return omm;
}

function finish(side: EyeSide, az: Float32Array, el: Float32Array, spacing: number): Ommatidia {
  const count = az.length;
  const dirs = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const a = az[i]!;
    const e = el[i]!;
    dirs[i * 3 + 0] = Math.sin(a) * Math.cos(e);
    dirs[i * 3 + 1] = Math.sin(e);
    dirs[i * 3 + 2] = -Math.cos(a) * Math.cos(e);
  }

  return { side, count, dirs, az, el, spacing, col: new Int32Array(0) };
}
