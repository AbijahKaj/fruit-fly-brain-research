/**
 * Shared graph format, produced by data/extract_*.py and consumed here and
 * (later) by the PyTorch trainer. Edge weights are raw synapse counts; the
 * sign comes from the presynaptic unit's predicted transmitter.
 *
 * Two on-disk forms load into the same in-memory `Graph`:
 *   v1  one JSON file with plain arrays (small graphs)
 *   v2  JSON header + .bin of typed arrays (see data/graphio.py)
 */
export const ROLES = ["input", "brain", "dn", "vnc", "output", "optic"] as const;
export type UnitRole = (typeof ROLES)[number];
export const SIDES = ["L", "R", "M"] as const;

export interface GraphType {
  name: string;
  superclass: string;
  count: number;
  nt: string;
  /** Membrane time constant, seconds. */
  tau: number;
}

export interface Columns {
  count: number;
  side: Int8Array;
  h1: Int16Array;
  h2: Int16Array;
  az: Float32Array;
  el: Float32Array;
}

export interface Graph {
  version: number;
  source: string;
  n: number;
  m: number;
  types: GraphType[];
  type: Int32Array;
  side: Int8Array;
  role: Int8Array;
  sign: Int8Array;
  /** Column index into `columns`, or -1. */
  col: Int32Array;
  bodyId: Float64Array;
  pre: Int32Array;
  post: Int32Array;
  weight: Float32Array;
  columns: Columns;
}

/** Compressed sparse row by POST unit: incoming edges of unit i are [indptr[i], indptr[i+1]). */
export interface CSR {
  n: number;
  indptr: Int32Array;
  pre: Int32Array;
  /** Signed synapse count: weight * sign(pre). */
  w: Float32Array;
}

export function buildCSR(g: Pick<Graph, "n" | "m" | "pre" | "post" | "weight" | "sign">): CSR {
  const w = new Float32Array(g.m);
  for (let e = 0; e < g.m; e++) w[e] = g.weight[e]! * g.sign[g.pre[e]!]!;
  return buildCSRWeighted(g.n, g.m, g.pre, g.post, w);
}

/** CSR from per-edge weights that are already signed and scaled. */
export function buildCSRWeighted(n: number, m: number, epre: Int32Array, epost: Int32Array, ew: Float32Array): CSR {
  const g = { n, m, pre: epre, post: epost };
  const counts = new Int32Array(n + 1);
  for (let e = 0; e < m; e++) counts[g.post[e]! + 1]!++;
  for (let i = 0; i < n; i++) counts[i + 1]! += counts[i]!;
  const indptr = counts.slice();
  const fill = counts.slice(0, n);
  const pre = new Int32Array(m);
  const w = new Float32Array(m);
  for (let e = 0; e < m; e++) {
    const k = fill[g.post[e]!]!++;
    pre[k] = g.pre[e]!;
    w[k] = ew[e]!;
  }
  return { n, indptr, pre, w };
}

interface V1JSON {
  version: 1;
  source: string;
  units: { count: number; bodyId: number[]; type: number[]; side: string[]; role: string[]; sign: number[] };
  types: GraphType[];
  edges: { count: number; pre: number[]; post: number[]; weight: number[] };
}

interface ArrayDesc {
  dtype: "int8" | "int16" | "int32" | "float32" | "float64";
  offset: number;
  length: number;
}

interface V2Header {
  version: 2;
  source: string;
  units: { count: number };
  edges: { count: number };
  columns: { count: number };
  roles: string[];
  sides: string[];
  types: GraphType[];
  arrays: Record<string, ArrayDesc>;
}

const CTOR = {
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  float32: Float32Array,
  float64: Float64Array,
} as const;

function view<K extends keyof typeof CTOR>(buf: ArrayBuffer, d: ArrayDesc, dtype: K): InstanceType<(typeof CTOR)[K]> {
  if (d.dtype !== dtype) throw new Error(`array dtype ${d.dtype}, expected ${dtype}`);
  return new CTOR[dtype](buf, d.offset, d.length) as InstanceType<(typeof CTOR)[K]>;
}

const emptyColumns = (): Columns => ({
  count: 0,
  side: new Int8Array(0),
  h1: new Int16Array(0),
  h2: new Int16Array(0),
  az: new Float32Array(0),
  el: new Float32Array(0),
});

export function fromV1(j: V1JSON): Graph {
  const n = j.units.count;
  const roleIdx = (r: string): number => {
    const i = (ROLES as readonly string[]).indexOf(r);
    if (i < 0) throw new Error(`unknown role ${r}`);
    return i;
  };
  return {
    version: 1,
    source: j.source,
    n,
    m: j.edges.count,
    types: j.types,
    type: Int32Array.from(j.units.type),
    side: Int8Array.from(j.units.side, (s) => Math.max(0, (SIDES as readonly string[]).indexOf(s))),
    role: Int8Array.from(j.units.role, roleIdx),
    sign: Int8Array.from(j.units.sign),
    col: new Int32Array(n).fill(-1),
    bodyId: Float64Array.from(j.units.bodyId),
    pre: Int32Array.from(j.edges.pre),
    post: Int32Array.from(j.edges.post),
    weight: Float32Array.from(j.edges.weight),
    columns: emptyColumns(),
  };
}

export function fromV2(h: V2Header, buf: ArrayBuffer): Graph {
  const a = h.arrays;
  const A = (k: string): ArrayDesc => {
    const d = a[k];
    if (!d) throw new Error(`missing array ${k}`);
    return d;
  };
  const roleMap = h.roles.map((r) => (ROLES as readonly string[]).indexOf(r));
  const role = view(buf, A("units.role"), "int8").slice();
  for (let i = 0; i < role.length; i++) role[i] = roleMap[role[i]!]!;
  return {
    version: 2,
    source: h.source,
    n: h.units.count,
    m: h.edges.count,
    types: h.types,
    type: view(buf, A("units.type"), "int32"),
    side: view(buf, A("units.side"), "int8"),
    role,
    sign: view(buf, A("units.sign"), "int8"),
    col: view(buf, A("units.col"), "int32"),
    bodyId: view(buf, A("units.bodyId"), "float64"),
    pre: view(buf, A("edges.pre"), "int32"),
    post: view(buf, A("edges.post"), "int32"),
    weight: view(buf, A("edges.weight"), "float32"),
    columns: {
      count: h.columns.count,
      side: view(buf, A("columns.side"), "int8"),
      h1: view(buf, A("columns.h1"), "int16"),
      h2: view(buf, A("columns.h2"), "int16"),
      az: view(buf, A("columns.az"), "float32"),
      el: view(buf, A("columns.el"), "float32"),
    },
  };
}

/** Load either form by URL stem or .json path. */
export async function loadGraph(url: string): Promise<Graph> {
  const jsonUrl = url.endsWith(".json") ? url : `${url}.json`;
  const res = await fetch(jsonUrl);
  if (!res.ok) throw new Error(`failed to load graph ${jsonUrl}: ${res.status}`);
  const j = (await res.json()) as V1JSON | V2Header;
  if (j.version === 1) return fromV1(j);
  const binUrl = jsonUrl.replace(/\.json$/, ".bin");
  const bin = await fetch(binUrl);
  if (!bin.ok) throw new Error(`failed to load graph ${binUrl}: ${bin.status}`);
  return fromV2(j, await bin.arrayBuffer());
}

export function typeName(g: Graph, i: number): string {
  return g.types[g.type[i]!]!.name;
}

export function sideName(g: Graph, i: number): string {
  return SIDES[g.side[i]!] ?? "M";
}

export function roleName(g: Graph, i: number): UnitRole {
  return ROLES[g.role[i]!] ?? "input";
}

/** Indices of units satisfying a predicate on (type name, side, role). */
export function unitsWhere(g: Graph, pred: (typeName: string, side: string, role: UnitRole) => boolean): Int32Array {
  const out: number[] = [];
  for (let i = 0; i < g.n; i++) if (pred(typeName(g, i), sideName(g, i), roleName(g, i))) out.push(i);
  return Int32Array.from(out);
}

/** Per-unit tau from the type table. */
export function unitTau(g: Graph): Float32Array {
  const tau = new Float32Array(g.n);
  for (let i = 0; i < g.n; i++) tau[i] = g.types[g.type[i]!]!.tau;
  return tau;
}
