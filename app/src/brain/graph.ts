/**
 * Shared graph format, produced by data/extract_flight.py and consumed here
 * and (later) by the PyTorch trainer. Edge weights are raw synapse counts;
 * the sign comes from the presynaptic unit's predicted transmitter.
 */
export type UnitRole = "input" | "brain" | "dn" | "vnc" | "output";

export interface GraphType {
  name: string;
  superclass: string;
  count: number;
  nt: string;
  /** Membrane time constant, seconds. Fitted later; defaults from the extractor. */
  tau: number;
}

export interface GraphJSON {
  version: 1;
  source: string;
  license: string;
  units: {
    count: number;
    bodyId: number[];
    type: number[];
    side: string[];
    role: UnitRole[];
    sign: number[];
    nt: string[];
  };
  types: GraphType[];
  edges: {
    count: number;
    pre: number[];
    post: number[];
    weight: number[];
  };
}

/** Compressed sparse row by POST unit: for unit i, incoming edges are [indptr[i], indptr[i+1]). */
export interface CSR {
  n: number;
  indptr: Int32Array;
  pre: Int32Array;
  /** Signed synapse count: weight * sign(pre). */
  w: Float32Array;
}

export function buildCSR(g: GraphJSON): CSR {
  const n = g.units.count;
  const m = g.edges.count;
  const counts = new Int32Array(n + 1);
  for (let e = 0; e < m; e++) counts[g.edges.post[e]! + 1]!++;
  for (let i = 0; i < n; i++) counts[i + 1]! += counts[i]!;
  const indptr = counts.slice();
  const fill = counts.slice(0, n);
  const pre = new Int32Array(m);
  const w = new Float32Array(m);
  for (let e = 0; e < m; e++) {
    const p = g.edges.pre[e]!;
    const q = g.edges.post[e]!;
    const k = fill[q]!++;
    pre[k] = p;
    w[k] = g.edges.weight[e]! * g.units.sign[p]!;
  }
  return { n, indptr, pre, w };
}

export async function loadGraph(url: string): Promise<GraphJSON> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load graph ${url}: ${res.status}`);
  return (await res.json()) as GraphJSON;
}

/** Indices of units whose type name matches, optionally on one side. */
export function unitsWhere(
  g: GraphJSON,
  pred: (typeName: string, side: string, role: UnitRole) => boolean,
): Int32Array {
  const out: number[] = [];
  for (let i = 0; i < g.units.count; i++) {
    const t = g.types[g.units.type[i]!]!.name;
    if (pred(t, g.units.side[i]!, g.units.role[i]!)) out.push(i);
  }
  return Int32Array.from(out);
}
