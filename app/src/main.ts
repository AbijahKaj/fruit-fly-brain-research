/**
 * The closed loop:
 *
 *   world/body  ->  eye  ->  brain (optic-v2)  ->  motor  ->  world/body
 *
 * The eye samples the scene at the connectome's own column directions, the
 * brain is the per-column MaleCNS optic lobe with fitted parameters, the
 * motor layer turns its HS and LC4/LPLC2 readouts into wing amplitudes.
 */
import * as THREE from "three";
import { buildWorld, FLY_LAYER } from "./world/scene";
import { FlyBody } from "./world/fly";
import { Loomer } from "./world/loom";
import { ommatidiaFromColumns, type Ommatidia } from "./eye/ommatidia";
import { CompoundEye } from "./eye/eye";
import { OpticBrain } from "./brain/optic";
import { gpuAvailable } from "./brain/net-backend";
import type { NetBackendKind } from "./brain/net-backend";
import { loadGraph, type Graph } from "./brain/graph";
import { loadFlyvis, type FlyvisParams } from "./brain/flyvis";
import type { MotorCommand } from "./brain/types";
import { wingsToForces, defaultWingParams } from "./motor/wings";
import { SimLoop } from "./sim/loop";
import { Hud } from "./ui/hud";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};
const BASE = import.meta.env.BASE_URL;

// ---------- renderer + world ----------
const canvas = $<HTMLCanvasElement>("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(window.innerWidth, window.innerHeight);

const world = buildWorld();
const body = new FlyBody();
body.applyTo(world.flyRoot, world.flyBody);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 500);
camera.layers.enable(FLY_LAYER);
window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// ---------- eye ----------
interface EyeSet {
  ommL: Ommatidia;
  ommR: Ommatidia;
  lumL: Float32Array;
  lumR: Float32Array;
}
const eye = new CompoundEye(renderer, world.scene, world.flyRoot, 48);
let eyes: EyeSet | undefined;

// ---------- brain ----------
let optic: OpticBrain | undefined;
let brainOn = true;
const hover: MotorCommand = { left: defaultWingParams.hoverAmp, right: defaultWingParams.hoverAmp };
let loadStatus = "loading graph…";

// Fitted parameters (train/) win over the raw flyvis transfer when present.
const loadParams = (): ReturnType<typeof loadFlyvis> =>
  loadFlyvis(`${BASE}graphs/fitted-params.json`).catch(() => loadFlyvis(`${BASE}graphs/flyvis-params.json`));
const netSel = $<HTMLSelectElement>("netSel");
netSel.value = gpuAvailable() ? "gpu" : "worker";
if (!gpuAvailable()) netSel.querySelector<HTMLOptionElement>('option[value="gpu"]')!.disabled = true;
let opticGraph: [Graph, FlyvisParams] | undefined;
const buildOptic = (): void => {
  if (!opticGraph || !eyes) return;
  const [g, fv] = opticGraph;
  optic?.net.dispose();
  optic = new OpticBrain(g, fv, eyes.ommL, eyes.ommR, netSel.value as NetBackendKind);
  syncBrainControls();
  sidesCalibrated = false;
  void calibrateSides();
};
netSel.addEventListener("change", buildOptic);
Promise.all([loadGraph(`${BASE}graphs/optic-v2`), loadParams()])
  .then(([g, fv]) => {
    const ommL = ommatidiaFromColumns("left", g.columns);
    const ommR = ommatidiaFromColumns("right", g.columns);
    eye.register(ommL);
    eye.register(ommR);
    eyes = { ommL, ommR, lumL: new Float32Array(ommL.count), lumR: new Float32Array(ommR.count) };
    opticGraph = [g, fv];
    buildOptic();
  })
  .catch((err) => {
    loadStatus = String(err);
    console.error(err);
  });

const loomer = new Loomer(world.scene);
/** Collisions with pillars (box half-width 0.75 plus the fly's 0.3), counted once per contact. */
let collisions = 0;
let inContact = false;
const countCollisions = (): void => {
  const p = body.state.position;
  let touching = false;
  const near = (o: THREE.Object3D): boolean => Math.abs(o.position.x - p.x) < 1.05 && Math.abs(o.position.z - p.z) < 1.05;
  for (const o of world.obstacles) if (near(o)) touching = true;
  if (world.course.visible) for (const o of world.course.children) if (near(o)) touching = true;
  if (loomer.active && loomer.distance < loomer.params.radius + 0.1) touching = true;
  if (touching && !inContact) collisions++;
  inContact = touching;
};

// ---------- ui ----------
const hud = new Hud($<HTMLCanvasElement>("eyes"), $("stats"));
const netCanvas = $<HTMLCanvasElement>("net");
const drumSlider = $<HTMLInputElement>("drum");
const drumVal = $("drumVal");
const brainToggle = $<HTMLInputElement>("brainOn");
const followToggle = $<HTMLInputElement>("follow");
const wScaleSlider = $<HTMLInputElement>("wScale");
const outGainSlider = $<HTMLInputElement>("outGain");
const loomGainSlider = $<HTMLInputElement>("loomGain");

const syncBrainControls = (): void => {
  const gain = parseFloat(wScaleSlider.value);
  const outputGain = parseFloat(outGainSlider.value);
  const loomGain = parseFloat(loomGainSlider.value);
  $("wScaleVal").textContent = gain.toFixed(2);
  $("outGainVal").textContent = outputGain.toFixed(2);
  $("loomGainVal").textContent = loomGain.toFixed(1);
  if (optic) Object.assign(optic.params, { wScale: gain, outputGain, loomGain });
};
for (const el of [wScaleSlider, outGainSlider, loomGainSlider]) el.addEventListener("input", syncBrainControls);
brainToggle.addEventListener("change", () => {
  brainOn = brainToggle.checked;
  reset();
});
brainToggle.checked = brainOn;

let drumOmega = 0; // rad/s about +Y; positive = counterclockwise seen from above
const setDrum = (v: number): void => {
  drumOmega = Math.max(-3, Math.min(3, v));
  drumSlider.value = drumOmega.toFixed(2);
  drumVal.textContent = drumOmega.toFixed(2);
};
drumSlider.addEventListener("input", () => setDrum(parseFloat(drumSlider.value)));

const reset = (): void => {
  body.reset();
  body.applyTo(world.flyRoot, world.flyBody);
  optic?.reset();
};

window.addEventListener("keydown", (e) => {
  switch (e.key) {
    case "[":
      setDrum(drumOmega - 0.25);
      break;
    case "]":
      setDrum(drumOmega + 0.25);
      break;
    case " ":
      setDrum(0);
      e.preventDefault();
      break;
    case "b":
      brainOn = !brainOn;
      brainToggle.checked = brainOn;
      reset();
      break;
    case "r":
      reset();
      break;
    case "v":
      hud.view = hud.view === "luminance" ? "brain" : "luminance";
      break;
    case "l":
      // a world-fixed object aimed at where the fly is now, so turning and sideslipping away works
      if (loomer.active) loomer.stop();
      else loomer.launch(body.state.position, body.state.yaw, { retinal: false, loop: false, speed: 2, startDistance: 12 });
      break;
  }
});

/**
 * Open-loop side calibration: spin the drum each way with the output off, take
 * each side's preferred-direction response, and hand the gains to the brain.
 * Runs once after the brain has settled; `fly.calibrateSides()` re-runs it.
 */
let sidesCalibrated = false;
const calibrateSides = async (): Promise<{ L: number; R: number } | undefined> => {
  const o = optic;
  if (!o) return undefined;
  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
  while (!o.calibrated) await sleep(100);
  const outGain = o.params.outputGain;
  const loomGain = o.params.loomGain;
  o.params.outputGain = 0;
  o.params.loomGain = 0;
  const drum0 = drumOmega;
  const measure = async (omega: number): Promise<{ L: number; R: number }> => {
    setDrum(omega);
    await sleep(1200);
    const acc = { L: 0, R: 0 };
    let n = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < 1000) {
      const s = o.readoutSides();
      acc.L += s.L;
      acc.R += s.R;
      n++;
      await sleep(25);
    }
    return { L: acc.L / n, R: acc.R / n };
  };
  const rest = await measure(0);
  const ccw = await measure(1);
  const cw = await measure(-1);
  setDrum(drum0);
  const pdL = Math.max(ccw.L - rest.L, cw.L - rest.L);
  const pdR = Math.max(ccw.R - rest.R, cw.R - rest.R);
  await sleep(1000);
  o.setSideGain(pdL, pdR);
  o.params.outputGain = outGain;
  o.params.loomGain = loomGain;
  sidesCalibrated = true;
  return o.sideGain;
};

// ---------- recorder ----------
/**
 * Records what the eyes see during scripted self-motion, for training on the scene's own
 * statistics (train/train_loom.py --scene). Each episode drives the body directly with the
 * brain off: a yaw rate, a forward speed, or nothing (static). Luminance per column, 8 bit.
 */
interface Episode {
  kind: string;
  seconds: number;
  omega?: number;
  speed?: number;
  start?: { x: number; z: number; yaw: number };
}
const RECORD_HZ = 120;
let recorderHook: ((dt: number) => void) | undefined;
const record = async (episodes: Episode[]): Promise<Array<Record<string, unknown>>> => {
  if (!eyes || !opticGraph) throw new Error("graph not loaded");
  const C = opticGraph[0].columns.count;
  const { ommL, ommR } = eyes;
  const wasOn = brainOn;
  brainOn = false;
  setDrum(0);
  loomer.stop();
  world.course.visible = true;
  const out: Array<Record<string, unknown>> = [];
  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
  for (const ep of episodes) {
    body.reset();
    if (ep.start) {
      body.state.position.x = ep.start.x;
      body.state.position.z = ep.start.z;
      body.state.yaw = ep.start.yaw;
    }
    body.applyTo(world.flyRoot, world.flyBody);
    const frames: Uint8Array[] = [];
    recorderHook = () => {
      if (!eyes) return;
      const buf = new Uint8Array(C);
      for (let i = 0; i < ommL.count; i++) buf[ommL.col[i]!] = Math.round(255 * Math.max(0, Math.min(1, eyes.lumL[i]!)));
      for (let i = 0; i < ommR.count; i++) buf[ommR.col[i]!] = Math.round(255 * Math.max(0, Math.min(1, eyes.lumR[i]!)));
      frames.push(buf);
      body.state.yawRate = ep.omega ?? 0;
      body.state.speed = ep.speed ?? 0;
      body.state.sideSpeed = 0;
    };
    // fixed 120 Hz frames, independent of the real frame rate; yield now and then so the page stays alive
    const n = Math.round(ep.seconds * RECORD_HZ);
    loop.stop();
    for (let k = 0; k < n; k++) {
      loop.step(1 / RECORD_HZ);
      if (k % 30 === 29) await sleep(0);
    }
    loop.start();
    recorderHook = undefined;
    const all = new Uint8Array(frames.length * C);
    frames.forEach((f, k) => all.set(f, k * C));
    let bin = "";
    for (let k = 0; k < all.length; k += 8192) bin += String.fromCharCode(...all.subarray(k, k + 8192));
    out.push({ kind: ep.kind, omega: ep.omega ?? 0, speed: ep.speed ?? 0, columns: C, frames: frames.length, dt: 1 / RECORD_HZ, data: btoa(bin) });
  }
  world.course.visible = false;
  brainOn = wasOn;
  reset();
  return out;
};

// ---------- loop ----------
const fwd = new THREE.Vector3();
const camTarget = new THREE.Vector3();
const camPos = new THREE.Vector3();
const PHYS_DT = 1 / 1000;
let cmd: MotorCommand = hover;
let wingPhase = 0;
let dirL = new Float32Array(0);
let dirR = new Float32Array(0);

const loop = new SimLoop({
  frame(dt, time) {
    // 1. world: spin the drum, keep it centred on the fly
    world.drum.rotation.y += drumOmega * dt;
    world.drum.position.x = body.state.position.x;
    world.drum.position.z = body.state.position.z;
    loomer.update(dt, body.state.position, body.state.yaw);
    countCollisions();

    // 2. eye: render once, sample both eyes
    world.flyRoot.updateMatrixWorld(true);
    eye.render();
    if (eyes) {
      eye.sample(eyes.ommL, eyes.lumL);
      eye.sample(eyes.ommR, eyes.lumR);
    }

    recorderHook?.(dt);

    // 3. brain
    const brain = brainOn ? optic : undefined;
    cmd = brain && eyes ? brain.step({ lumLeft: eyes.lumL, lumRight: eyes.lumR }, dt) : hover;

    // 4. motor + body, fixed 1 ms substeps
    const forces = wingsToForces(cmd);
    let remaining = dt;
    while (remaining > 0) {
      const h = Math.min(PHYS_DT, remaining);
      body.step(forces, h);
      remaining -= h;
    }
    body.applyTo(world.flyRoot, world.flyBody);

    wingPhase += dt * 40;
    const flap = Math.sin(wingPhase);
    world.wings.left.rotation.y = -flap * cmd.left * 0.9;
    world.wings.right.rotation.y = flap * cmd.right * 0.9;

    body.forward(fwd);
    if (followToggle.checked) {
      camPos.copy(body.state.position).addScaledVector(fwd, -2.2);
      camPos.y += 1.0;
      camera.position.lerp(camPos, Math.min(1, 6 * dt));
      camTarget.copy(body.state.position).addScaledVector(fwd, 2);
      camera.lookAt(camTarget);
    } else {
      camera.position.set(body.state.position.x + 6, 6, body.state.position.z + 6);
      camera.lookAt(body.state.position);
    }
    renderer.render(world.scene, camera);

    // hud
    if (eyes) {
      if (optic) {
        if (dirL.length !== eyes.ommL.count) dirL = new Float32Array(eyes.ommL.count);
        if (dirR.length !== eyes.ommR.count) dirR = new Float32Array(eyes.ommR.count);
        optic.directionMap("L", dirL);
        optic.directionMap("R", dirR);
        hud.drawEyes(eyes.ommL, eyes.lumL, dirL, eyes.ommR, eyes.lumR, dirR);
        hud.drawNet(netCanvas, optic.net.r, Object.entries(optic.groups));
      } else {
        hud.drawEyes(eyes.ommL, eyes.lumL, undefined, eyes.ommR, eyes.lumR, undefined);
      }
    }
    const t = brain?.telemetry() ?? {};
    const extra: Array<[string, string | number]> = Object.entries(t)
      .filter(([k]) => !["turn"].includes(k))
      .map(([k, v]) => [k, v]);
    const status = optic ? (brainOn ? optic.name : `off (hover); ${optic.name}`) : loadStatus;
    hud.setStats([
      ["fps", loop.fps],
      ["t", time],
      ["brain", status],
      ["drum ω", drumOmega],
      ["loom", loomer.active ? `${loomer.distance.toFixed(1)} away, hits ${loomer.hits}` : "off"],
      ["collisions", collisions],
      ["yaw rate", body.state.yawRate],
      ["slip", body.state.yawRate - drumOmega],
      ["heading°", ((body.state.yaw * 180) / Math.PI) % 360],
      ["turn", t["turn"] ?? 0],
      ["wing L/R", `${cmd.left.toFixed(2)} / ${cmd.right.toFixed(2)}`],
      ...extra,
    ]);
  },
});

loop.start();

Object.assign(window, {
  fly: {
    body,
    get brain() {
      return optic;
    },
    get brainOn() {
      return brainOn;
    },
    setBrain(on: boolean) {
      brainOn = on;
      brainToggle.checked = on;
      reset();
    },
    get eyes() {
      return eyes;
    },
    eye,
    setDrum,
    loomer,
    get collisions() {
      return collisions;
    },
    resetCollisions() {
      collisions = 0;
    },
    calibrateSides,
    get sidesCalibrated() {
      return sidesCalibrated;
    },
    /** Cruise through the pillar field: mean wing amplitude above hover. 0 stops. */
    cruise(baseAmp = 0.8) {
      if (optic) optic.params.baseAmp = baseAmp > 0 ? baseAmp : 0.5;
      world.course.visible = baseAmp > 0;
    },
    /** Launch an approach from azimuth azDeg (positive = right); opts override LoomParams. */
    loom(azDeg = 45, opts: Record<string, unknown> = {}) {
      loomer.launch(body.state.position, body.state.yaw, { az: (azDeg * Math.PI) / 180, ...opts });
    },
    reset,
    loop,
    record,
  },
});
