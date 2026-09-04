/**
 * Milestone 3: the closed loop with a choice of brains.
 *
 *   world/body  ->  eye  ->  brain (stub | connectome | optic)  ->  motor  ->  world/body
 *
 *   stub        milestone 1, hand-written correlator + P controller
 *   connectome  milestone 2, correlator -> MaleCNS flight graph (1k units)
 *   optic       milestone 3, per-column MaleCNS optic lobe + flight graph (40k units, Web Worker)
 */
import * as THREE from "three";
import { buildWorld, FLY_LAYER } from "./world/scene";
import { FlyBody } from "./world/fly";
import { buildOmmatidia, ommatidiaFromColumns, type Ommatidia } from "./eye/ommatidia";
import { CompoundEye } from "./eye/eye";
import { Photoreceptors } from "./eye/photoreceptor";
import { StubBrain } from "./brain/stub";
import { ConnectomeBrain } from "./brain/connectome";
import { OpticBrain } from "./brain/optic";
import { loadGraph, unitsWhere } from "./brain/graph";
import { loadFlyvis } from "./brain/flyvis";
import type { Brain, Lattice, MotorCommand } from "./brain/types";
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
body.applyTo(world.flyRoot);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 500);
camera.layers.enable(FLY_LAYER);
window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// ---------- eye: one lattice per brain family ----------
interface EyeSet {
  ommL: Ommatidia;
  ommR: Ommatidia;
  lumL: Float32Array;
  lumR: Float32Array;
  prL: Photoreceptors;
  prR: Photoreceptors;
}
const eye = new CompoundEye(renderer, world.scene, world.flyRoot, 48);
const makeEyeSet = (ommL: Ommatidia, ommR: Ommatidia): EyeSet => {
  eye.register(ommL);
  eye.register(ommR);
  return {
    ommL,
    ommR,
    lumL: new Float32Array(ommL.count),
    lumR: new Float32Array(ommR.count),
    prL: new Photoreceptors(ommL.count),
    prR: new Photoreceptors(ommR.count),
  };
};
const eyes: Partial<Record<Lattice, EyeSet>> = {
  fibonacci: makeEyeSet(buildOmmatidia("left"), buildOmmatidia("right")),
};

// ---------- brains ----------
type BrainKey = "off" | "stub" | "connectome" | "optic";
const fib = eyes.fibonacci!;
const brains: Partial<Record<BrainKey, Brain>> = { stub: new StubBrain(fib.ommL, fib.ommR) };
let brainKey: BrainKey = "optic";
let connectome: ConnectomeBrain | undefined;
let optic: OpticBrain | undefined;
let flightGroups: Array<[string, Int32Array]> = [];
const hover: MotorCommand = { left: defaultWingParams.hoverAmp, right: defaultWingParams.hoverAmp };
const loadStatus: Record<string, string> = {};

loadGraph(`${BASE}graphs/flight-v1.json`)
  .then((g) => {
    connectome = new ConnectomeBrain(g, fib.ommL, fib.ommR);
    brains.connectome = connectome;
    flightGroups = [
      ["LPTC", unitsWhere(g, (t) => /^(HS[ENST]|VS|VST1|VST2|VSm|H2|DCH|VCH)$/.test(t))],
      ["LC", unitsWhere(g, (t) => /^(LPLC2|LC4)$/.test(t))],
      ["brain", unitsWhere(g, (_t, _s, r) => r === "brain")],
      ["DNg02", unitsWhere(g, (t) => /^DNg02_/.test(t))],
      ["DNp", unitsWhere(g, (t) => /^DNp0/.test(t))],
      ["vnc", unitsWhere(g, (_t, _s, r) => r === "vnc")],
      ["MN", unitsWhere(g, (_t, _s, r) => r === "output")],
    ];
    syncBrainControls();
  })
  .catch((err) => {
    loadStatus["connectome"] = String(err);
    console.error(err);
  });

// Fitted parameters (train/train_optic.py) win over the raw flyvis transfer when present.
const loadParams = (): ReturnType<typeof loadFlyvis> =>
  loadFlyvis(`${BASE}graphs/fitted-params.json`).catch(() => loadFlyvis(`${BASE}graphs/flyvis-params.json`));
Promise.all([loadGraph(`${BASE}graphs/optic-v2`), loadParams()])
  .then(([g, fv]) => {
    const cols = makeEyeSet(ommatidiaFromColumns("left", g.columns), ommatidiaFromColumns("right", g.columns));
    eyes.columns = cols;
    optic = new OpticBrain(g, fv, cols.ommL, cols.ommR);
    brains.optic = optic;
    syncBrainControls();
  })
  .catch((err) => {
    loadStatus["optic"] = String(err);
    console.error(err);
  });

const currentBrain = (): Brain | undefined => (brainKey === "off" ? undefined : brains[brainKey]);

// ---------- ui ----------
const hud = new Hud($<HTMLCanvasElement>("eyes"), $("stats"));
const netCanvas = $<HTMLCanvasElement>("net");
const drumSlider = $<HTMLInputElement>("drum");
const drumVal = $("drumVal");
const brainSel = $<HTMLSelectElement>("brainSel");
const followToggle = $<HTMLInputElement>("follow");
const wScaleSlider = $<HTMLInputElement>("wScale");
const outGainSlider = $<HTMLInputElement>("outGain");
const flipReadout = $<HTMLInputElement>("flipReadout");
const readoutSel = $<HTMLSelectElement>("readout");

const syncBrainControls = (): void => {
  const gain = parseFloat(wScaleSlider.value);
  const outputGain = parseFloat(outGainSlider.value);
  const readoutSign = flipReadout.checked ? 1 : -1;
  $("wScaleVal").textContent = gain.toFixed(2);
  $("outGainVal").textContent = outputGain.toFixed(2);
  if (connectome) {
    Object.assign(connectome.params, { wScale: 0.03 * gain, outputGain, readoutSign });
    const ro = readoutSel.value as "dng02" | "mn";
    if (ro !== connectome.params.readout) {
      connectome.params.readout = ro;
      connectome.reset();
    }
  }
  if (optic) Object.assign(optic.params, { wScale: gain, outputGain, readoutSign });
};
for (const el of [wScaleSlider, outGainSlider, flipReadout, readoutSel]) el.addEventListener("input", syncBrainControls);
brainSel.addEventListener("change", () => {
  brainKey = brainSel.value as BrainKey;
  reset();
});
brainSel.value = brainKey;

let drumOmega = 0; // rad/s about +Y; positive = counterclockwise seen from above
const setDrum = (v: number): void => {
  drumOmega = Math.max(-3, Math.min(3, v));
  drumSlider.value = drumOmega.toFixed(2);
  drumVal.textContent = drumOmega.toFixed(2);
};
drumSlider.addEventListener("input", () => setDrum(parseFloat(drumSlider.value)));

const reset = (): void => {
  body.reset();
  body.applyTo(world.flyRoot);
  for (const b of Object.values(brains)) b.reset();
  for (const e of Object.values(eyes)) {
    e.prL.reset();
    e.prR.reset();
  }
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
    case "b": {
      const order: BrainKey[] = ["off", "stub", "connectome", "optic"];
      brainKey = order[(order.indexOf(brainKey) + 1) % order.length]!;
      brainSel.value = brainKey;
      reset();
      break;
    }
    case "r":
      reset();
      break;
    case "v":
      hud.view = hud.view === "luminance" ? "highpass" : hud.view === "highpass" ? "brain" : "luminance";
      break;
  }
});

// ---------- loop ----------
const fwd = new THREE.Vector3();
const camTarget = new THREE.Vector3();
const camPos = new THREE.Vector3();
const PHYS_DT = 1 / 1000;
let cmd: MotorCommand = hover;
let wingPhase = 0;
const dirMapL = new Float32Array(0);
let dirL = dirMapL;
let dirR = dirMapL;

const loop = new SimLoop({
  frame(dt, time) {
    // 1. world: spin the drum, keep it centred on the fly
    world.drum.rotation.y += drumOmega * dt;
    world.drum.position.x = body.state.position.x;
    world.drum.position.z = body.state.position.z;

    // 2. eye: render once, sample every registered lattice
    world.flyRoot.updateMatrixWorld(true);
    eye.render();
    for (const e of Object.values(eyes)) {
      eye.sample(e.ommL, e.lumL);
      eye.sample(e.ommR, e.lumR);
      e.prL.update(e.lumL, dt);
      e.prR.update(e.lumR, dt);
    }

    // 3. brain
    const brain = currentBrain();
    const es = brain ? eyes[brain.lattice] : undefined;
    cmd =
      brain && es
        ? brain.step({ left: es.prL.out, right: es.prR.out, lumLeft: es.lumL, lumRight: es.lumR }, dt)
        : hover;

    // 4. motor + body, fixed 1 ms substeps
    const forces = wingsToForces(cmd);
    let remaining = dt;
    while (remaining > 0) {
      const h = Math.min(PHYS_DT, remaining);
      body.step(forces, h);
      remaining -= h;
    }
    body.applyTo(world.flyRoot);

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
    const shown = es ?? fib;
    if (brain === optic && optic) {
      if (dirL.length !== shown.ommL.count) dirL = new Float32Array(shown.ommL.count);
      if (dirR.length !== shown.ommR.count) dirR = new Float32Array(shown.ommR.count);
      optic.directionMap("L", dirL);
      optic.directionMap("R", dirR);
      hud.drawEyes(shown.ommL, shown.prL.logLum, shown.prL.out, dirL, shown.ommR, shown.prR.logLum, shown.prR.out, dirR);
      hud.drawNet(netCanvas, optic.net.r, Object.entries(optic.groups));
    } else {
      hud.drawEyes(shown.ommL, shown.prL.logLum, shown.prL.out, undefined, shown.ommR, shown.prR.logLum, shown.prR.out, undefined);
      if (brain === connectome && connectome) hud.drawNet(netCanvas, connectome.net.r, flightGroups);
      else netCanvas.getContext("2d")!.clearRect(0, 0, netCanvas.width, netCanvas.height);
    }
    const t = brain?.telemetry() ?? {};
    const extra: Array<[string, string | number]> = Object.entries(t)
      .filter(([k]) => !["turn"].includes(k))
      .map(([k, v]) => [k, v]);
    const status = brain
      ? brain.name
      : brainKey === "off"
        ? "off (hover)"
        : (loadStatus[brainKey] ?? "loading graph…");
    hud.setStats([
      ["fps", loop.fps],
      ["t", time],
      ["brain", status],
      ["drum ω", drumOmega],
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
    brains,
    eyes,
    get brain() {
      return currentBrain();
    },
    setBrain(k: BrainKey) {
      brainKey = k;
      brainSel.value = k;
      reset();
    },
    eye,
    setDrum,
    reset,
    loop,
  },
});
