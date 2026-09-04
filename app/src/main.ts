/**
 * Milestone 2: the closed loop with a choice of brains.
 *
 *   world/body  ->  eye  ->  brain (stub | connectome)  ->  motor  ->  world/body
 *
 * The connectome brain is real MaleCNS wiring (data/extract_flight.py)
 * behind the same Brain interface the stub uses.
 */
import * as THREE from "three";
import { buildWorld, FLY_LAYER } from "./world/scene";
import { FlyBody } from "./world/fly";
import { buildOmmatidia } from "./eye/ommatidia";
import { CompoundEye } from "./eye/eye";
import { Photoreceptors } from "./eye/photoreceptor";
import { StubBrain } from "./brain/stub";
import { ConnectomeBrain } from "./brain/connectome";
import { loadGraph, unitsWhere } from "./brain/graph";
import type { Brain, MotorCommand } from "./brain/types";
import { wingsToForces, defaultWingParams } from "./motor/wings";
import { SimLoop } from "./sim/loop";
import { Hud } from "./ui/hud";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

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

// ---------- eye ----------
const ommL = buildOmmatidia("left");
const ommR = buildOmmatidia("right");
const eye = new CompoundEye(renderer, world.scene, world.flyRoot, 48);
eye.register(ommL);
eye.register(ommR);
const lumL = new Float32Array(ommL.count);
const lumR = new Float32Array(ommR.count);
const prL = new Photoreceptors(ommL.count);
const prR = new Photoreceptors(ommR.count);

// ---------- brains ----------
type BrainKey = "off" | "stub" | "connectome";
const brains: Partial<Record<BrainKey, Brain>> = { stub: new StubBrain(ommL, ommR) };
let brainKey: BrainKey = "connectome";
let connectome: ConnectomeBrain | undefined;
let netGroups: Array<[string, Int32Array]> = [];
const hover: MotorCommand = { left: defaultWingParams.hoverAmp, right: defaultWingParams.hoverAmp };

loadGraph(`${import.meta.env.BASE_URL}graphs/flight-v1.json`)
  .then((g) => {
    connectome = new ConnectomeBrain(g, ommL, ommR);
    brains.connectome = connectome;
    netGroups = [
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
  .catch((err) => console.error(err));

const currentBrain = (): Brain | undefined => (brainKey === "off" ? undefined : brains[brainKey]);

// ---------- ui ----------
const hud = new Hud($<HTMLCanvasElement>("eyes"), $("stats"), ommL, ommR);
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
  if (!connectome) return;
  connectome.params.wScale = parseFloat(wScaleSlider.value);
  connectome.params.outputGain = parseFloat(outGainSlider.value);
  connectome.params.readoutSign = flipReadout.checked ? 1 : -1;
  const ro = readoutSel.value as "dng02" | "mn";
  if (ro !== connectome.params.readout) {
    connectome.params.readout = ro;
    connectome.reset();
  }
  $("wScaleVal").textContent = connectome.params.wScale.toFixed(4);
  $("outGainVal").textContent = connectome.params.outputGain.toFixed(2);
};
for (const el of [wScaleSlider, outGainSlider, flipReadout, readoutSel]) el.addEventListener("input", syncBrainControls);
brainSel.addEventListener("change", () => {
  brainKey = brainSel.value as BrainKey;
  reset();
});

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
  prL.reset();
  prR.reset();
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
      const order: BrainKey[] = ["off", "stub", "connectome"];
      brainKey = order[(order.indexOf(brainKey) + 1) % order.length]!;
      brainSel.value = brainKey;
      reset();
      break;
    }
    case "r":
      reset();
      break;
    case "v":
      hud.view = hud.view === "luminance" ? "highpass" : "luminance";
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

const loop = new SimLoop({
  frame(dt, time) {
    // 1. world: spin the drum, keep it centred on the fly
    world.drum.rotation.y += drumOmega * dt;
    world.drum.position.x = body.state.position.x;
    world.drum.position.z = body.state.position.z;

    // 2. eye: render from the fly's pose, sample, photoreceptors
    world.flyRoot.updateMatrixWorld(true);
    eye.render();
    eye.sample(ommL, lumL);
    eye.sample(ommR, lumR);
    const hpL = prL.update(lumL, dt);
    const hpR = prR.update(lumR, dt);

    // 3. brain
    const brain = currentBrain();
    cmd = brain ? brain.step({ left: hpL, right: hpR }, dt) : hover;

    // 4. motor + body, fixed 1 ms substeps
    const forces = wingsToForces(cmd);
    let remaining = dt;
    while (remaining > 0) {
      const h = Math.min(PHYS_DT, remaining);
      body.step(forces, h);
      remaining -= h;
    }
    body.applyTo(world.flyRoot);

    // wing flap visuals
    wingPhase += dt * 40;
    const flap = Math.sin(wingPhase);
    world.wings.left.rotation.y = -flap * cmd.left * 0.9;
    world.wings.right.rotation.y = flap * cmd.right * 0.9;

    // camera
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
    hud.drawEyes(prL.logLum, hpL, prR.logLum, hpR);
    if (brain === connectome && connectome) hud.drawNet(netCanvas, connectome.net.r, netGroups);
    else netCanvas.getContext("2d")!.clearRect(0, 0, netCanvas.width, netCanvas.height);
    const t = brain?.telemetry() ?? {};
    const extra: Array<[string, string | number]> = Object.entries(t)
      .filter(([k]) => !["flowL", "flowR", "turn"].includes(k))
      .map(([k, v]) => [k, v]);
    hud.setStats([
      ["fps", loop.fps],
      ["t", time],
      ["brain", brain ? brain.name : brainKey === "connectome" ? "loading graph…" : "off (hover)"],
      ["drum ω", drumOmega],
      ["yaw rate", body.state.yawRate],
      ["slip", body.state.yawRate - drumOmega],
      ["heading°", ((body.state.yaw * 180) / Math.PI) % 360],
      ["speed", body.state.speed],
      ["flow L", t["flowL"] ?? 0],
      ["flow R", t["flowR"] ?? 0],
      ["turn", t["turn"] ?? 0],
      ["wing L/R", `${cmd.left.toFixed(2)} / ${cmd.right.toFixed(2)}`],
      ...extra,
    ]);
  },
});

loop.start();

// Expose for poking from the console / tests.
Object.assign(window, {
  fly: {
    body,
    brains,
    get brain() {
      return currentBrain();
    },
    setBrain(k: BrainKey) {
      brainKey = k;
      brainSel.value = k;
      reset();
    },
    eye,
    ommL,
    ommR,
    setDrum,
    reset,
    loop,
  },
});
