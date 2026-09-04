/**
 * Milestone 1: the closed loop with a stub brain.
 *
 *   world/body  ->  eye  ->  brain (stub)  ->  motor  ->  world/body
 *
 * Success criterion from the README: the stub controller holds heading
 * against drum rotation at a stable frame budget.
 */
import * as THREE from "three";
import { buildWorld, FLY_LAYER } from "./world/scene";
import { FlyBody } from "./world/fly";
import { buildOmmatidia } from "./eye/ommatidia";
import { CompoundEye } from "./eye/eye";
import { Photoreceptors } from "./eye/photoreceptor";
import { StubBrain } from "./brain/stub";
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

// ---------- brain ----------
const brain: Brain = new StubBrain(ommL, ommR);
const hover: MotorCommand = { left: defaultWingParams.hoverAmp, right: defaultWingParams.hoverAmp };

// ---------- ui ----------
const hud = new Hud($<HTMLCanvasElement>("eyes"), $("stats"), ommL, ommR);
const drumSlider = $<HTMLInputElement>("drum");
const drumVal = $("drumVal");
const brainToggle = $<HTMLInputElement>("brainOn");
const followToggle = $<HTMLInputElement>("follow");

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
  brain.reset();
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
    case "b":
      brainToggle.checked = !brainToggle.checked;
      break;
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
    cmd = brainToggle.checked ? brain.step({ left: hpL, right: hpR }, dt) : hover;

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
    const t = brain.telemetry();
    hud.setStats([
      ["fps", loop.fps],
      ["t", time],
      ["brain", brainToggle.checked ? brain.name : "off (hover)"],
      ["drum ω", drumOmega],
      ["yaw rate", body.state.yawRate],
      ["slip", body.state.yawRate - drumOmega],
      ["heading°", ((body.state.yaw * 180) / Math.PI) % 360],
      ["speed", body.state.speed],
      ["flow L", t["flowL"] ?? 0],
      ["flow R", t["flowR"] ?? 0],
      ["turn", t["turn"] ?? 0],
      ["wing L/R", `${cmd.left.toFixed(2)} / ${cmd.right.toFixed(2)}`],
    ]);
  },
});

loop.start();

// Expose for poking from the console / tests.
Object.assign(window, { fly: { body, brain, eye, ommL, ommR, setDrum, reset, loop } });
