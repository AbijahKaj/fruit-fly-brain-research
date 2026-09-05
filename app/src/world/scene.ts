/**
 * Layer 1 (world): the arena.
 *
 * A striped drum (the classic optomotor stimulus) that re-centres on the
 * fly every frame so translation never reaches the wall, a checkerboard
 * ground for translational flow, and a few boxes for later milestones.
 */
import * as THREE from "three";

export const FLY_LAYER = 1;

export interface World {
  scene: THREE.Scene;
  drum: THREE.Mesh;
  /** Pillars, for collision counting. */
  obstacles: THREE.Mesh[];
  /** A denser pillar field for the obstacle course, hidden until `course.visible` is set. */
  course: THREE.Group;
  flyRoot: THREE.Object3D;
  wings: { left: THREE.Mesh; right: THREE.Mesh };
}

function stripeTexture(stripes: number, width = 1024): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = 8;
  const ctx = c.getContext("2d")!;
  const w = width / stripes;
  for (let i = 0; i < stripes; i++) {
    ctx.fillStyle = i % 2 ? "#f2f2f2" : "#151515";
    ctx.fillRect(i * w, 0, w, c.height);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function checkerTexture(cells = 8, size = 256): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  const s = size / cells;
  for (let y = 0; y < cells; y++)
    for (let x = 0; x < cells; x++) {
      ctx.fillStyle = (x + y) % 2 ? "#3a3f46" : "#22262b";
      ctx.fillRect(x * s, y * s, s, s);
    }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(40, 40);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function buildWorld(): World {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#6b7480");

  scene.add(new THREE.HemisphereLight("#cfd6df", "#2a2e33", 1.2));
  const sun = new THREE.DirectionalLight("#ffffff", 1.0);
  sun.position.set(5, 10, 3);
  scene.add(sun);

  // Ground: coarse 50-unit checker. A fine texture gives strong ground flow that the (unfitted)
  // HS cells answer on both sides regardless of direction, which swamps the optomotor readout.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(400, 400),
    new THREE.MeshStandardMaterial({ map: checkerTexture() }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const drum = new THREE.Mesh(
    new THREE.CylinderGeometry(40, 40, 60, 64, 1, true),
    new THREE.MeshBasicMaterial({ map: stripeTexture(24), side: THREE.BackSide }),
  );
  drum.position.y = 20;
  scene.add(drum);

  const boxMat = new THREE.MeshStandardMaterial({ color: "#b0473c" });
  const rng = mulberry32(7);
  const obstacles: THREE.Mesh[] = [];
  for (let i = 0; i < 6; i++) {
    const r = 8 + rng() * 14;
    const a = rng() * Math.PI * 2;
    const h = 1 + rng() * 4;
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.5, h, 1.5), boxMat);
    box.position.set(Math.cos(a) * r, h / 2, Math.sin(a) * r);
    scene.add(box);
    obstacles.push(box);
  }

  // Obstacle course: 40 more pillars over a 70 x 70 area, none within 4 of the origin.
  const course = new THREE.Group();
  course.visible = false;
  const rng2 = mulberry32(11);
  while (course.children.length < 40) {
    const x = (rng2() - 0.5) * 70;
    const z = (rng2() - 0.5) * 70;
    if (Math.hypot(x, z) < 4) continue;
    const h = 1.5 + rng2() * 4;
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.5, h, 1.5), boxMat);
    box.position.set(x, h / 2, z);
    course.add(box);
  }
  scene.add(course);

  // The fly. Body on FLY_LAYER so its own eyes cannot see it.
  const flyRoot = new THREE.Object3D();
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshStandardMaterial({ color: "#3b2f27", roughness: 0.6 }),
  );
  body.scale.set(0.12, 0.1, 0.3);
  body.layers.set(FLY_LAYER);
  flyRoot.add(body);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.09, 12, 10),
    new THREE.MeshStandardMaterial({ color: "#8c2b2b", roughness: 0.4 }),
  );
  head.position.set(0, 0.02, -0.32);
  head.layers.set(FLY_LAYER);
  flyRoot.add(head);

  const wingMat = new THREE.MeshBasicMaterial({
    color: "#dfe8f0",
    transparent: true,
    opacity: 0.45,
    side: THREE.DoubleSide,
  });
  const wingGeom = new THREE.PlaneGeometry(0.45, 0.16);
  wingGeom.translate(0.225, 0, 0);
  const left = new THREE.Mesh(wingGeom, wingMat);
  left.rotation.x = -Math.PI / 2;
  left.rotation.z = Math.PI; // extend to -x
  left.position.set(-0.05, 0.08, 0);
  left.layers.set(FLY_LAYER);
  const right = new THREE.Mesh(wingGeom, wingMat);
  right.rotation.x = -Math.PI / 2;
  right.position.set(0.05, 0.08, 0);
  right.layers.set(FLY_LAYER);
  flyRoot.add(left, right);
  scene.add(flyRoot);

  return { scene, drum, obstacles, course, flyRoot, wings: { left, right } };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
