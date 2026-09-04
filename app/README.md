# app — milestone 1

Closed loop with a stub brain. See the root README roadmap.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + bundle
```

| Layer | Path | What |
| --- | --- | --- |
| 1 world + body | `src/world/scene.ts`, `src/world/fly.ts` | Striped drum, ground, obstacles, fly rigid body |
| 2 eye | `src/eye/ommatidia.ts`, `src/eye/eye.ts`, `src/eye/photoreceptor.ts` | ~750 ommatidia per eye, six-camera render + gather, log luminance + high-pass |
| 3 brain | `src/brain/types.ts`, `src/brain/stub.ts` | `Brain` interface; stub = Hassenstein–Reichardt correlator + P controller |
| 4 motor | `src/motor/wings.ts` | Wing amplitudes → thrust, yaw torque, bank |
| glue | `src/sim/loop.ts`, `src/ui/hud.ts`, `src/main.ts` | Frame loop, eye HUD, wiring |

Controls: `[` `]` drum speed, space stops the drum, `b` toggles the brain, `r` resets the fly, `v` switches the eye HUD between luminance and high-pass.

Done when: with the brain on, the fly's yaw rate follows the drum's angular velocity (slip → 0) at a steady frame rate. Milestone 2 swaps `StubBrain` for real MaleCNS wiring behind the same `Brain` interface.
