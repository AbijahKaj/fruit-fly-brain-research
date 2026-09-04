/** Fixed-substep simulation loop driven by requestAnimationFrame. */
export interface LoopCallbacks {
  /** Called once per animation frame with the clamped frame dt. */
  frame(dt: number, time: number): void;
}

export class SimLoop {
  running = false;
  time = 0;
  fps = 0;
  private last = 0;
  private frames = 0;
  private fpsAccum = 0;

  constructor(
    private readonly cb: LoopCallbacks,
    private readonly maxDt = 0.05,
  ) {}

  start(): void {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    requestAnimationFrame(this.tick);
  }

  stop(): void {
    this.running = false;
  }

  private readonly tick = (now: number): void => {
    if (!this.running) return;
    const real = (now - this.last) / 1000;
    this.last = now;
    const dt = Math.min(this.maxDt, Math.max(1e-4, real));
    this.time += dt;
    this.cb.frame(dt, this.time);
    this.frames++;
    this.fpsAccum += real;
    if (this.fpsAccum >= 0.5) {
      this.fps = this.frames / this.fpsAccum;
      this.frames = 0;
      this.fpsAccum = 0;
    }
    requestAnimationFrame(this.tick);
  };
}
