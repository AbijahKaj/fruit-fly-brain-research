"""Eye input recorded from the browser scene (app: fly.record) as trainer stimuli.

Each episode: luminance per column, 8 bit, at 120 Hz, with the self-motion that
produced it (yaw rate, forward speed). `photoreceptor()` runs the browser's own
pipeline (optic.ts injectEye: Weber adaptation, virtual photoreceptor low-pass)
so the network is trained on exactly what it gets at run time."""
from __future__ import annotations
import base64, json, math
from pathlib import Path
import numpy as np
import torch


class SceneEpisodes:
    def __init__(self, path: str | Path, dev: str, dt: float, fv: dict, adapt_tau: float = 1.0, stim_max: float = 1.5):
        d = json.load(open(path))
        self.dev, self.dt = dev, dt
        pr = fv["photoreceptor"]
        self.kinds: list[str] = []
        self.omega: list[float] = []
        self.speed: list[float] = []
        self.rates: list[torch.Tensor] = []          # photoreceptor rate (T, C) at `dt`
        for ep in d["episodes"]:
            raw = np.frombuffer(base64.b64decode(ep["data"]), dtype=np.uint8).reshape(ep["frames"], ep["columns"]) / 255.0
            # resample the recorded frames to the trainer's dt
            t_rec = np.arange(ep["frames"]) * ep["dt"]
            t_new = np.arange(0, t_rec[-1], dt)
            lum = np.empty((len(t_new), raw.shape[1]), np.float32)
            for c in range(raw.shape[1]):
                lum[:, c] = np.interp(t_new, t_rec, raw[:, c])
            lum_t = torch.tensor(lum, device=dev)
            self.rates.append(self.photoreceptor(lum_t, pr, adapt_tau, stim_max))
            self.kinds.append(ep["kind"])
            self.omega.append(float(ep["omega"]))
            self.speed.append(float(ep["speed"]))
        self.by_kind = {k: [i for i, kk in enumerate(self.kinds) if kk == k] for k in set(self.kinds)}

    def photoreceptor(self, lum: torch.Tensor, pr: dict, adapt_tau: float, stim_max: float) -> torch.Tensor:
        """optic.ts injectEye, vectorised over columns."""
        T, C = lum.shape
        alpha = min(1.0, self.dt / pr["tau"])
        a_adapt = min(1.0, self.dt / adapt_tau)
        mean = lum[0].clone()
        v = torch.zeros(C, device=lum.device)
        out = torch.empty_like(lum)
        for t in range(T):
            if t > 0:
                mean = mean + a_adapt * (lum[t] - mean)
            stim = torch.clamp(0.5 * lum[t] / (mean + 1e-3), max=stim_max)
            target = pr["restOffset"] + pr["stimGain"] * stim
            v = v + alpha * (target - v)
            out[t] = torch.relu(v)
        return out

    def crop(self, kind_sel, T: int, t_min: float = 0.5) -> tuple[torch.Tensor, int]:
        """A random T-step window from a random episode matching kind_sel (a kind name or a
        predicate on (kind, omega, speed)). Returns (rates (T, C), episode index)."""
        if callable(kind_sel):
            idx = [i for i in range(len(self.kinds)) if kind_sel(self.kinds[i], self.omega[i], self.speed[i])]
        else:
            idx = self.by_kind[kind_sel]
        i = idx[int(torch.randint(len(idx), ()).item())]
        r = self.rates[i]
        lo = int(t_min / self.dt)
        hi = r.shape[0] - T
        s = lo if hi <= lo else lo + int(torch.randint(hi - lo + 1, ()).item())
        return r[s: s + T], i

    def batch(self, kinds: list[str], T: int) -> torch.Tensor:
        """Photoreceptor rates (T, B, C) for the HS stage's kinds: static, rotCCW, rotCW,
        transFwd, transAsym (the last two are both recorded cruises, different episodes)."""
        sel = {
            "static": "static",
            "rotCCW": lambda k, w, v: k.startswith("rot") and w > 0,
            "rotCW": lambda k, w, v: k.startswith("rot") and w < 0,
            "transFwd": "trans",
            "transAsym": "trans",
        }
        out = torch.stack([self.crop(sel[k], T)[0] for k in kinds], dim=1)
        return out
