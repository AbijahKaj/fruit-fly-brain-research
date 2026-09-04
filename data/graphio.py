"""Binary graph writer: <name>.json header + <name>.bin typed arrays (8-byte aligned)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DTYPES = {"int8": np.int8, "int16": np.int16, "int32": np.int32, "float32": np.float32, "float64": np.float64}


def write_graph(path_stem: Path, meta: dict, arrays: dict[str, np.ndarray]) -> None:
    header = dict(meta)
    header["arrays"] = {}
    blob = bytearray()
    for name, arr in arrays.items():
        dt = next(k for k, v in DTYPES.items() if arr.dtype == v)
        while len(blob) % 8:
            blob.append(0)
        header["arrays"][name] = {"dtype": dt, "offset": len(blob), "length": int(arr.size)}
        blob += np.ascontiguousarray(arr).tobytes()
    path_stem.with_suffix(".json").write_text(json.dumps(header, separators=(",", ":")))
    path_stem.with_suffix(".bin").write_bytes(bytes(blob))
