# Neuroglancer

Local notes from the [PyTorch Connectomics Neuroglancer tutorial](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html) (Google Connectomics Team). Official code: [github.com/google/neuroglancer](https://github.com/google/neuroglancer).

Neuroglancer is a **TypeScript / WebGL** viewer for petabyte-scale neuroscience volumes. It shows:

- EM image stacks (arbitrary, not only axis-aligned, cross-sections)
- Segmentation (each neuron is a colored segment + 3D mesh)
- Skeletons (centerline graphs)
- Point annotations (synapses, nuclei, landmarks)

That is why the Male CNS paper points at it: the raw data is too large for Fiji / napari-style desktop viewers. The MaleCNS v1.0 scene is already hosted:

[neuroglancer-demo.appspot.com — male-cns-v1.0](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso)

Chrome is the supported browser. Press **H** for the command list.

## Viewer basics (public data, no install)

Top-left HUD:

- Center voxel `(x, y, z)` and resolution in parentheses (MaleCNS / hemibrain are **8 nm isotropic**)
- Orange numbers = cursor voxel

Layout: three orthogonal EM slices + a 3D mesh pane. They stay locked; rotating or panning one updates the others.

Layers (tabs, upper left):

- **Image** — raw electron micrographs
- **Segmentation** — proofread neuron IDs; double-click a cell to isolate its mesh
- Optional mesh / annotation / synapse layers

Useful keys / gestures:

| Action | Control |
| --- | --- |
| Scroll through slices | mouse wheel |
| Zoom | ctrl + wheel |
| Pan | left-drag |
| Select a neuron | double-click |
| Reset view | `Z` |
| Help | `H` |
| Layer properties | right-click the layer tab |

`+` in the upper left adds a data source (`precomputed://`, `gs://`, `vtk://`, local, etc.). After the source loads, pick a layer type (image, segmentation, mesh, annotation). Supported sources are listed in the [Neuroglancer README](https://github.com/google/neuroglancer#supported-data-sources).

## Python API (local server)

The `neuroglancer` pip package starts a local web server that serves the client **and** can stream `LocalVolume` arrays or proxy public `precomputed://` buckets. Keep the process alive (`python -i script.py` or a notebook). Anyone with the printed URL can see the data and any credentials the process holds.

```python
import neuroglancer

neuroglancer.set_server_bind_address(bind_address="localhost", bind_port=9999)
viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    s.layers["image"] = neuroglancer.ImageLayer(
        source="precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg/"
    )
    s.layers["segmentation"] = neuroglancer.SegmentationLayer(
        source="precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation",
        selected_alpha=0.3,
    )

print(viewer)
```

The tutorial’s public example is the older **hemibrain**, not MaleCNS. Same pattern; swap sources for:

- EM: `precomputed://gs://flyem-male-cns/em/em-clahe-jpeg`
- Segmentation: `precomputed://gs://flyem-male-cns/v1.0/segmentation`

Local TIFF / HDF5 volumes become `neuroglancer.LocalVolume` with an explicit `CoordinateSpace` (`z,y,x` names, nm units, voxel scales). Segmentation masks must be `volume_type="segmentation"` or meshes will not appear.

## Hooks this project can use

The Python viewer is a state machine we can script:

- `viewer.state.layers["segmentation"].segments` — currently selected neuron IDs
- `viewer.actions.add(...)` + key / mouse bindings — custom picks (e.g. “add this cell to the flight subgraph”)
- `ActionState.mouse_voxel_coordinates` — click-to-annotate
- `LocalAnnotationLayer` / `PointAnnotation` — drop synapses or landmarks
- `LocalVolume.invalidate()` — re-render after we change a volume
- GLSL shaders on image layers — overlay activity, polarity, or simulated firing rates

For the TypeScript webapp we do **not** need to embed the full EM viewer on day one. Neuroglancer stays the place to inspect morphology. Our app consumes the **graph** (neuPrint / feather tables): cell types, edges, weights. Later we can deep-link out to a Neuroglancer scene with a selected body ID, or embed the official client.

## Install (only if we script the viewer)

```bash
pip install neuroglancer imageio h5py cloud-volume
# optional: jupyter
```

Building from source clones [github.com/google/neuroglancer](https://github.com/google/neuroglancer) and needs Node via nvm, then `python setup.py install`.
