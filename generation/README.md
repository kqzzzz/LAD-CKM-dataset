# LAD-CKM Dataset Generation

Code for generating the LAD-CKM dynamic MIMO-OFDM channel dataset with [Sionna RT](https://github.com/NVlabs/sionna-rt).

## Folder Structure

```
generation/
├── generate.py            # entry point
├── configs/default.yaml   # all generation parameters
├── ladckm_gen/
│   ├── scene.py           # scene loading, antenna arrays, UE dropping
│   ├── scatterers.py      # placement of pedestrians and vehicles
│   ├── csi.py             # ray tracing and uplink/downlink CFR
│   └── utils.py           # logging and random seeds
└── scene/
    ├── scene.xml          # Mitsuba scene used by Sionna RT
    ├── meshes/            # building / ground / pedestrian / vehicle meshes (.ply)
    ├── scene.blend        # Blender project (materials assigned), for editing the scene
    └── scene.osm          # raw OpenStreetMap data of the area
```

## Installation

An NVIDIA GPU is recommended. The dataset was generated on Linux.

```bash
cd generation
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 11.8 build of PyTorch. If your GPU or driver needs another CUDA version, install the matching PyTorch wheel first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

## Quick Start

```bash
cd generation
python generate.py
```

This generates 20 frames (coherence blocks) with 2,500 UEs each for a 4×4 BS array and 2×2 UE arrays, and saves:

- `generated/Tx4x4_Rx2x2.npy`: the dataset
- `generated/Tx4x4_Rx2x2.yaml`: the full configuration used
- `logs/<timestamp>.log`: the log

To use a specific GPU, set `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=1 python generate.py`.

## Configuration

All parameters are in [`configs/default.yaml`](configs/default.yaml). Any entry can be overridden on the command line with `key=value`:

```bash
# BS with an 8x8 UPA (Nt = 64), single-antenna UEs (Nr = 1)
python generate.py tx.num_rows=8 tx.num_cols=8 rx.num_rows=1 rx.num_cols=1

# a small run for testing
python generate.py dataset.num_frame=2 dataset.num_rx=100 ray_tracing.samples_per_src=100000 ray_tracing.max_num_paths_per_src=100000

# your own config file
python generate.py --config configs/my_config.yaml
```

Main parameters:

| Key | Default | Description |
|---|---|---|
| `tx.num_rows`, `tx.num_cols` | 4, 4 | BS UPA size, $N_t$ = rows × cols |
| `rx.num_rows`, `rx.num_cols` | 2, 2 | UE UPA size, $N_r$ = rows × cols |
| `tx.position` | [35, 10, 20] | BS position (m) |
| `ray_tracing.fc_up`, `fc_down` | 6.715 GHz, 6.765 GHz | uplink / downlink carrier frequencies |
| `ray_tracing.max_depth` | 5 | maximum number of interactions per path |
| `ray_tracing.samples_per_src` | 1e6 | number of rays shot per source |
| `dataset.num_frame` | 20 | number of frames (coherence blocks) |
| `dataset.num_rx` | 2500 | number of UEs per frame |
| `dataset.num_sc`, `sc_spacing` | 52, 312.5 kHz | OFDM subcarriers |
| `scene.pedestrian.num_range` | [500, 2500] | number of pedestrians per frame |
| `scene.vehicle.num_range` | [20, 70] | number of vehicles per frame |
| `seed` | 0 | random seed |

## How It Works

For every frame, `generate.py`:

1. draws a random scene temperature;
2. drops `num_rx` UEs uniformly at random in `rx.dist_range`;
3. re-draws the dynamic scatterers: a random number of pedestrians on the sidewalks and open areas, and a random number of vehicles on the driving lanes;
4. traces the propagation paths with Sionna RT's `PathSolver` at the uplink and downlink carrier frequencies, and computes the channel frequency response (CFR) on `num_sc` subcarriers;
5. keeps the UEs that receive at least one path in both links, drops new UEs to replace the others, and repeats step 4 until `num_rx` valid UEs are collected.

## Output Format

```python
import numpy as np

dataset = np.load("generated/Tx4x4_Rx2x2.npy", allow_pickle=True)   # num_frame dicts
frame = dataset[0]
frame["csi"]           # complex64, [num_rx, Nr, Nt, 2 * num_sc]
                       #   [..., :num_sc]  uplink CFR
                       #   [..., num_sc:]  downlink CFR
frame["rx_positions"]  # float32, [3, num_rx], UE positions (x, y, z) in meters
```

The CFR is computed with `Paths.cfr(normalize=False, normalize_delays=True)`, i.e., the path gains are not normalized. The file can be read directly with [`loader.py`](../loader.py).

## Scene

The 3D geometry of the campus is downloaded from [OpenStreetMap](https://www.openstreetmap.org/) (© OpenStreetMap contributors, [ODbL](https://opendatacommons.org/licenses/odbl/)), imported into [Blender](https://www.blender.org/) to assign ITU radio materials, and exported to `scene/scene.xml`. To edit the scene, open `scene/scene.blend` and re-export it with the [Mitsuba Blender add-on](https://github.com/mitsuba-renderer/mitsuba-blender).

The driving lanes and pedestrian areas used to place the dynamic scatterers are defined at the top of [`ladckm_gen/scatterers.py`](ladckm_gen/scatterers.py). Update them if you use another scene.
