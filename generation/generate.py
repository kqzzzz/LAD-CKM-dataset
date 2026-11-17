"""Generate the LAD-CKM dynamic MIMO-OFDM channel dataset with Sionna RT.

For every frame (coherence block):
    1. draw a random scene temperature,
    2. drop ``dataset.num_rx`` UEs uniformly at random,
    3. re-draw the dynamic scatterers (pedestrians and vehicles),
    4. ray-trace the uplink (``fc_up``) and downlink (``fc_down``) channels and
       compute the CFR on ``dataset.num_sc`` subcarriers,
    5. keep the UEs with valid CSI in both links; re-drop new UEs for the
       missing ones until ``dataset.num_rx`` valid UEs are collected.

Usage:
    python generate.py                                   # configs/default.yaml
    python generate.py --config configs/default.yaml tx.num_rows=4 tx.num_cols=8

Any config entry can be overridden with ``key=value`` (OmegaConf dot-list).

Output (``<dataset.save_dir>/Tx{R}x{C}_Rx{R}x{C}.npy``), loaded with
``np.load(path, allow_pickle=True)``: an array of ``dataset.num_frame`` dicts
    "csi":          complex64 [num_rx, Nr, Nt, 2 * num_sc]
                    (uplink subcarriers first, then downlink subcarriers)
    "rx_positions": float32   [3, num_rx]  (x, y, z in meters)
The resolved config is saved next to it as ``Tx{R}x{C}_Rx{R}x{C}.yaml``.
"""

import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf
from sionna.rt import PathSolver, Transmitter

from ladckm_gen.csi import align_csi, calculate_csi
from ladckm_gen.scatterers import distribute_dynamic_scatters
from ladckm_gen.scene import distribute_receivers, init_antenna_arrays, load_and_configure_scene
from ladckm_gen.utils import init_logger, init_random_seed

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"),
                        help="path to the YAML config file")
    parser.add_argument("overrides", nargs="*",
                        help="config overrides as key=value, e.g. tx.num_cols=8 dataset.num_rx=500")
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    # The scene path is relative to this folder, independent of the working directory
    if not os.path.isabs(cfg.scene.path):
        cfg.scene.path = os.path.join(ROOT, cfg.scene.path)
    return cfg


def preliminary(cfg):
    """Set up logging, output path and RNG; log the key configuration."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    os.makedirs(cfg.dataset.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.dataset.log_dir, f"{timestamp}.log")
    file_logger, disp_logger = init_logger(log_path)
    disp_logger.info(f"==> Log saved at: {log_path}")

    os.makedirs(cfg.dataset.save_dir, exist_ok=True)
    tag = f"Tx{cfg.tx.num_rows}x{cfg.tx.num_cols}_Rx{cfg.rx.num_rows}x{cfg.rx.num_cols}"
    save_path = os.path.join(cfg.dataset.save_dir, f"{tag}.npy")
    OmegaConf.save(cfg, os.path.join(cfg.dataset.save_dir, f"{tag}.yaml"))
    disp_logger.info(f"==> Dataset will be saved at: {save_path}")

    init_random_seed(cfg.seed)
    disp_logger.info(f"==> Using random seed: {cfg.seed}")
    disp_logger.info("\n" + OmegaConf.to_yaml(cfg))

    return file_logger, disp_logger, save_path


def generate_dataset(cfg):
    file_logger, disp_logger, save_path = preliminary(cfg)

    # Scene, antenna arrays and the BS
    scene, scatters_ped, scatters_veh = load_and_configure_scene(cfg, disp_logger)
    init_antenna_arrays(cfg, scene, disp_logger)
    scene.add(Transmitter(name="tx", position=list(cfg.tx.position), power_dbm=cfg.tx.power_dbm))

    num_rx = cfg.dataset.num_rx
    num_sc = cfg.dataset.num_sc
    tx_size = cfg.tx.num_rows * cfg.tx.num_cols
    rx_size = cfg.rx.num_rows * cfg.rx.num_cols
    fc_up, fc_down = cfg.ray_tracing.fc_up, cfg.ray_tracing.fc_down
    dataset = []

    for frame in range(cfg.dataset.num_frame):
        disp_logger.info(f"\n{'*' * 30} Frame {frame + 1} {'*' * 30}")

        # Random environment of this frame
        scene.temperature = random.uniform(*cfg.scene.temperature)
        rx_positions = distribute_receivers(num_rx, cfg, scene, disp_logger, file_logger, frame)
        distribute_dynamic_scatters(cfg, scene, scatters_ped, scatters_veh, disp_logger, file_logger, frame)

        # Ray tracing; UEs without valid CSI are replaced by newly dropped ones
        valid_rx_positions = torch.empty((3, 0))   # [3, num_valid_rx]
        h_up_all, h_down_all = [], []
        while valid_rx_positions.shape[1] < num_rx:
            p_solver = PathSolver()
            disp_logger.info("==> Calculating CSI...")
            h_up, duration, indices_up = calculate_csi(cfg, scene, p_solver, fc_up)
            disp_logger.info(f"==> Calculating uplink CSI... Completed. Duration={duration} mins:secs")
            h_down, duration, indices_down = calculate_csi(cfg, scene, p_solver, fc_down)
            disp_logger.info(f"==> Calculating downlink CSI... Completed. Duration={duration} mins:secs\n")

            h_up, h_down, common_indices = align_csi(h_up, indices_up, h_down, indices_down)
            h_up_all.append(h_up)
            h_down_all.append(h_down)
            valid_rx_positions = torch.cat((valid_rx_positions, rx_positions[:, common_indices.tolist()]), dim=1)

            if valid_rx_positions.shape[1] < num_rx:
                disp_logger.info(f"==> Not enough valid Rx points ({valid_rx_positions.shape[1]}). Refilling...")
                rx_positions = distribute_receivers(num_rx - valid_rx_positions.shape[1],
                                                    cfg, scene, disp_logger, file_logger)

            del h_up, h_down, indices_up, indices_down
            torch.cuda.empty_cache()

        # [num_rx, Nr, 1, Nt, 1, num_sc] x 2 -> [num_rx, Nr, Nt, 2 * num_sc]
        h = torch.cat((torch.cat(h_up_all, dim=0), torch.cat(h_down_all, dim=0)), dim=-1)
        h = torch.reshape(h, (num_rx, rx_size, tx_size, 2 * num_sc))
        dataset.append({
            "csi": h.cpu().numpy(),
            "rx_positions": valid_rx_positions.cpu().numpy(),
        })

        del h_up_all, h_down_all, h
        torch.cuda.empty_cache()

    disp_logger.info(f"\n{'*' * 30} All Frames Done {'*' * 30}")
    np.save(save_path, dataset)
    disp_logger.info(f"==> Dataset saved to: {save_path}\n")


if __name__ == "__main__":
    generate_dataset(load_config())
