"""Scene loading, antenna arrays and receiver (UE) dropping."""

import mitsuba as mi
import numpy as np
import torch
from sionna.rt import PlanarArray, Receiver, load_scene
from tqdm import tqdm

from .utils import Timer

# Materials of the static scene that are given a non-zero scattering coefficient
SCATTERING_MATERIALS = ["itu_marble", "itu_concrete", "itu_medium_dry_ground"]


def load_and_configure_scene(cfg, disp_logger):
    """Load the campus scene and create the pools of dynamic scatterers.

    The scene file contains one template ``pedestrian`` box and one template
    ``vehicle`` box. Each template is cloned ``num_range[1]`` times (the maximum
    number that can appear in a frame) and then removed. In every frame, a
    random subset of these clones is placed into the scene, see
    :func:`ladckm_gen.scatterers.distribute_dynamic_scatters`.

    Returns:
        scene (sionna.rt.Scene): The loaded scene.
        scatters_ped (list): Pool of pedestrian objects.
        scatters_veh (list): Pool of vehicle objects.
    """
    disp_logger.info("==> Loading scene...")
    scene = load_scene(cfg.scene.path, merge_shapes=True)
    for obj in scene.objects.values():
        if obj.radio_material.name in SCATTERING_MATERIALS:
            obj.radio_material.scattering_coefficient = cfg.scene.scattering_coefficient
            obj.radio_material.xpd_coefficient = cfg.scene.xpd_coefficient
    disp_logger.info("==> Loading scene... Completed.\n")

    disp_logger.info("==> Initializing dynamic scatters...")
    # Pedestrians: dry-ground dielectric boxes (0.5 m x 0.5 m x 1.8 m)
    ped = scene.get("pedestrian")
    ped.radio_material.thickness = cfg.scene.pedestrian.thickness
    scatters_ped = [ped.clone(name=f"pedestrian-{i}")
                    for i in range(cfg.scene.pedestrian.num_range[1])]
    scene.edit(add=scatters_ped)
    scene.edit(remove=ped)
    # Vehicles: metal boxes (2 m x 4 m x 1.6 m)
    veh = scene.get("vehicle")
    scatters_veh = [veh.clone(name=f"vehicle-{i}")
                    for i in range(cfg.scene.vehicle.num_range[1])]
    scene.edit(add=scatters_veh)
    scene.edit(remove=veh)
    disp_logger.info("==> Initializing dynamic scatters... Completed.\n")

    return scene, scatters_ped, scatters_veh


def init_antenna_arrays(cfg, scene, disp_logger):
    """Set the planar antenna arrays of the BS (Tx) and of all UEs (Rx)."""
    disp_logger.info("==> Configuring antenna arrays...")
    for side in ("tx", "rx"):
        arr = cfg[side]
        disp_logger.info(
            f"    {side.upper()}: {arr.num_rows}x{arr.num_cols} UPA, "
            f"spacing (v, h) = ({arr.vertical_spacing}, {arr.horizontal_spacing}) wavelength, "
            f"pattern = {arr.pattern}, polarization = {arr.polarization}")
        array = PlanarArray(num_rows=arr.num_rows,
                            num_cols=arr.num_cols,
                            vertical_spacing=arr.vertical_spacing,
                            horizontal_spacing=arr.horizontal_spacing,
                            pattern=arr.pattern,
                            polarization=arr.polarization)
        if side == "tx":
            scene.tx_array = array
        else:
            scene.rx_array = array
    disp_logger.info("==> Configuring antenna arrays... Completed.\n")


def distribute_receivers(num_rx, cfg, scene, disp_logger, file_logger, frame=None):
    """Drop ``num_rx`` UEs uniformly at random and add them to the scene.

    Existing receivers are removed first, so the scene always contains exactly
    the receivers returned by this function, named ``rx-0 ... rx-{num_rx-1}``.
    The x/y coordinates keep a 5 m margin from the border of ``rx.dist_range``.

    Args:
        num_rx (int): Number of UEs to drop.
        frame (int, optional): Frame index, only used for logging.

    Returns:
        torch.Tensor: UE positions, shape ``[3, num_rx]`` (x, y, z in meters).
    """
    dist_range = cfg.rx.dist_range
    rx_positions = np.array([
        dist_range[0][0] + 5 + np.random.rand(num_rx) * (dist_range[0][1] - dist_range[0][0] - 10),
        dist_range[1][0] + 5 + np.random.rand(num_rx) * (dist_range[1][1] - dist_range[1][0] - 10),
        dist_range[2][0] + np.random.rand(num_rx) * (dist_range[2][1] - dist_range[2][0]),
    ]).astype(np.float32)

    disp_logger.info("==> Distributing receivers...")
    for name in list(scene.receivers.keys()):
        if name.startswith("rx-"):
            scene.remove(name)

    prefix = f"Frame [{frame + 1}/{cfg.dataset.num_frame}] || " if frame is not None else ""
    with Timer() as timer:
        for i in tqdm(range(num_rx), unit="rx", colour="green", desc=f"{prefix}Rx distributing"):
            scene.add(Receiver(name=f"rx-{i}",
                               position=mi.Point3f(*rx_positions[:, i].tolist())))
    file_logger.info(f"Rx distributing || {prefix}Duration={timer.duration} mins:secs")
    disp_logger.info("==> Distributing receivers... Completed.\n")

    return torch.tensor(rx_positions)
