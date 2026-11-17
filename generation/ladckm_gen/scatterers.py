"""Placement of dynamic scatterers (pedestrians and vehicles).

The road layout below describes the bundled campus scene (``scene/scene.xml``),
in the scene's local coordinate system (meters). Adapt it if you use another
scene.
"""

import random

import mitsuba as mi
import numpy as np
from tqdm import tqdm

from .utils import Timer

# Height of the vehicle center above the ground (m)
VEHICLE_Z = 0.8

# Minimum distance between two vehicles on the same lane (m)
MIN_VEHICLE_GAP = 30

# Driving lanes. A lane along the x-axis spans ``x_range`` at fixed ``y``;
# a lane along the y-axis spans ``y_range`` at fixed ``x``. Vehicles are not
# placed inside the listed intersection areas.
LANES = [
    {'x_range': [-336, 310], 'y': -91, 'axis': 'x',
     'intersections': [{'x': [-121, -114], 'y': [-98, -91]}, {'x': [95, 102], 'y': [-98, -91]}]},
    {'x_range': [-336, 310], 'y': -98, 'axis': 'x',
     'intersections': [{'x': [-121, -114], 'y': [-98, -91]}, {'x': [95, 102], 'y': [-98, -91]}]},
    {'x_range': [-90, 310], 'y': 88, 'axis': 'x', 'intersections': []},
    {'x_range': [-90, 310], 'y': 81, 'axis': 'x', 'intersections': []},
    {'x': -121, 'y_range': [-190, 60], 'axis': 'y', 'intersections': [{'x': [-121, -114], 'y': [-98, -91]}]},
    {'x': -114, 'y_range': [-190, 60], 'axis': 'y', 'intersections': [{'x': [-121, -114], 'y': [-98, -91]}]},
    {'x': 95, 'y_range': [-190, 60], 'axis': 'y', 'intersections': [{'x': [95, 102], 'y': [81, 88]}]},
    {'x': 102, 'y_range': [-190, 60], 'axis': 'y', 'intersections': [{'x': [95, 102], 'y': [81, 88]}]},
]

# x-coordinates of the lanes along the y-axis; vehicles on them are rotated
VERTICAL_LANE_X = [lane['x'] for lane in LANES if lane['axis'] == 'y']

# Areas where pedestrians are not placed
PEDESTRIAN_EXCLUSION_ZONES = [
    {'x_range': [-336 - 2, 310 + 2], 'y_range': [-91 - 2, -91 + 2]},
    {'x_range': [-336 - 2, 310 + 2], 'y_range': [-98 - 2, -98 + 2]},
    {'x_range': [-121 - 2, -114 + 2], 'y_range': [-190 - 2, 60 + 2]},
    {'x_range': [95 - 2, 102 + 2], 'y_range': [-190 - 2, 60 + 2]},
]


def generate_scatter_positions(cfg, num_vehicles, num_pedestrians):
    """Randomly draw positions of vehicles and pedestrians.

    Vehicles: a lane is picked uniformly at random and a position is drawn
    uniformly along it; the draw is rejected if it is closer than
    ``MIN_VEHICLE_GAP`` to another vehicle on the same lane or falls into an
    intersection area.

    Pedestrians: positions are drawn uniformly within
    ``scene.pedestrian.dist_range`` (with a 5 m margin) and rejected if they
    fall into ``PEDESTRIAN_EXCLUSION_ZONES``.

    Returns:
        vehicle_positions (np.ndarray): Shape ``[3, num_vehicles]``.
        pedestrian_positions (np.ndarray): Shape ``[3, num_pedestrians]``.
    """
    # ---------------- vehicles ----------------
    lane_vehicles = [[] for _ in range(len(LANES))]

    def is_valid_vehicle_position(x, y, vehicles_on_lane, intersections, axis):
        for veh in vehicles_on_lane:
            if axis == 'x' and abs(x - veh[0]) < MIN_VEHICLE_GAP:
                return False
            if axis == 'y' and abs(y - veh[1]) < MIN_VEHICLE_GAP:
                return False
        for area in intersections:
            if area['x'][0] < x < area['x'][1] and area['y'][0] < y < area['y'][1]:
                return False
        return True

    while sum(len(vehs) for vehs in lane_vehicles) < num_vehicles:
        lane_idx = np.random.randint(0, len(LANES))
        lane = LANES[lane_idx]
        if 'x_range' in lane:
            x = np.random.uniform(lane['x_range'][0], lane['x_range'][1])
            y = lane['y']
        else:
            x = lane['x']
            y = np.random.uniform(lane['y_range'][0], lane['y_range'][1])
        if is_valid_vehicle_position(x, y, lane_vehicles[lane_idx], lane['intersections'], lane['axis']):
            lane_vehicles[lane_idx].append([x, y, VEHICLE_Z])

    vehicle_positions = np.array([veh for lane in lane_vehicles for veh in lane])

    # ---------------- pedestrians ----------------
    dist_range = cfg.scene.pedestrian.dist_range
    pedestrian_z = dist_range[2]

    def is_valid_pedestrian_position(x, y):
        for zone in PEDESTRIAN_EXCLUSION_ZONES:
            if (zone['x_range'][0] <= x <= zone['x_range'][1] and
                    zone['y_range'][0] <= y <= zone['y_range'][1]):
                return False
        return True

    pedestrians = []
    while len(pedestrians) < num_pedestrians:
        x = dist_range[0][0] + 5 + np.random.rand() * (dist_range[0][1] - dist_range[0][0] - 10)
        y = dist_range[1][0] + 5 + np.random.rand() * (dist_range[1][1] - dist_range[1][0] - 10)
        if is_valid_pedestrian_position(x, y):
            pedestrians.append([x, y, pedestrian_z])

    pedestrian_positions = np.array(pedestrians)

    return vehicle_positions.T, pedestrian_positions.T


def distribute_dynamic_scatters(cfg, scene, scatters_ped, scatters_veh, disp_logger, file_logger, frame):
    """Re-draw the dynamic scatterers for one frame (coherence block).

    The numbers of vehicles and pedestrians are drawn uniformly from
    ``scene.vehicle.num_range`` and ``scene.pedestrian.num_range``; the first
    ``num`` objects of each pool are added to the scene and moved to new random
    positions, the rest of the pool is removed from the scene.
    """
    disp_logger.info("==> Distributing dynamic scatters...")
    num_veh = random.randint(*cfg.scene.vehicle.num_range)
    num_ped = random.randint(*cfg.scene.pedestrian.num_range)
    scene.edit(remove=scatters_ped)
    scene.edit(remove=scatters_veh)
    scene.edit(add=scatters_ped[:num_ped])
    scene.edit(add=scatters_veh[:num_veh])

    veh_positions, ped_positions = generate_scatter_positions(cfg, num_veh, num_ped)
    veh_positions = veh_positions.astype(np.float32)
    ped_positions = ped_positions.astype(np.float32)

    prefix = f"Frame [{frame + 1}/{cfg.dataset.num_frame}] || {scene.temperature[0]:.2f} K"

    with Timer() as timer:
        for i in tqdm(range(num_ped), unit="pedestrian", colour="yellow",
                      desc=f"{prefix} || {num_ped} pedestrians distributing"):
            scatters_ped[i].position = mi.Point3f(*ped_positions[:, i].tolist())
    file_logger.info(f"Pedestrian distributing || {prefix} || Total={num_ped} || Duration={timer.duration} mins:secs")

    with Timer() as timer:
        for i in tqdm(range(num_veh), unit="vehicle", colour="red",
                      desc=f"{prefix} || {num_veh} vehicles distributing"):
            scatters_veh[i].position = mi.Point3f(*veh_positions[:, i].tolist())
            # Vehicles on lanes along the y-axis are turned to face +y
            if veh_positions[0, i] in VERTICAL_LANE_X:
                scatters_veh[i].look_at(scatters_veh[i].position + mi.Point3f(0., 1., 0.))
    file_logger.info(f"Vehicle distributing || {prefix} || Total={num_veh} || Duration={timer.duration} mins:secs")

    disp_logger.info("==> Distributing dynamic scatters... Completed.\n")
