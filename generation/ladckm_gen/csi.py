"""Ray tracing and uplink/downlink channel frequency response (CFR)."""

import math

import numpy as np
import torch
from sionna.rt.utils import subcarrier_frequencies

from .utils import Timer

# Path attributes stored per receiver (first dimension = receiver index)
PATH_ATTRS = ['_a_real', '_a_imag', '_tau', '_theta_t', '_phi_t', '_theta_r', '_phi_r', '_doppler']

# Upper bound on the number of CFR elements computed in a single ``paths.cfr`` call
MAX_CFR_SIZE = 2 ** 32


def clean_path(paths):
    """Drop receivers that are not reached by any path.

    Args:
        paths (sionna.rt.Paths): Paths with ``a`` of shape
            ``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]``.

    Returns:
        paths (sionna.rt.Paths): The same object, restricted (in place) to the
            receivers with at least one non-zero path coefficient.
        keep_indices (torch.Tensor): Indices of the kept receivers (on the CPU).
    """
    a_real = paths.a[0].torch()
    a_imag = paths.a[1].torch()
    is_zero = (torch.all(a_real == 0, dim=(1, 2, 3, 4)) & torch.all(a_imag == 0, dim=(1, 2, 3, 4))).cpu()
    keep_indices = torch.where(~is_zero)[0]

    def keep(attr_name, data):
        setattr(paths, attr_name, type(getattr(paths, attr_name))(data[keep_indices]))

    keep('_a_real', a_real)
    keep('_a_imag', a_imag)
    for attr_name in PATH_ATTRS[2:]:
        keep(attr_name, getattr(paths, attr_name).torch())

    return paths, keep_indices


def split_and_call_cfr(cfg, paths, frequencies, normalize=False, normalize_delays=True):
    """Compute ``paths.cfr`` in receiver chunks to bound GPU memory usage.

    The receivers are split into ``ceil(size / MAX_CFR_SIZE)`` chunks, where
    ``size`` is the number of real values of the CFR (path coefficients x
    subcarriers x 2). Each chunk is moved to the CPU right after it is
    computed. The original path attributes are restored afterwards.

    Returns:
        torch.Tensor: CFR on the CPU, shape
            ``[num_rx, num_rx_ant, num_tx, num_tx_ant, 1, num_sc]``.
    """
    num_rx = paths.tau.shape[0]
    if num_rx == 0:
        return torch.tensor([])

    total_size = np.prod(paths.a[0].shape) * cfg.dataset.num_sc * 2
    split_factor = 1 if total_size < MAX_CFR_SIZE else math.ceil(total_size / MAX_CFR_SIZE)
    step = num_rx // split_factor

    orig_attrs = {name: getattr(paths, name) for name in PATH_ATTRS}
    full_attrs = {
        '_a_real': paths.a[0], '_a_imag': paths.a[1], '_tau': paths.tau,
        '_theta_t': paths.theta_t, '_phi_t': paths.phi_t,
        '_theta_r': paths.theta_r, '_phi_r': paths.phi_r, '_doppler': paths.doppler,
    }

    results = []
    for i in range(split_factor):
        stop = (i + 1) * step if i < split_factor - 1 else num_rx
        for name, value in full_attrs.items():
            setattr(paths, name, value[i * step:stop, ...])
        results.append(paths.cfr(frequencies=frequencies, normalize=normalize,
                                 normalize_delays=normalize_delays, out_type="torch").cpu())

    for name, value in orig_attrs.items():
        setattr(paths, name, value)

    return torch.cat(results, dim=0)


def calculate_csi(cfg, scene, p_solver, fc):
    """Trace paths at carrier frequency ``fc`` and compute the CFR of all UEs.

    Args:
        fc (float): Carrier frequency (Hz).

    Returns:
        h (torch.Tensor): CFR of the UEs that are reached by at least one path,
            shape ``[num_valid_rx, num_rx_ant, 1, num_tx_ant, 1, num_sc]``.
            Path gains are not normalized (``normalize=False``), and delays are
            normalized so that the first path arrives at t = 0
            (``normalize_delays=True``).
        duration (str): Elapsed time, ``mm:ss``.
        indices (torch.Tensor): Indices of these UEs among all UEs in the scene.
    """
    rt = cfg.ray_tracing
    with Timer() as timer:
        scene.frequency = fc
        paths = p_solver(scene=scene,
                         max_depth=rt.max_depth,
                         los=True,
                         specular_reflection=rt.specular_reflection,
                         diffuse_reflection=rt.diffuse_reflection,
                         refraction=rt.refraction,
                         samples_per_src=rt.samples_per_src,
                         max_num_paths_per_src=rt.max_num_paths_per_src,
                         synthetic_array=rt.synthetic_array,
                         seed=cfg.seed)
        paths, indices = clean_path(paths)
        # Subcarrier frequencies relative to the carrier frequency
        frequencies = subcarrier_frequencies(cfg.dataset.num_sc, cfg.dataset.sc_spacing)
        h = split_and_call_cfr(cfg, paths, frequencies, normalize=False, normalize_delays=True)
    return h, timer.duration, indices


def align_csi(h_up, indices_up, h_down, indices_down):
    """Keep only the UEs that have valid CSI in both uplink and downlink.

    Returns:
        h_up_aligned, h_down_aligned (torch.Tensor): CFRs of the common UEs, in
            ascending UE index order.
        common_indices (torch.Tensor): Indices of the common UEs among all UEs
            in the scene.
    """
    common = sorted(set(indices_up.tolist()) & set(indices_down.tolist()))
    common_indices = torch.tensor(common, device=indices_up.device)

    up_mapping = torch.nonzero(torch.isin(indices_up, common_indices), as_tuple=True)[0]
    down_mapping = torch.nonzero(torch.isin(indices_down, common_indices), as_tuple=True)[0]

    return h_up[up_mapping], h_down[down_mapping], common_indices
