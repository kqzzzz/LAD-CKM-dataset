"""Logging and random-seed helpers."""

import logging
import random
import time

import numpy as np
import torch


def init_logger(log_path):
    """Create the two loggers used throughout the generator.

    Args:
        log_path (str): Path of the log file.

    Returns:
        file_logger: Writes to the log file only (per-step timing details).
        disp_logger: Writes to both the console and the log file.
    """
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_logger = logging.getLogger("ladckm.file")
    disp_logger = logging.getLogger("ladckm.disp")
    for logger, handlers in ((file_logger, [file_handler]),
                             (disp_logger, [console_handler, file_handler])):
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in handlers:
            logger.addHandler(handler)
    return file_logger, disp_logger


def init_random_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs.

    All random draws of the generator (temperature, scatterer counts and
    positions, UE positions) go through Python's ``random`` and NumPy, so a
    fixed seed makes the generated dataset reproducible.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    np.random.seed(seed)
    random.seed(seed)


class Timer:
    """Context manager measuring wall-clock time, formatted as ``mm:ss``."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        mins, secs = divmod(time.time() - self._start, 60)
        self.duration = f"{int(mins):02d}:{int(secs):02d}"
        return False
