"""
utils.py
Author: Svanik Tandon
Date: 2026-01-26

Shared utility functions for GramsOccupancy scripts.
"""
import numpy as np
from typing import Tuple, Optional

from config import GRAMSSIM_DICTIONARY


def load_root():
    """
    Import ROOT and load the GramsSim dictionary.

    Returns
    -------
    ROOT module
    """
    import ROOT
    ROOT.gSystem.Load(GRAMSSIM_DICTIONARY)
    return ROOT


def theta_phi(px: float, py: float, pz: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert momentum components to spherical angles.

    Parameters
    ----------
    px, py, pz : float
        Momentum components

    Returns
    -------
    theta : float or None
        Polar angle (radians)
    phi : float or None
        Azimuthal angle (radians)
    """
    p = np.sqrt(px * px + py * py + pz * pz)
    if p == 0:
        return None, None
    costh = pz / p
    th = np.arccos(np.clip(costh, -1.0, 1.0))
    ph = np.mod(np.arctan2(py, px), 2 * np.pi)
    return th, ph
