"""
weights.py
Author: Svanik Tandon
Date: 2026-01-25

Calculation of weights for GramsOccupancy analysis. Computes weight factors for
all particles based on the total integrated flux S, and saves to
MAPS_DIR/pkl/weights.pkl.

S is computed directly from the GramsSky *input* HEALPix FITS maps (the same
files make_flux_maps.py generates and feeds to gramssky via --MapEnergyBandsFile),
replicating gramssky's MapEnergyBands band-integration in Python. No int_flux_*.txt
files and no edit to GramsSim are required.
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import pickle
import argparse

import healpy as hp
from astropy.io import fits

from config import (NSIDE, NUM_EVENTS, PARTICLE_DICT, MAPS_DIR, LOCATION, DATE,
                    ENERGY_MIN, ENERGY_MAX, TPC_AVG_CROSS_SECTION)

def integrated_flux_from_fits(fits_path: str,
                              energy_min: float = ENERGY_MIN,
                              energy_max: float = ENERGY_MAX
                              ) -> Tuple[np.ndarray, List[np.ndarray], float]:
    """
    Compute band-integrated flux maps directly from a GramsSky input FITS map,
    replicating gramssky's MapEnergyBands integration (GramsSky/src/MapEnergyBands.cc).

    The FITS map stores one HEALPix differential-flux map per energy band, with
    the band count in header key NMAP and band energies (MeV) in keys ENE{n}.
    For each adjacent band pair (e1, e2) with per-pixel differential fluxes
    (n1, n2), a power law f(E) = n1 * (E/e1)^(-alpha) is assumed and integrated
    analytically over [e1, e2]:

        alpha  = -log(n2/n1) / log(e2/e1)
        intg_1 = n1 * e1 / (-alpha + 1)
        intg_2 = n1 * (1/e1)^(-alpha) * e2^(-alpha+1) / (-alpha + 1)
        J      = intg_2 - intg_1                     # per pixel, per band

    Only bands with energy in [max(1e-9, energy_min), energy_max) are used,
    matching the clamp/limits gramssky applies (EnergyMin/EnergyMax).

    Returns
    -------
    pixel_idx : np.ndarray
        Pixel indices (0 .. Npix-1).
    J_bands : list of np.ndarray
        One per-pixel integrated-flux array per band.
    S : float
        Total integrated flux summed over all pixels and bands
        [cm^-2 s^-1 sr^-1].
    """
    with fits.open(fits_path) as hdul:
        header = hdul[1].header
        nmap = int(header["NMAP"])
        energies = np.array([header[f"ENE{i}"] for i in range(1, nmap + 1)],
                            dtype=float)
    maps = np.atleast_2d(hp.read_map(fits_path, field=range(nmap), hdu=1))

    # Select bands within the gramssky energy limits (EnergyMin clamped to 1e-9).
    emin = max(1e-9, energy_min)
    keep = (energies >= emin) & (energies < energy_max)
    energies = energies[keep]
    maps = maps[keep]

    J_bands = []
    for i in range(len(energies) - 1):
        n1, n2 = maps[i], maps[i + 1]
        e1, e2 = energies[i], energies[i + 1]
        alpha = -np.log(n2 / n1) / np.log(e2 / e1)
        intg_1 = n1 * e1 / (-alpha + 1)
        intg_2 = n1 * (1.0 / e1) ** (-alpha) * e2 ** (-alpha + 1) / (-alpha + 1)
        J_bands.append(intg_2 - intg_1)

    S = float(sum(np.sum(J) for J in J_bands))
    pixel_idx = np.arange(maps.shape[1])

    return pixel_idx, J_bands, S

def calculate_weight_factor(S: float, nside: int, n_events: int) -> Tuple[float, float]:
    """Calculate the weight factor for converting event counts to rates."""
    omega = (4 * np.pi) / (12 * nside ** 2)  # HEALPix pixel solid angle
    T = n_events / (S * omega)  # effective time represented by simulation
    w = TPC_AVG_CROSS_SECTION / T # weighting factor to convert counts to rates (in Hz)

    return T, w

def main():
    parser = argparse.ArgumentParser(description="Calculate weights for GramsOccupancy analysis")
    parser.add_argument("--particles", type=str, nargs="+", help="Particles to calculate weights for (default: all)")
    args = parser.parse_args()

    particles_to_run = args.particles if args.particles else list(PARTICLE_DICT.keys())
    n_events = NUM_EVENTS

    weights: Dict[str, Dict[str, float]] = {}

    for particle in particles_to_run:
        if particle not in PARTICLE_DICT:
            print(f"[SKIP] Unknown particle: {particle}")
            continue

        fits_file = os.path.join(
            MAPS_DIR, "fits", f"{LOCATION}_{DATE}_{particle}.fits"
        )

        if not os.path.exists(fits_file):
            print(f"[ERROR] FITS map not found for {particle}: {fits_file}")
            print(f"  Run make_flux_maps.py first!")
            continue

        _, _, S = integrated_flux_from_fits(fits_file)
        T, w = calculate_weight_factor(S, NSIDE, n_events)

        weights[particle] = {'S': S, 'T': T, 'w': w}

        print(f"[INFO] Particle: {particle}, S: {S:.3e} cm^-2 s^-1 sr^-1, T: {T:.3e} cm^2 s, w: {w:.3e} s^-1,")

    # save weights to pickle file
    pkl_dir = os.path.join(MAPS_DIR, "pkl")
    os.makedirs(pkl_dir, exist_ok=True)
    weights_file = os.path.join(pkl_dir, "weights.pkl")
    with open(weights_file, "wb") as f:
        pickle.dump(weights, f)

    print(f"[INFO] Weights saved to: {weights_file}")


if __name__ == "__main__":
    main()
