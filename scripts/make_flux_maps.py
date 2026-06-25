#!/usr/bin/env python3
"""
make_flux_maps.py
Author: Svanik Tandon
Date: 2025-01-22

Convert PARMA/EXPACS angular flux CSV to HEALPix FITS maps for all particles.

Configuration is set in config.py (LOCATION, DATE, etc.)

Usage:
    python make_flux_maps.py
    python make_flux_maps.py --particles photon proton
    python make_flux_maps.py --gifs
    
"""
import numpy as np
import healpy as hp
import pandas as pd
import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# plt.style.use('~/latex-cm.mplstyle')

from config import (
    EXPACS_DIR, PARTICLE_DICT, NSIDE,
    LOCATION, DATE, FLUX_CSV, MAPS_DIR,
    ensure_dirs
)


def make_flux_map(nside, df, particle, energy):
    """Create a HEALPix flux map for a given particle and energy bin."""
    _df = df.query('particle == "{}" & energy == {}'.format(particle, energy))
    costhetas = _df['costheta'].values
    delta_costheta = 2.0 / (len(costhetas) - 1)
    fluxes = _df['flux'].values  # /cm2/s/MeV/sr

    npix = hp.nside2npix(nside)
    flux_map = np.zeros(npix)

    for ipix in range(npix):
        ang = hp.pix2ang(nside, ipix, nest=False)
        costheta = np.cos(ang[0])

        flux = 0
        if costheta >= 1.0 - delta_costheta:
            flux = fluxes[-2]
        else:
            index_costheta = int((costheta + 1) / delta_costheta)
            fraction = (costheta + 1) / delta_costheta - index_costheta
            flux = (1.0 - fraction) * fluxes[index_costheta] + fraction * fluxes[index_costheta + 1]

        flux_map[ipix] = flux

    return flux_map


def run_mk_healpix_map(csv_file, output_fits, particle):
    """Run the mk_healpix_map.py script to generate FITS file."""
    mk_healpix_script = os.path.join(EXPACS_DIR, "fitsfile_20221207", "mk_healpix_map.py")

    if not os.path.exists(mk_healpix_script):
        print(f"[ERROR] mk_healpix_map.py not found at: {mk_healpix_script}")
        return False

    cmd = f'python "{mk_healpix_script}" "{csv_file}" "{output_fits}" "{particle}"'
    print(f"  Running HEALPix map script for {particle}...")
    ret = os.system(cmd)
    return ret == 0


def create_flux_map_gif(df, particle, particle_name, output_gif, nside=NSIDE, fps=5, frame_step=2):
    """Create animated GIF showing HEALPix flux maps scrolling across energy bins."""
    import warnings

    _df = df.query('particle == "{}"'.format(particle))
    energies = np.sort(np.unique(_df['energy']))
    N = len(energies)

    # generate flux maps for all energies
    all_flux_maps = []
    for i in range(N):
        flux_map = make_flux_map(nside, df, particle, energies[i])
        all_flux_maps.append(flux_map)

    fig = plt.figure(figsize=(8, 5))

    def draw_frame(i):
        fig.clf()
        data = np.asarray(all_flux_maps[i])
        data = np.log10(np.maximum(data, 1e-30))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hp.mollview(
                data,
                title=f"{particle_name} Flux Map for {energies[i]:.2E} MeV",
                unit="log10(1/cm²/s/MeV/sr)",
                fig=fig.number,
                hold=False
            )
            hp.graticule()

    ani = animation.FuncAnimation(fig, draw_frame, frames=range(0, N, frame_step), interval=1000 / fps)
    ani.save(output_gif, writer="pillow", fps=fps * 1.5, dpi=200)
    plt.close(fig)

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate HEALPix FITS maps from PARMA flux data")
    parser.add_argument("--particles", type=str, nargs="+", help="Particles to process (default: all)")
    parser.add_argument("--gifs", action="store_true", help="Generate animated GIFs of flux maps")
    args = parser.parse_args()

    # use config values
    csv_file = FLUX_CSV
    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV file not found: {csv_file}")
        print(f"  Check LOCATION, DATE, ALTITUDE_M in config.py")
        sys.exit(1)

    fits_dir = os.path.join(MAPS_DIR, "fits")

    print(f"Configuration:")
    print(f"  Location: {LOCATION}")
    print(f"  Date: {DATE}")

    print(f"\nLoading flux data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")

    ensure_dirs(MAPS_DIR)

    # determine which particles to process
    particles = args.particles if args.particles else list(PARTICLE_DICT.keys())

    print(f"\nGenerating FITS maps for {len(particles)} particles...")

    gifs_dir = os.path.join(MAPS_DIR, "gifs")

    for particle in particles:
        if particle not in PARTICLE_DICT:
            print(f"  [WARN] Unknown particle: {particle}, skipping. Check PARTICLE_DICT in config.py.")
            continue

        particle_name = PARTICLE_DICT[particle][0]
        output_fits = os.path.join(fits_dir, f"{LOCATION}_{DATE}_{particle}.fits")
        print(f"\n[{particle}] -> {output_fits}")

        success = run_mk_healpix_map(csv_file, output_fits, particle)
        if success:
            print(f"  [OK] Created FITS for {particle}")
        else:
            print(f"  [FAIL] Failed to create FITS for {particle}")

        # create GIF if requested
        if args.gifs:
            output_gif = os.path.join(gifs_dir, f"{LOCATION}_{DATE}_{particle}.gif")
            print(f"  Creating GIF for {particle}...")
            try:
                create_flux_map_gif(df, particle, particle_name, output_gif)
                print(f"  [OK] Created GIF for {particle}")
            except Exception as e:
                print(f"  [FAIL] GIF creation failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
