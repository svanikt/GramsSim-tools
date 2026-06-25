#!/usr/bin/env python3
"""
run_sim.py
Author: Svanik Tandon
Date: 2026-01-25
Allows you to run configurable portions of the GramsSim chain for specific set of particles. Bill Seligman
has a similar version of this built into GramsSim as well (./runsim), but this is a bit more 

Runs: gramssky -> gramsg4 -> gramsdetsim -> gramsreadoutsim -> gramselecsim

Configuration is set in config.py (LOCATION, DATE, etc.)

Usage:
    python run_sim.py
    python run_sim.py --particles photon proton
    python run_sim.py --stop-after gramsg4
"""
import argparse
import os
import sys
import subprocess

from config import (
    PARTICLE_DICT, OPTIONS_FILE, GS_DIR, NUM_EVENTS,
    LOCATION, DATE, MAPS_DIR, LIGHTMAP_DIR,
    ensure_dirs
)

# simulation stages in order
SIM_STAGES = ['gramssky', 'gramsg4', 'gramsdetsim', 'gramsreadoutsim', 'gramselecsim', 'opticalsim', 'opdetsim']

def run_stage(exe, options_file, args_list):
    """Run a GramsSim stage with given arguments."""
    cmd = [exe, options_file] + args_list
    print(f"    {' '.join(cmd)}")

    # Run from GS_DIR so relative paths in options.xml work
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=GS_DIR)
    if result.returncode != 0:
        print(f"[ERROR] {os.path.basename(exe)} failed:")
        print(result.stderr[:500] if result.stderr else "(no stderr)")
        return False
    return True


def get_file_paths(sim_dir, prefix):
    """Get file paths for a given map simulation."""
    return {
        'hepmc3': os.path.join(sim_dir, f"{prefix}.hepmc3"),
        'g4': os.path.join(sim_dir, f"{prefix}_g4.root"),
        'detsim': os.path.join(sim_dir, f"{prefix}_detsim.root"),
        'readoutsim': os.path.join(sim_dir, f"{prefix}_readoutsim.root"),
        'elecsim': os.path.join(sim_dir, f"{prefix}_elecsim.root"),
        'opticalsim': os.path.join(sim_dir, f"{prefix}_opticalsim.root"),
        'opdetsim': os.path.join(sim_dir, f"{prefix}_opdetsim.root")
    }


def run_particle_sim(particle, fits_file, sim_dir, options_file, num_events,
                     start_stage='gramssky', stop_stage='gramselecsim'):
    """Run full simulation chain for a single particle."""
    particle_name, pdg_code, mass, flux_suffix = PARTICLE_DICT[particle]
    prefix = f"{LOCATION}_{particle}"
    paths = get_file_paths(sim_dir, prefix)

    stages_to_run = SIM_STAGES[SIM_STAGES.index(start_stage):SIM_STAGES.index(stop_stage)+1]

    for stage in stages_to_run:
        exe = os.path.join(GS_DIR, stage)

        if stage == 'gramssky':
            args = [
                "--MapEnergyBandsFile", fits_file,
                "--PrimaryPDG", pdg_code,
                "-n", str(num_events),
                "-o", paths['hepmc3'],
            ]
            print(f"  [{stage}] -> {os.path.basename(paths['hepmc3'])}")

        elif stage == 'gramsg4':
            if not os.path.exists(paths['hepmc3']):
                print(f"  [ERROR] HepMC3 file not found: {paths['hepmc3']}")
                return False
            args = ["-i", paths['hepmc3'], "-o", paths['g4']]
            print(f"  [{stage}] -> {os.path.basename(paths['g4'])}")

        elif stage == 'gramsdetsim':
            if not os.path.exists(paths['g4']):
                print(f"  [ERROR] G4 file not found: {paths['g4']}")
                return False
            args = ["-i", paths['g4'], "-o", paths['detsim']]
            print(f"  [{stage}] -> {os.path.basename(paths['detsim'])}")

        elif stage == 'gramsreadoutsim':
            if not os.path.exists(paths['detsim']):
                print(f"  [ERROR] DetSim file not found: {paths['detsim']}")
                return False
            args = ["-i", paths['detsim'], "-o", paths['readoutsim']]
            print(f"  [{stage}] -> {os.path.basename(paths['readoutsim'])}")

        elif stage == 'gramselecsim':
            if not os.path.exists(paths['detsim']):
                print(f"  [ERROR] DetSim file not found: {paths['detsim']}")
                return False
            if not os.path.exists(paths['readoutsim']):
                print(f"  [ERROR] ReadoutSim file not found: {paths['readoutsim']}")
                return False
            args = ["-i", paths['detsim'], "-m", paths['readoutsim'], "-o", paths['elecsim']]
            print(f"  [{stage}] -> {os.path.basename(paths['elecsim'])}")

        elif stage == 'opticalsim':
            if not os.path.exists(paths['g4']):
                print(f"  [ERROR] G4 file not found: {paths['g4']}")
                return False
            if not os.path.exists(LIGHTMAP_DIR):
                print(f"  [ERROR] Lightmap directory not found: {LIGHTMAP_DIR}")
                return False
            args = ["-i", paths['g4'], "-m", os.path.join(LIGHTMAP_DIR, "lightmap*.root"), "-o", paths['opticalsim']]
            print(f"  [{stage}] -> {os.path.basename(paths['opticalsim'])}")

        elif stage == 'opdetsim':
            if not os.path.exists(paths['opticalsim']):
                print(f"  [ERROR] OpticalSim file not found: {paths['opticalsim']}")
                return False
            args = ["-i", paths['opticalsim'], "-o", paths['opdetsim']]
            print(f"  [{stage}] -> {os.path.basename(paths['opdetsim'])}")

        else:
            print(f"  [SKIP] Unknown stage: {stage}")
            continue

        success = run_stage(exe, options_file, args)
        if not success:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run full GramsSim chain for particle flux maps")
    parser.add_argument("--particles", type=str, nargs="+", help="Particles to simulate (default: all)")
    parser.add_argument("--num-events", type=int, default=NUM_EVENTS, help=f"Number of events (default, from config.py: {NUM_EVENTS})")
    parser.add_argument("--options", type=str, default=OPTIONS_FILE, help="GramsSim options XML file")
    parser.add_argument("--start-from", type=str, default="gramssky", choices=SIM_STAGES,
                        help="Start from this stage (default: gramssky)")
    parser.add_argument("--stop-after", type=str, default="opdetsim", choices=SIM_STAGES,
                        help="Stop after this stage (default: opdetsim)")
    args = parser.parse_args()

    ensure_dirs(MAPS_DIR)

    if not os.path.exists(args.options):
        print(f"[ERROR] Options file not found: {args.options}")
        sys.exit(1)

    particles = args.particles if args.particles else list(PARTICLE_DICT.keys())
    fits_dir = os.path.join(MAPS_DIR, "fits")
    sim_dir = os.path.join(MAPS_DIR, "sim")

    print(f"Configuration:")
    print(f"  Location: {LOCATION}")
    print(f"  Date: {DATE}")
    print(f"  Num events: {args.num_events}")
    print(f"  Stages: {args.start_from} -> {args.stop_after}")
    print(f"  Particles: {particles}")
    print()

    for particle in particles:
        if particle not in PARTICLE_DICT:
            print(f"[WARN] Unknown particle: {particle}, skipping")
            continue

        particle_name = PARTICLE_DICT[particle][0]
        fits_file = os.path.join(fits_dir, f"{LOCATION}_{DATE}_{particle}.fits")

        print(f"=== {particle_name} ({particle}) ===")

        if not os.path.exists(fits_file):
            print(f"  [ERROR] FITS file not found: {fits_file}")
            print(f"  Run make_flux_maps.py first!")
            continue

        success = run_particle_sim(
            particle, fits_file, sim_dir, args.options, args.num_events,
            args.start_from, args.stop_after
        )

        if success:
            print(f"  [OK] Done")
        else:
            print(f"  [FAIL] Simulation chain failed")
        print()


if __name__ == "__main__":
    main()
