# GramsSim-tools

This repo is an amalgamation of tools developed for running [GramsSim](https://codeberg.org/wgseligman/GramsSim) and analyzing its outputs. It started with code developed for predicting the occupancy/background rates that the miniGRAMS detector will see during the 2026 pGRAMS flight, but I've since expanded it to do all kinds of post-processing and simulation exploration, since I found certain tools/scripts were useful everywhere. Reach out with any questions to st3624@columbia.edu.


## Directory Stucture

```
GramsSim-tools/
├── scripts/              # Python scripts and simulation driver pipeline
├── notebooks/            # Jupyter notebooks for plotting and working with script outputs
│   ├── exploring/        # random data exploration
│   └── occupancy/        # occupancy & trigger-rate notebooks
├── data/                 # output maps / sim files (gitignored; see .gitignore)
├── occupancy_grams.xml   # GramsSim options file for the occupancy study
└── psf.xml               # GramsSim options file for the PSF study
```

Simulation outputs (`*.root`, `*.hepmc3`, `*.pkl`, `*.fits`) and generated maps should live under a different data directory configured in [scripts/config.py](scripts/config.py).


## Scripts

| Script | Purpose |
| --- | --- |
| [config.py](scripts/config.py) | Main config file. Use to setup directory paths, particle definitions (PDG codes, masses), HEALPix map properties, TPC geometry, flight location/date/altitude. Used by all other scripts. Supports both CLI use (run from `scripts/`) and notebook import (`from scripts.config import ...`) |
| [make_flux_maps.py](scripts/make_flux_maps.py) | Convert PARMA/EXPACS angular-flux CSV into per-particle HEALPix FITS flux maps, to be input into GramsSky (optionally also produce animated GIFs). |
| [run_sim.py](scripts/run_sim.py) | Drive the full GramsSim chain (`gramssky → gramsg4 → gramsdetsim → gramsreadoutsim → gramselecsim → opticalsim → opdetsim`) for any set of particles. You can select specific portions of the chain to run using `--start-from` / `--stop-after`. |
| [weights.py](scripts/weights.py) | Compute the total integrated flux `S` directly from the GramsSky input FITS maps and derive per-particle weight factors converting event counts into physical rates (Hz). Saves to `weights.pkl` inside MAP_DIR. |
| [extract_df.py](scripts/extract_df.py) | Extract per-event quantities from all GramsSim ROOT trees into a single `events.parquet`. |
| [extract_compton.py](scripts/extract_compton.py) | Truth-based per-Compton-daughter extraction (vertices, deposited energy, pixel/SiPM hits) into `compton.parquet`. |
| [event_display.py](scripts/event_display.py) | Build 2D LArTPC event-display images (xz / yz views) from GramsElecSim ADC waveforms. |
| [light_config.py](scripts/light_config.py) | Reference module for the pGRAMS light-sensor layout: 144 SiPMs → 36 channels → 9 cells, plus the trigger scheme. Run directly for self-tests. |
| [utils.py](scripts/utils.py) | Shared helpers (ROOT/dictionary loading, momentum → spherical angles). |

## Usage

Configure first, then run the pipeline. All scripts are run from `scripts/` and
take their defaults from `config.py`.

```bash
cd scripts

# 1. Build HEALPix flux maps from the PARMA/EXPACS CSV
python make_flux_maps.py                       # all particles
python make_flux_maps.py --particles photon proton --gifs

# 2. Run the GramsSim chain
python run_sim.py                              # all particles, full chain
python run_sim.py --particles photon --stop-after gramsg4

# 3. Weights + per-event / per-Compton extraction
python weights.py
python extract_df.py
python extract_compton.py

# Explore outputs in notebooks!
```

Most scripts accept `--particles <names...>`; particle names are the keys of
`PARTICLE_DICT` in `config.py` (e.g. `photon`, `proton`, `neutro`, `he---4`,
`muplus`, `electr`). Extraction scripts accept `--force` to overwrite cached output.

## Requirements

- A built [GramsSim](https://codeberg.org/wgseligman/GramsSim) work area (set
  `GS_DIR` in `config.py`); the stage executables and `libDictionary.so` must exist there.
- ROOT (with PyROOT), NumPy, pandas, healpy, matplotlib.
- PARMA/EXPACS flux CSVs already generated for the chosen location/date (see `EXPACS_DIR` / `FLUX_CSV`).
