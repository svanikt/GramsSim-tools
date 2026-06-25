"""
config.py
Author: Svanik Tandon + CC
Date: 2026-01-22

Config for GramsOccupancy sim and analysis scripts.
"""

import os
import xml.etree.ElementTree as ET # for parsing the options file you use

# =============================================================================
# Directory and File Paths
# =============================================================================
# repo root (this file lives in <repo>/scripts/)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS_DIR = "/nevis/wanaka/share/standon/GRAMS/GramsSim-work"

# directory to store simulation outputs
DATA_DIR = "/nevis/wanaka/data/standon/minigrams_occupancy"

# directory for PARMA/EXPACS
PARMA_DIR = "/nevis/wanaka/share/standon/GRAMS/parma/parma_cpp"
EXPACS_DIR = "/nevis/wanaka/share/standon/GRAMS/parma/expacs_share"

# directory for pre-generated light maps
LIGHTMAP_DIR = "/nevis/riverside/share/seligman/grams/GramsG4-build"

# options file to be used with GramsSim
OPTIONS_FILE = os.path.join(REPO_DIR, "occupancy_grams.xml")

# GramsSim data structures ROOT dictionary
GRAMSSIM_DICTIONARY = os.path.join(GS_DIR, "libDictionary.so")

# =============================================================================
# GramsSim Options XML parser
# =============================================================================
# read option values straight from the options XML (OPTIONS_FILE) so that anything
# configured there is consistent here
_OPT_CAST = {
    "double": float,
    "integer": int,
    "boolean": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "string": str,
}

def get_option(name: str, section: str = "gramssky", options_file: str = None):
    """Return a typed option value from the GramsSim options XML.

    Looks in `section` first (the program that actually uses the option),
    then falls back to <global>.
    """
    tree = ET.parse(options_file or OPTIONS_FILE)
    root = tree.getroot()
    for sect in (section, "global"):
        node = root.find(sect)
        if node is None:
            continue
        for opt in node.findall("option"):
            if opt.get("name") == name:
                return _OPT_CAST.get(opt.get("type"), str)(opt.get("value"))
    raise KeyError(f"option {name!r} not found in {options_file or OPTIONS_FILE}")

# =============================================================================
# Particle Dictionary
# =============================================================================
# format: 'parma_particle_name': [display name for plots, PDG code, mass in MeV]

PARTICLE_DICT = {
    'neutro': ['Neutron', '2112', 939.565,],
    'proton': ['Proton', '2212', 938.272,],
    'he---4': ['Helium-4', '1000020040', 3727.38],
    'muplus': [r'$\mu^+$', '-13', 105.66],
    'mumins': [r'$\mu^-$', '13', 105.66],
    'electr': ['Electron', '11', 0.511],
    'positr': ['Positron', '-11', 0.511],
    'photon': ['Photon', '22', 0.0],
}

# =============================================================================
# HEALPix Configuration
# =============================================================================
NSIDE = 32  # HEALPix resolution. NSIDE=32 corresponds to 2^32 = 12288 pixels

# =============================================================================
# TPC Geometry
# =============================================================================
# miniGRAMS TPC dimensions
TPC_DIMENSIONS = {
    'x': 30,  # cm
    'y': 30,  # cm
    'z': 10,  # cm (drift, changed to 10 cm 6/25/26)
}

# average cross-sectional area of TPC
TPC_SURFACE_AREA = 2 * (TPC_DIMENSIONS['x'] * TPC_DIMENSIONS['y']) + 4 * (TPC_DIMENSIONS['x'] * TPC_DIMENSIONS['z'])  # cm^2
TPC_AVG_CROSS_SECTION = TPC_SURFACE_AREA / 6  # cm^2

# =============================================================================
# GramsSim Parameters (from the options XML)
# =============================================================================
# NOTE: if any of these were overridden with flags when running the sim, make sure to manually account for that
NUM_EVENTS = get_option("events", "gramssky")

# energy band limits and unit, used for weights calculation in weights.py
ENERGY_MIN = get_option("EnergyMin", "gramssky")   # MeV
ENERGY_MAX = get_option("EnergyMax", "gramssky")   # MeV
ENERGY_UNIT = get_option("EnergyUnit", "global")   # "MeV"

# =============================================================================
# Location Configuration
# =============================================================================
# pGRAMS flight location, date, and altitude
LOCATION = "tucson"
DATE = "2025_8_31" # maybe try and use predictive solar data?
ALTITUDE_M = 20000 # changed to 20km 6/25/26

# location name mappings for CSV filenames
LOCATION_NAMES = {
    'tucson': 'SpaceportTucson_Arizona',
    'esrange': 'Esrange_Sweden',
}

# path for flux CSV file
loc_name = LOCATION_NAMES.get(LOCATION, LOCATION)
FLUX_CSV = os.path.join(EXPACS_DIR, "parma_cpp_edit", "AngOutCsv", f"{loc_name}_{DATE}_alt{ALTITUDE_M}m.csv")

# output directory for maps and simulation data
MAPS_DIR = os.path.join(DATA_DIR, f"{LOCATION}_{DATE}_maps")

# ensure directories exist
def ensure_dirs(maps_dir: str) -> None:
    subdirs = ['fits', 'sim', 'txt', 'gifs']
    for subdir in subdirs:
        os.makedirs(os.path.join(maps_dir, subdir), exist_ok=True)
