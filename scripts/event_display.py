#!/usr/bin/env python3
"""
event_display.py
Author: Svanik Tandon
Date: 2026-06-16

LArTPC event-display images from GramsElecSim output.

For a chosen particle and event index, read the per-pixel digital ADC
waveforms (grams::ReadoutWaveforms in the *_elecsim.root ElecSim tree) and
create two 2D projection views:

  - xz view: anode x channel number vs drift z
  - yz view: anode y channel number vs drift z

Each readout pixel (grams::ReadoutID) carries an (x_index, y_index) anode
channel number and a digitized drift-time waveform (ReadoutWaveform.digital,
12-bit ADC). The x/y axes are just the channel numbers. The drift (z) axis is
reconstructed from the ADC sample index: each sample is SAMPLE_PERIOD_US long
and ionization electrons drift at DRIFT_VELOCITY, so

    z[cm] = sample_index * SAMPLE_PERIOD_US * DRIFT_VELOCITY .

This module only needs the elecsim file; the ReadoutID already encodes the
anode (x, y) pixel, so no readout map is required.

Usage (see notebooks/exploring/event_display.ipynb):

    from scripts.event_display import load_event_images, list_busy_events
    img = load_event_images('muplus', event_index=0)
"""
import os
import numpy as np

# Support both CLI (scripts/ on sys.path) and notebook import
# (GramsOccupancy on sys.path), mirroring extract_df.py.
try:
    from config import LOCATION, PARTICLE_DICT
    from extract_df import load_root, SIM_DIR
except ImportError:
    from scripts.config import LOCATION, PARTICLE_DICT
    from scripts.extract_df import load_root, SIM_DIR

# ---------------------------------------------------------------------------
# Digitization / drift constants
# ---------------------------------------------------------------------------
# The x/y axes are plotted as raw anode channel numbers, so no pixel pitch is
# needed. For reference, the pitch would be anodeTilePlaneXSize / x_resolution
# = 33.0 cm / 100 = 0.33 cm.

# ADC digitization. sample_freq = 2 MHz (occupancy_grams.xml, <gramselecsim>)
# -> 0.5 us per stored digital sample.
SAMPLE_PERIOD_US = 0.5         # us per ADC sample
# Electron drift velocity (occupancy_grams.xml, <gramsdetsim>):
#   ElectronDriftVelocity = 0.00016 cm/ns = 0.16 cm/us.
DRIFT_VELOCITY = 0.16          # cm/us
# Drift distance covered by one ADC sample.
Z_PER_SAMPLE = SAMPLE_PERIOD_US * DRIFT_VELOCITY   # cm/sample

# 12-bit digital waveform ceiling (baseline 0, saturates at 4096).
ADC_SATURATION = 4096

# Fixed display grid. The anode active region is read out as a 90 x 90 channel
# array (x and y channels both span -45..44), and every channel records a
# fixed-length readout window. The event display always spans this full grid so
# that every event is drawn at the same size, regardless of which channels fired.
X_CHANNEL_MIN, X_CHANNEL_MAX = -45, 44   # 90 x channels
Y_CHANNEL_MIN, Y_CHANNEL_MAX = -45, 44   # 90 y channels
N_READOUT_SAMPLES = 400                  # full readout window length (samples)


def elecsim_path(particle):
    """Path to the GramsElecSim output ROOT file for a given particle."""
    return os.path.join(SIM_DIR, f"{LOCATION}_{particle}_elecsim.root")


def _digital_to_array(digital):
    """Convert a grams ReadoutWaveform.digital (std::vector<int>) to ndarray."""
    n = digital.size()
    if n == 0:
        return np.empty(0, dtype=np.float64)
    try:
        # cppyy exposes the buffer; this is the fast path.
        return np.asarray(digital, dtype=np.float64)
    except Exception:
        return np.fromiter(digital, dtype=np.float64, count=n)


def load_event_images(particle, event_index, projection="max", z_units="cm"):
    """Build xz and yz ADC event-display images for one event.

    The images always span the full fixed grid: 90 x channels
    (X_CHANNEL_MIN..X_CHANNEL_MAX), 90 y channels, and the full
    N_READOUT_SAMPLES-long readout window, so every event is drawn at the same
    size. Channels/samples that did not fire are simply zero.

    Parameters
    ----------
    particle : str
        PARTICLE_DICT key (e.g. 'muplus', 'photon', 'proton').
    event_index : int
        Entry number in the ElecSim tree (0-based). Change this to step
        through events for the chosen particle.
    projection : {'max', 'sum'}
        How to collapse the projected-out axis. 'max' keeps the colour axis
        in true ADC amplitude (peak sample along the projection); 'sum' gives
        the integrated ADC across the projected axis.
    z_units : {'cm', 'us', 'sample'}
        Units for the drift (z) axis. 'cm' = drift distance, 'us' = drift
        time, 'sample' = raw ADC sample number.

    Returns
    -------
    dict with keys:
        particle, event_index, event_id, n_channels, projection, z_units
        xz, yz          : 2D ndarrays, shape (90, N) / (90, N), ADC counts
        x_edges, y_edges : 1D ndarrays of channel-number bin edges
        z_edges          : 1D ndarray of drift bin edges in `z_units`
        z_label          : matplotlib-ready axis label for z
        nsamples         : length of the readout window (samples)
    Events with no readout waveforms still return full-size all-zero images.
    """
    if projection not in ("max", "sum"):
        raise ValueError("projection must be 'max' or 'sum'")
    if z_units not in ("cm", "us", "sample"):
        raise ValueError("z_units must be 'cm', 'us', or 'sample'")

    ROOT = load_root()
    path = elecsim_path(particle)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No elecsim file for particle '{particle}': {path}")

    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise IOError(f"Could not open {path}")
    tree = f.Get("ElecSim")
    n_entries = tree.GetEntries()
    if not (0 <= event_index < n_entries):
        f.Close()
        raise IndexError(
            f"event_index {event_index} out of range [0, {n_entries}) "
            f"for particle '{particle}'")

    tree.GetEntry(event_index)
    event_id = tree.EventID.Index()

    # Fixed full-grid images: 90 x channels, 90 y channels, full readout window.
    nx = X_CHANNEL_MAX - X_CHANNEL_MIN + 1
    ny = Y_CHANNEL_MAX - Y_CHANNEL_MIN + 1
    nsamp = N_READOUT_SAMPLES
    xz = np.zeros((nx, nsamp), dtype=np.float64)
    yz = np.zeros((ny, nsamp), dtype=np.float64)

    n_channels = 0
    for readoutID, wf in tree.ReadoutWaveforms:
        arr = _digital_to_array(wf.digital)
        if arr.size == 0:
            continue
        n_channels += 1
        ix = readoutID.X() - X_CHANNEL_MIN
        iy = readoutID.Y() - Y_CHANNEL_MIN
        L = min(arr.size, nsamp)
        if 0 <= ix < nx:
            if projection == "max":
                np.maximum(xz[ix, :L], arr[:L], out=xz[ix, :L])
            else:
                xz[ix, :L] += arr[:L]
        if 0 <= iy < ny:
            if projection == "max":
                np.maximum(yz[iy, :L], arr[:L], out=yz[iy, :L])
            else:
                yz[iy, :L] += arr[:L]
    f.Close()

    # Channel-number bin edges: channel k is centered at integer k, so its
    # cell spans [k-0.5, k+0.5).
    x_edges = np.arange(X_CHANNEL_MIN, X_CHANNEL_MAX + 2) - 0.5
    y_edges = np.arange(Y_CHANNEL_MIN, Y_CHANNEL_MAX + 2) - 0.5

    # Drift (z) bin edges in the requested units. Sample s spans [s, s+1).
    sample_edges = np.arange(0, nsamp + 1)
    if z_units == "cm":
        z_edges = sample_edges * Z_PER_SAMPLE
        z_label = "drift z [cm]"
    elif z_units == "us":
        z_edges = sample_edges * SAMPLE_PERIOD_US
        z_label = r"drift time [$\mu$s]"
    else:  # 'sample'
        z_edges = sample_edges.astype(float)
        z_label = "ADC sample"

    return {
        "particle": particle,
        "event_index": event_index,
        "event_id": event_id,
        "n_channels": n_channels,
        "projection": projection,
        "z_units": z_units,
        "xz": xz, "yz": yz,
        "x_edges": x_edges, "y_edges": y_edges,
        "z_edges": z_edges, "z_label": z_label,
        "nsamples": nsamp,
    }


def _z_conversion(z_units):
    """Return (scale, label) to turn an ADC sample index into a z coordinate."""
    if z_units == "cm":
        return Z_PER_SAMPLE, "drift z [cm]"
    if z_units == "us":
        return SAMPLE_PERIOD_US, r"drift time [$\mu$s]"
    if z_units == "sample":
        return 1.0, "ADC sample"
    raise ValueError("z_units must be 'cm', 'us', or 'sample'")


def load_event_points(particle, event_index, z_units="cm",
                      threshold=None, rel_threshold=0.05):
    """Return sparse 3D voxel hits for one event, for a 3D scatter display.

    Each fired pixel carries a drift-time waveform; every ADC sample above
    threshold becomes one voxel at (x_channel, y_channel, z), coloured by its
    ADC value. The sample index is mapped to z exactly as in
    `load_event_images` (sample centre = (s + 0.5) * scale).

    Parameters
    ----------
    particle, event_index : as in load_event_images.
    z_units : {'cm', 'us', 'sample'}
        Units for the drift (z) coordinate.
    threshold : float, optional
        Absolute ADC cut. Samples with ADC > threshold are kept.
    rel_threshold : float
        If `threshold` is None, keep samples above rel_threshold * (event peak
        ADC). Default 0.05 (5% of the peak).

    Returns
    -------
    dict with keys:
        particle, event_index, event_id, n_channels, n_points, z_units
        x, y       : 1D ndarrays of channel numbers (one entry per voxel)
        z          : 1D ndarray of drift coordinate in `z_units`
        adc        : 1D ndarray of ADC values
        z_label    : matplotlib-ready axis label for z
        max_adc    : peak ADC in the event
        threshold  : the absolute ADC cut actually used
        grid       : (X_CHANNEL_MIN, X_CHANNEL_MAX, Y_CHANNEL_MIN,
                      Y_CHANNEL_MAX, z_full) full fixed-grid extent, where
                      z_full is the full readout window in `z_units`.
    """
    scale, z_label = _z_conversion(z_units)

    ROOT = load_root()
    path = elecsim_path(particle)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No elecsim file for particle '{particle}': {path}")
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise IOError(f"Could not open {path}")
    tree = f.Get("ElecSim")
    n_entries = tree.GetEntries()
    if not (0 <= event_index < n_entries):
        f.Close()
        raise IndexError(
            f"event_index {event_index} out of range [0, {n_entries}) "
            f"for particle '{particle}'")

    tree.GetEntry(event_index)
    event_id = tree.EventID.Index()

    # First pass: pull every pixel waveform into memory and find the peak ADC.
    pix = []  # (x_channel, y_channel, ndarray)
    max_adc = 0.0
    for readoutID, wf in tree.ReadoutWaveforms:
        arr = _digital_to_array(wf.digital)
        if arr.size == 0:
            continue
        pix.append((readoutID.X(), readoutID.Y(), arr))
        m = arr.max()
        if m > max_adc:
            max_adc = float(m)
    f.Close()

    thr = threshold if threshold is not None else rel_threshold * max_adc
    grid = (X_CHANNEL_MIN, X_CHANNEL_MAX, Y_CHANNEL_MIN, Y_CHANNEL_MAX,
            N_READOUT_SAMPLES * scale)

    xs, ys, zs, adcs = [], [], [], []
    for x, y, arr in pix:
        samples = np.nonzero(arr > thr)[0]
        if samples.size == 0:
            continue
        xs.append(np.full(samples.size, x))
        ys.append(np.full(samples.size, y))
        zs.append((samples + 0.5) * scale)
        adcs.append(arr[samples])

    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        adc = np.concatenate(adcs)
    else:
        x = y = z = adc = np.empty(0)

    return {
        "particle": particle,
        "event_index": event_index,
        "event_id": event_id,
        "n_channels": len(pix),
        "n_points": x.size,
        "z_units": z_units,
        "x": x, "y": y, "z": z, "adc": adc,
        "z_label": z_label,
        "max_adc": max_adc,
        "threshold": thr,
        "grid": grid,
    }


def list_busy_events(particle, n=15, max_scan=300):
    """List the events with the most fired readout pixels, to help pick one.

    Returns a list of (event_index, n_channels) sorted by n_channels
    descending. Scans the first `max_scan` events (~0.05 s/event, since each
    entry deserializes the full waveform map); set max_scan=None for the
    whole tree.
    """
    ROOT = load_root()
    path = elecsim_path(particle)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No elecsim file for particle '{particle}': {path}")
    f = ROOT.TFile.Open(path)
    tree = f.Get("ElecSim")
    n_entries = tree.GetEntries()
    if max_scan is not None:
        n_entries = min(n_entries, max_scan)
    counts = np.empty(n_entries, dtype=int)
    for i in range(n_entries):
        tree.GetEntry(i)
        counts[i] = tree.ReadoutWaveforms.size()
    f.Close()
    order = np.argsort(counts)[::-1][:n]
    return [(int(i), int(counts[i])) for i in order]
