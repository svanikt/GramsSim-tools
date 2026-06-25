#!/usr/bin/env python3
"""
extract_compton.py
Author: Svanik Tandon
Date: 2026-05-12

Per-Compton-daughter (truth-based) extraction from GramsSim ROOT files.
One row per (event, direct-daughter-of-primary) pair, listing the quantities
on the "What GramsSim can provide" slide:

  For each incoming primary:                 [stored on every daughter row]
    x, y, z, px, py, pz_true

  For each direct daughter (Compton electron / scattered gamma / ...):
    - Pixel ID where (max) deposition occurred
    - Sum of measured energy over all pixels (and uncertainty)
    - Average x, y, t of pixels (and uncertainty)
    - Time of (first hit) pixel
    - SiPM ID where (max) scintillation occurred
    - Time of (first hit) SiPM

Hits are attributed to a direct daughter by walking ParentID chains: any LAr
hit whose track ancestry leads back through a given direct daughter of the
primary is counted toward that daughter. SiPM matching is geometric: within
the same anode cell as the daughter's energy-weighted-mean (x, y), pick the
channel with the largest summed-ADC peak.

Output: {MAPS_DIR}/pkl/compton.parquet

Usage:
    python extract_compton.py
    python extract_compton.py --particles photon
    python extract_compton.py --force
"""
import os
import argparse
import numpy as np
import pandas as pd

try:
    from config import (PARTICLE_DICT, MAPS_DIR, GRAMSSIM_DICTIONARY,
                        LOCATION, DATA_DIR, DATE)
    from light_config import N_CHANNELS, CHANNEL_TO_CELL_XY, opdet_to_channel
    from extract_df import BASELINE_ADC
except ImportError:
    from scripts.config import (PARTICLE_DICT, MAPS_DIR, GRAMSSIM_DICTIONARY,
                                LOCATION, DATA_DIR, DATE)
    from scripts.light_config import N_CHANNELS, CHANNEL_TO_CELL_XY, opdet_to_channel
    from scripts.extract_df import BASELINE_ADC


# ---------------------------------------------------------------------------
# Geometry — derived from pgrams.gdml + GramsReadoutSim options.xml.
# ---------------------------------------------------------------------------
# LArTPC active region: 28.75 x 28.75 cm, centered at (0, 0).
# Readout plane: 100 x 100 pixels covering the active region.
# 3x3 cells with 0.5 cm separator sheets; cell centers at {-9.75, 0, +9.75} cm.
LAR_X_SIZE = 28.75   # cm
LAR_Y_SIZE = 28.75
READOUT_NX = 100
READOUT_NY = 100
PIXEL_PITCH_X = LAR_X_SIZE / READOUT_NX   # 0.2875 cm
PIXEL_PITCH_Y = LAR_Y_SIZE / READOUT_NY
PIXEL_ORIGIN_X = -LAR_X_SIZE / 2.0        # x of pixel (0, *)
PIXEL_ORIGIN_Y = -LAR_Y_SIZE / 2.0

# Cell-boundary x/y values that split the 3x3 grid (cell pitch 9.25 + 0.5 sep).
_CELL_BOUNDS_X = (-5.125, 5.125)
_CELL_BOUNDS_Y = (-5.125, 5.125)

# Time per OpDet ADC sample (ns). Verify against gramsopdetsim options if
# digitization settings have changed.
SIPM_SAMPLE_PERIOD_NS = 4.0
# Above-baseline pulse-onset threshold for "first SiPM hit time".
SIPM_ONSET_ADC = 50


def _pixel_index(x_cm, y_cm):
    """Return (ix, iy) ∈ [0, READOUT_NX-1] x [0, READOUT_NY-1] for a hit."""
    ix = int(np.clip(np.floor((x_cm - PIXEL_ORIGIN_X) / PIXEL_PITCH_X),
                     0, READOUT_NX - 1))
    iy = int(np.clip(np.floor((y_cm - PIXEL_ORIGIN_Y) / PIXEL_PITCH_Y),
                     0, READOUT_NY - 1))
    return (ix, iy)


def _pixel_center(ix, iy):
    return (PIXEL_ORIGIN_X + (ix + 0.5) * PIXEL_PITCH_X,
            PIXEL_ORIGIN_Y + (iy + 0.5) * PIXEL_PITCH_Y)


def _cell_xy(x_cm, y_cm):
    """Return (cell_x, cell_y) ∈ {0,1,2}^2 for the pgrams 3x3 layout."""
    cx = 0 if x_cm < _CELL_BOUNDS_X[0] else (2 if x_cm > _CELL_BOUNDS_X[1] else 1)
    cy = 0 if y_cm < _CELL_BOUNDS_Y[0] else (2 if y_cm > _CELL_BOUNDS_Y[1] else 1)
    return (cx, cy)


def _weighted_mean_std(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    wsum = w.sum()
    if wsum == 0 or v.size == 0:
        return float('nan'), float('nan')
    mean = np.average(v, weights=w)
    var = np.average((v - mean) ** 2, weights=w)
    return float(mean), float(np.sqrt(max(var, 0.0)))


def load_root():
    import ROOT
    ROOT.gSystem.Load(GRAMSSIM_DICTIONARY)
    return ROOT


# ---------------------------------------------------------------------------
# Per-event extraction
# ---------------------------------------------------------------------------

def _extract_event(tree, ROOT, has_opdet):
    """Return list of per-daughter rows from the current tree entry."""
    event_index = tree.EventID.Index()

    # --- LArHits ---
    hits = []
    for key, hit in tree.LArHits:
        try:
            tid = ROOT.std.get[0](key)
        except TypeError:
            tid = ROOT.std.get(key, 0)
        hits.append({
            'trackID': tid,
            'x': hit.StartX(), 'y': hit.StartY(), 'z': hit.StartZ(),
            't': hit.StartT(),
            'E': hit.Energy(),
        })

    # --- TrackList: build lookup and find primary ---
    track_lookup = {}   # trackID -> dict
    primary_id = None
    primary_info = None
    for trackID, track in tree.TrackList:
        label = track.Process() or "UnknownCreation"
        parent_id = track.ParentID()
        traj = track.Trajectory()
        if len(traj) > 0:
            p0 = traj[0]
            info = {
                'parent': parent_id, 'process': label,
                'pdg': track.PDGCode(),
                't_start': p0.t(),
                'x_start': p0.X(), 'y_start': p0.Y(), 'z_start': p0.Z(),
                'px': p0.Px(), 'py': p0.Py(), 'pz': p0.Pz(),
                'energy': p0.E(),
            }
        else:
            info = {
                'parent': parent_id, 'process': label,
                'pdg': track.PDGCode(),
                't_start': float('inf'),
                'x_start': np.nan, 'y_start': np.nan, 'z_start': np.nan,
                'px': np.nan, 'py': np.nan, 'pz': np.nan,
                'energy': np.nan,
            }
        track_lookup[trackID] = info
        if label == "Primary":
            primary_id = trackID
            primary_info = info

    if primary_id is None or not hits:
        return []

    # --- Attribute each hit to a direct daughter via ancestry walk ---
    hits_by_daughter = {}
    for h in hits:
        current = h['trackID']
        if current == primary_id:
            continue
        for _ in range(100):
            info = track_lookup.get(current)
            if info is None:
                current = None
                break
            if info['parent'] == primary_id:
                break
            current = info['parent']
        else:
            current = None
        if current is None or current == primary_id:
            continue
        hits_by_daughter.setdefault(current, []).append(h)

    if not hits_by_daughter:
        return []

    # --- Optionally read OpDetWaveforms once per event ---
    channel_peaks = np.zeros(N_CHANNELS, dtype=np.int64)
    channel_onset_sample = np.full(N_CHANNELS, -1, dtype=np.int64)
    if has_opdet:
        channel_wfs = {}
        for readoutID, waveform in tree.OpDetWaveforms:
            vec = waveform.Digital()
            if vec.size() == 0:
                continue
            arr = np.asarray(vec, dtype=np.int64) - BASELINE_ADC
            ch = opdet_to_channel(readoutID)
            existing = channel_wfs.get(ch)
            if existing is None:
                channel_wfs[ch] = arr.copy()
            else:
                ml = min(existing.size, arr.size)
                existing[:ml] += arr[:ml]
        for ch, wf in channel_wfs.items():
            channel_peaks[ch] = int(wf.max())
            above = np.where(wf > SIPM_ONSET_ADC)[0]
            if above.size > 0:
                channel_onset_sample[ch] = int(above[0])

    # --- Build per-daughter rows ---
    rows = []
    for d_tid, dh in hits_by_daughter.items():
        d_info = track_lookup[d_tid]

        # Bin hits into pixels.
        pixels = {}  # (ix, iy) -> {'E': float, 't_first': float}
        for h in dh:
            pid = _pixel_index(h['x'], h['y'])
            entry = pixels.get(pid)
            if entry is None:
                pixels[pid] = {'E': h['E'], 't_first': h['t']}
            else:
                entry['E'] += h['E']
                if h['t'] < entry['t_first']:
                    entry['t_first'] = h['t']

        pix_keys = list(pixels.keys())
        pix_E = np.array([pixels[k]['E'] for k in pix_keys])
        pix_t = np.array([pixels[k]['t_first'] for k in pix_keys])
        pix_xc = np.array([_pixel_center(*k)[0] for k in pix_keys])
        pix_yc = np.array([_pixel_center(*k)[1] for k in pix_keys])

        max_idx = int(np.argmax(pix_E))
        max_ix, max_iy = pix_keys[max_idx]
        max_pixel_E = float(pix_E[max_idx])
        max_pixel_x, max_pixel_y = _pixel_center(max_ix, max_iy)

        E_sum = float(pix_E.sum())
        # Poisson-like uncertainty on summed energy (treats deposit as ~sqrt(E)).
        E_err = float(np.sqrt(E_sum)) if E_sum > 0 else 0.0

        x_mean, x_err = _weighted_mean_std(pix_xc, pix_E)
        y_mean, y_err = _weighted_mean_std(pix_yc, pix_E)
        t_mean, t_err = _weighted_mean_std(pix_t, pix_E)
        first_pixel_t = float(pix_t.min())

        # SiPM matching: in the cell containing the energy-weighted mean.
        max_sipm_channel = -1
        max_sipm_peak_adc = -1
        first_sipm_hit_time_ns = float('nan')
        sipm_cell_x, sipm_cell_y = -1, -1
        if has_opdet:
            cxy = _cell_xy(x_mean, y_mean)
            sipm_cell_x, sipm_cell_y = cxy
            cell_channels = [ch for ch, c in CHANNEL_TO_CELL_XY.items() if c == cxy]
            if cell_channels:
                peaks = [(ch, int(channel_peaks[ch])) for ch in cell_channels]
                max_sipm_channel, max_sipm_peak_adc = max(peaks, key=lambda p: p[1])
                onsets = [channel_onset_sample[ch] for ch in cell_channels
                          if channel_onset_sample[ch] >= 0]
                if onsets:
                    first_sipm_hit_time_ns = float(min(onsets)) * SIPM_SAMPLE_PERIOD_NS

        rows.append({
            'event_index': event_index,
            # Daughter ID / kinematics
            'daughter_track_id': int(d_tid),
            'daughter_process': d_info['process'],
            'daughter_pdg': int(d_info['pdg']),
            'daughter_t_start': float(d_info['t_start']),
            'daughter_x_start': float(d_info['x_start']),
            'daughter_y_start': float(d_info['y_start']),
            'daughter_z_start': float(d_info['z_start']),
            'daughter_px': float(d_info['px']),
            'daughter_py': float(d_info['py']),
            'daughter_pz': float(d_info['pz']),
            'daughter_energy': float(d_info['energy']),
            # Primary truth (replicated on each daughter row)
            'primary_pdg': int(primary_info['pdg']),
            'primary_x_true': float(primary_info['x_start']),
            'primary_y_true': float(primary_info['y_start']),
            'primary_z_true': float(primary_info['z_start']),
            'primary_px_true': float(primary_info['px']),
            'primary_py_true': float(primary_info['py']),
            'primary_pz_true': float(primary_info['pz']),
            'primary_energy': float(primary_info['energy']),
            # Pixel summary
            'n_hits': len(dh),
            'n_pixels': len(pixels),
            'max_pixel_ix': int(max_ix),
            'max_pixel_iy': int(max_iy),
            'max_pixel_x': float(max_pixel_x),
            'max_pixel_y': float(max_pixel_y),
            'max_pixel_energy': max_pixel_E,
            'pixel_energy_sum': E_sum,
            'pixel_energy_err': E_err,
            'pixel_x_mean': x_mean,
            'pixel_y_mean': y_mean,
            'pixel_t_mean': t_mean,
            'pixel_x_err': x_err,
            'pixel_y_err': y_err,
            'pixel_t_err': t_err,
            'first_pixel_hit_time': first_pixel_t,
            # SiPM summary (in-cell light-charge match)
            'sipm_cell_x': int(sipm_cell_x),
            'sipm_cell_y': int(sipm_cell_y),
            'max_sipm_channel': int(max_sipm_channel),
            'max_sipm_peak_adc': int(max_sipm_peak_adc),
            'first_sipm_hit_time': first_sipm_hit_time_ns,
        })

    return rows


# ---------------------------------------------------------------------------
# Per-particle and orchestration
# ---------------------------------------------------------------------------

SIM_DIR = os.path.join(DATA_DIR, f"{LOCATION}_{DATE}_maps", "sim")
ACTIVE_PARENT_BRANCHES = ["EventID*", "TrackList*", "LArHits*"]
ACTIVE_FRIENDS = [("opdetsim", "OpDetSim", ["OpDetWaveforms*"])]


def _extract_particle(particle):
    ROOT = load_root()

    g4_path = os.path.join(SIM_DIR, f"{LOCATION}_{particle}_g4.root")
    if not os.path.exists(g4_path):
        print(f"[SKIP] g4 file not found: {g4_path}")
        return []

    g4_file = ROOT.TFile.Open(g4_path)
    if not g4_file or g4_file.IsZombie():
        print(f"[ERROR] Failed to open: {g4_path}")
        return []

    tree = g4_file.Get("gramsg4")
    if not tree:
        print(f"[ERROR] Tree 'gramsg4' not found in: {g4_path}")
        g4_file.Close()
        return []

    has_opdet = False
    for stage, tree_name, _patterns in ACTIVE_FRIENDS:
        friend_path = os.path.join(SIM_DIR, f"{LOCATION}_{particle}_{stage}.root")
        if os.path.exists(friend_path):
            tree.AddFriend(tree_name, friend_path)
            if stage == "opdetsim":
                has_opdet = True
        else:
            print(f"  [DEBUG] {stage} file not found for {particle}, skipping")

    tree.SetBranchStatus("*", 0)
    for pat in ACTIVE_PARENT_BRANCHES:
        tree.SetBranchStatus(pat, 1)
    for _stage, _tname, patterns in ACTIVE_FRIENDS:
        for pat in patterns:
            tree.SetBranchStatus(pat, 1)

    n_entries = tree.GetEntries()
    print(f"[INFO] Extracting {particle} ({n_entries} events)...", flush=True)

    rows = []
    for i_entry in range(n_entries):
        tree.GetEntry(i_entry)
        ev_rows = _extract_event(tree, ROOT, has_opdet)
        for r in ev_rows:
            r['particle'] = particle
        rows.extend(ev_rows)

    print(f"[INFO]   {particle} done ({n_entries} events, {len(rows)} daughters)",
          flush=True)
    g4_file.Close()
    return rows


def extract_all_daughters(particles=None, force=False, output_path=None,
                          n_workers=None):
    """Build (or load) the per-daughter parquet."""
    output = output_path or os.path.join(MAPS_DIR, "pkl", "compton.parquet")

    if not force and os.path.exists(output):
        print(f"[CACHE] Loading from {output}")
        return pd.read_parquet(output)

    if particles is None:
        particles = list(PARTICLE_DICT.keys())
    particles = [p for p in particles if p in PARTICLE_DICT]

    if n_workers is None:
        n_workers = min(8, len(particles))

    all_rows = []
    if n_workers <= 1 or len(particles) == 1:
        for p in particles:
            all_rows.extend(_extract_particle(p))
    else:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for rows in pool.imap_unordered(_extract_particle, particles):
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    int_cols = ['event_index', 'daughter_track_id', 'daughter_pdg', 'primary_pdg',
                'n_hits', 'n_pixels', 'max_pixel_ix', 'max_pixel_iy',
                'sipm_cell_x', 'sipm_cell_y',
                'max_sipm_channel', 'max_sipm_peak_adc']
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype('int64')

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"[INFO] Saved {len(df)} rows to {output}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Per-Compton-daughter truth extraction from GramsSim ROOT files")
    parser.add_argument("--particles", type=str, nargs="+",
                        help="Particles to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cached parquet exists")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output parquet path")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: min(8, n_particles))")
    args = parser.parse_args()

    df = extract_all_daughters(particles=args.particles, force=args.force,
                               output_path=args.output, n_workers=args.workers)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDaughters per particle:")
    print(df['particle'].value_counts().to_string())
    print(f"\nDaughter processes (top 10):")
    print(df['daughter_process'].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
