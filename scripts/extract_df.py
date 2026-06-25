#!/usr/bin/env python3
"""
extract_df.py
Author: Svanik Tandon
Date: 2026-04-09

Extract per-event quantities from all GramsSim ROOT trees into a single
pandas DataFrame saved as parquet. Loops once per particle, loading all
available simulation stage trees as friends.

Output: {MAPS_DIR}/pkl/events.parquet

Usage:
    python extract_df.py
    python extract_df.py --particles photon proton
    python extract_df.py --force
"""
import os
import argparse
import numpy as np
import pandas as pd

# Support both CLI (`python scripts/extract_df.py`, scripts/ on sys.path)
# and notebook import (`from scripts.extract_df import ...`, GramsOccupancy on sys.path).
try:
    from config import PARTICLE_DICT, MAPS_DIR, GRAMSSIM_DICTIONARY, LOCATION, DATA_DIR, DATE
    from light_config import opdet_to_channel, N_CHANNELS
except ImportError:
    from scripts.config import PARTICLE_DICT, MAPS_DIR, GRAMSSIM_DICTIONARY, LOCATION, DATA_DIR, DATE
    from scripts.light_config import opdet_to_channel, N_CHANNELS


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_root():
    """Import ROOT and load GramsSim dictionary."""
    import ROOT
    ROOT.gSystem.Load(GRAMSSIM_DICTIONARY)
    return ROOT

def decode_opdet_id(det_id):
    """Decode OpDet ID into cell/SiPM coordinates.
    ID scheme: 5ZZYYXX, with ZZ tens digit 0=VUV, 1=visible."""
    remainder = det_id - 5000000
    if remainder >= 100000:
        remainder -= 100000
    yy = remainder // 100
    xx = remainder % 100
    return {
        'cell_x': xx // 4, 'sipm_x': xx % 4,
        'cell_y': yy // 4, 'sipm_y': yy % 4,
    }

# 12-bit ADC constants. Each simulated SiPM waveform has a baseline of
# BASELINE_ADC and saturates at ADC_MAX. When chaining SIPMS_PER_GROUP SiPMs
# into one readout, we subtract the baseline from each waveform before summing
# so the summed baseline stays at zero, then clip the result to the summed
# above-baseline dynamic range. The stored peak_adc is the peak of the summed,
# baseline-subtracted waveform; add SUMMED_BASELINE_ADC back for plotting on
# the full summed ADC scale (0 to SUMMED_ADC_MAX).
BASELINE_ADC = 2048
ADC_MAX = 4096
ADC_DYNAMIC_RANGE = ADC_MAX - BASELINE_ADC  # 2048 (single SiPM)

SIPMS_PER_GROUP = 4
SUMMED_BASELINE_ADC = SIPMS_PER_GROUP * BASELINE_ADC  # 8192
SUMMED_ADC_MAX = SIPMS_PER_GROUP * ADC_MAX            # 16384
SUMMED_DYNAMIC_RANGE = SIPMS_PER_GROUP * ADC_DYNAMIC_RANGE  # 8192

_sipm_group_cache = {}

def get_sipm_group(det_id):
    """Return a group key for 2x2 SiPM chaining.
    Groups adjacent SiPMs: (cell_x, cell_y, sipm_x//2, sipm_y//2).
    Naturally separates VUV (sipm_x=2,3) from visible (sipm_x=0,1)."""
    g = _sipm_group_cache.get(det_id)
    if g is not None:
        return g
    d = decode_opdet_id(det_id)
    g = (d['cell_x'], d['cell_y'], d['sipm_x'] // 2, d['sipm_y'] // 2)
    _sipm_group_cache[det_id] = g
    return g


def _is_active_lar(volume_id):
    """Check if volumeID corresponds to active LAr TPC (T-digit == 1)."""
    return volume_id // 1000000 == 1


def _classify_containment(daughter_volumes, primary_final_vol, n_lar_hits):
    """
    Classify an event as contained, escaped, incomplete, or inactive.
        - inactive: no LAr hits
        - incomplete: has LAr hits but first daughter not in active LAr
        - escaped: first daughter in active LAr but some leakage
        - contained: all daughters and primary endpoint in active LAr

    Parameters
    ----------
    daughter_volumes : list of (time, volumeID)
        Sorted by time. Start volumeIDs of all direct daughters.
    primary_final_vol : int
        VolumeID at the primary's final trajectory point.
    n_lar_hits : int
        Number of LAr hits in the event.

    Returns
    -------
    str : "contained", "escaped", "incomplete", or "inactive"
    """
    if n_lar_hits == 0:
        return "inactive"

    # Has hits — check if first daughter is in active LAr
    if len(daughter_volumes) == 0 or not _is_active_lar(daughter_volumes[0][1]):
        return "incomplete"

    # First daughter in active LAr — check full containment
    all_in_active = all(_is_active_lar(vol) for _, vol in daughter_volumes)
    primary_in_active = _is_active_lar(primary_final_vol)

    if all_in_active and primary_in_active:
        return "contained"
    else:
        return "escaped"


# ---------------------------------------------------------------------------
# Per-event extraction from a single tree entry
# ---------------------------------------------------------------------------

def _extract_event(tree, ROOT, has_opdet):
    """Extract all quantities from the current tree entry.

    Returns a dict with one row's worth of data.
    """
    row = {}
    row['event_index'] = tree.EventID.Index()

    # ------------------------------------------------------------------
    # extractions from LArHits: number of photons, ionization energy, hit track IDs
    # ------------------------------------------------------------------
    scint_sum = 0
    cer_sum = 0
    ion_sum = 0.0
    hit_trackIDs = set()
    n_hits = 0

    for key, hit in tree.LArHits:
        n_hits += 1
        scint_sum += hit.numPhotons
        cer_sum += hit.cerPhotons
        ion_sum += hit.Energy()
        try:
            tid = ROOT.std.get[0](key)
        except TypeError:
            tid = ROOT.std.get(key, 0)
        hit_trackIDs.add(tid)

    row['n_lar_hits'] = n_hits
    row['total_scint_photons'] = scint_sum
    row['total_cer_photons'] = cer_sum
    row['total_ionization_energy'] = ion_sum

    # ------------------------------------------------------------------
    # TrackList: primary info, daughter processes, containment
    # ------------------------------------------------------------------
    # First pass: build full track lookup and identify primary. We need the
    # complete lookup before we can walk ancestry chains (a hit may come from
    # a grandchild whose parent isn't yet known when we first see it).
    track_lookup = {}  # trackID -> (parent_id, process_label, t_start, vol_start)
    primary_id = None
    primary_final_vol = -1

    for trackID, track in tree.TrackList:
        label = track.Process() or "UnknownCreation"
        parent_id = track.ParentID()
        traj = track.Trajectory()
        t_start = traj[0].t() if len(traj) > 0 else float('inf')
        vol_start = traj[0].Identifier() if len(traj) > 0 else -1
        track_lookup[trackID] = (parent_id, label, t_start, vol_start)

        if label == "Primary":
            primary_id = trackID
            if len(traj) > 0:
                row['primary_energy'] = traj[0].momentum.E()
                row['primary_pdg'] = track.PDGCode()
                row['primary_px'] = traj[0].momentum.Px()
                row['primary_py'] = traj[0].momentum.Py()
                row['primary_pz'] = traj[0].momentum.Pz()
                primary_final_vol = traj[len(traj) - 1].Identifier()

    if primary_id is None:
        row.setdefault('primary_energy', np.nan)
        row.setdefault('primary_pdg', 0)
        row.setdefault('primary_px', np.nan)
        row.setdefault('primary_py', np.nan)
        row.setdefault('primary_pz', np.nan)

    # Second pass: collect direct daughters (for daughter_processes / containment).
    all_daughter_procs = []  # (time, process_label)
    daughter_volumes = []    # (time, volumeID)
    if primary_id is not None:
        for tid, (pid, label, t_start, vol_start) in track_lookup.items():
            if tid == primary_id or pid != primary_id:
                continue
            all_daughter_procs.append((t_start, label))
            daughter_volumes.append((t_start, vol_start))

    # Dominant process (Method B): for every track that produced a hit, walk
    # up ParentID links until we reach a direct daughter of the primary. The
    # process label we record is that direct-daughter ancestor's, not the
    # grandchild's. Across all hit chains, pick the earliest-time ancestor.
    earliest_daughter = (float('inf'), None)
    if primary_id is not None:
        for hit_tid in hit_trackIDs:
            if hit_tid == primary_id:
                continue
            current = hit_tid
            for _ in range(100):  # safety cap on ancestry depth
                info = track_lookup.get(current)
                if info is None:
                    break
                pid, label, t_start, _ = info
                if pid == primary_id:
                    if t_start < earliest_daughter[0]:
                        earliest_daughter = (t_start, label)
                    break
                current = pid

    row['dominant_daughter_process'] = earliest_daughter[1]
    all_daughter_procs.sort(key=lambda x: x[0])
    row['daughter_processes'] = (
        ','.join(proc for _, proc in all_daughter_procs) if all_daughter_procs else None
    )

    # Containment
    daughter_volumes.sort(key=lambda x: x[0])
    row['containment'] = _classify_containment(daughter_volumes, primary_final_vol, row['n_lar_hits'])

    # ------------------------------------------------------------------
    # OpDetWaveforms: per-channel summed peak ADC (baseline-subtracted,
    # clipped to SUMMED_DYNAMIC_RANGE). Channels group 4 SiPMs each via
    # light_config.opdet_to_channel; see light_config.py for the mapping.
    # ------------------------------------------------------------------
    if has_opdet:
        channel_waveforms = {}
        for readoutID, waveform in tree.OpDetWaveforms:
            vec = waveform.Digital()
            if vec.size() == 0:
                continue
            arr = np.asarray(vec, dtype=np.int64) - BASELINE_ADC
            ch = opdet_to_channel(readoutID)
            existing = channel_waveforms.get(ch)
            if existing is None:
                channel_waveforms[ch] = arr.copy()
            else:
                ml = min(existing.size, arr.size)
                existing[:ml] += arr[:ml]

        channel_peaks = [0] * N_CHANNELS
        for ch, wf in channel_waveforms.items():
            peak = int(wf.max())
            if peak > SUMMED_DYNAMIC_RANGE:
                peak = SUMMED_DYNAMIC_RANGE
            elif peak < 0:
                peak = 0
            channel_peaks[ch] = peak
        row['channel_peak_adc'] = channel_peaks
        row['peak_adc'] = max(channel_peaks)
    else:
        row['channel_peak_adc'] = None
        row['peak_adc'] = pd.NA

    return row


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

SIM_DIR = os.path.join(DATA_DIR, f"{LOCATION}_{DATE}_maps", "sim")

# Branches we actually read from the parent tree (gramsg4).
# Everything else is disabled via SetBranchStatus for massive I/O savings.
ACTIVE_PARENT_BRANCHES = ["EventID*", "TrackList*", "LArHits*"]

# Friend trees whose branches we read. Each entry: (stage, tree_name, [branch_patterns])
# Expand this list when new columns need data from additional stages.
ACTIVE_FRIENDS = [
    ("opdetsim", "OpDetSim", ["OpDetWaveforms*"]),
]


def _extract_particle(particle):
    """Extract all events for a single particle. Worker for multiprocessing."""
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

    # Load only friend trees whose branches we actually read.
    has_opdet = False
    for stage, tree_name, _patterns in ACTIVE_FRIENDS:
        friend_path = os.path.join(SIM_DIR, f"{LOCATION}_{particle}_{stage}.root")
        if os.path.exists(friend_path):
            tree.AddFriend(tree_name, friend_path)
            if stage == "opdetsim":
                has_opdet = True
        else:
            print(f"  [DEBUG] {stage} file not found for {particle}, skipping")

    # Disable everything, then re-enable only what we need.
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
        row = _extract_event(tree, ROOT, has_opdet)
        row['particle'] = particle
        rows.append(row)

    print(f"[INFO]   {particle} done ({n_entries} events)", flush=True)
    g4_file.Close()
    return rows


def extract_all_events(
    particles=None,
    force=False,
    output_path=None,
    n_workers=None,
):
    """
    Extract all per-event quantities from ROOT files into a DataFrame.

    Parameters
    ----------
    particles : list of str, optional
        Subset of PARTICLE_DICT keys to process. Default: all particles.
    force : bool
        If True, re-extract even if cached file exists.
    output_path : str, optional
        Override default output path.
    n_workers : int, optional
        Number of parallel processes (one per particle). Default: min(8, n_particles).

    Returns
    -------
    pd.DataFrame
    """
    output = output_path or os.path.join(MAPS_DIR, "pkl", "events.parquet")

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
        for particle in particles:
            all_rows.extend(_extract_particle(particle))
    else:
        from multiprocessing import get_context
        ctx = get_context("spawn")  # spawn avoids ROOT fork issues
        with ctx.Pool(processes=n_workers) as pool:
            for rows in pool.imap_unordered(_extract_particle, particles):
                all_rows.extend(rows)

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Enforce types
    int_cols = ['event_index', 'primary_pdg', 'n_lar_hits',
                'total_scint_photons', 'total_cer_photons']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype('int64')

    if 'peak_adc' in df.columns:
        df['peak_adc'] = df['peak_adc'].astype('Int64')

    # Save
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"[INFO] Saved {len(df)} rows to {output}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-event data from ROOT trees into a parquet DataFrame"
    )
    parser.add_argument(
        "--particles", type=str, nargs="+",
        help="Particles to process (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if cached parquet exists"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output parquet path"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: min(8, n_particles))"
    )
    args = parser.parse_args()

    df = extract_all_events(
        particles=args.particles,
        force=args.force,
        output_path=args.output,
        n_workers=args.workers,
    )

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nParticle counts:")
    print(df['particle'].value_counts().to_string())
    print(f"\nNon-null peak_adc: {df['peak_adc'].notna().sum()}")
    print(f"Non-null dominant_daughter_process: {df['dominant_daughter_process'].notna().sum()}")


if __name__ == "__main__":
    main()
