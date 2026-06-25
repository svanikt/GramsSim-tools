#!/usr/bin/env python3
"""
light_config.py

Reference module for pGRAMS light-sensor layout:
  144 SiPMs (opdets) -> 36 channels -> 9 cells (3x3) across 3 motherboards.

Opdet volumeID / copy-number encoding from pgrams.gdml (5ZZYYXX):
  ZZ tens digit 0 = VUV (IDs 5,000,000-5,001,111)
  ZZ tens digit 1 = VIS (IDs 5,100,000-5,101,111)
  YY = cell_y*4 + sipm_y       (cell_y, sipm_y each in 0..2, 0..3)
  XX = cell_x*4 + sipm_x       (cell_x, sipm_x each in 0..2, 0..3)

Within a cell, VUV/VIS splits along physical y (sipm_y >= 2 is VUV).
A channel groups 4 adjacent SiPMs keyed by (sipm_x//2, sipm_y//2).

The (cell_x, cell_y) -> (TL, BL, TR, BR) channel table below is hardcoded
from the Shaper Ch. Map image, with the orientation:
  image-right = GDML +x, image-up = GDML +y (view from anode to cathode).
"""

N_CELLS = 9
N_CHANNELS = 36
N_SIPMS_PER_CHANNEL = 4
N_SIPMS = 144

NON_TRIGGER_CHANNELS = frozenset({32, 33, 34, 35})

# Image channel map: (cell_x, cell_y) -> (TL, BL, TR, BR).
# intra = 2*sipm_x_half + (1 - sipm_y_half), so TL/TR = sipm_y_half=1 (VUV),
# BL/BR = sipm_y_half=0 (VIS); left/right split by sipm_x_half.
CELL_CHANNELS = {
    (0, 2): (0, 1, 2, 3),       # MB3 left
    (1, 2): (4, 5, 6, 32),      # MB3 mid
    (2, 2): (7, 8, 9, 10),      # MB3 right
    (0, 1): (18, 19, 20, 34),   # MB1 left
    (1, 1): (14, 15, 16, 17),   # MB1 mid
    (2, 1): (11, 12, 13, 33),   # MB1 right
    (0, 0): (21, 22, 23, 24),   # MB2 left
    (1, 0): (25, 26, 27, 35),   # MB2 mid
    (2, 0): (28, 29, 30, 31),   # MB2 right
}

CELL_MB = {
    (cx, 2): "MB3" for cx in (0, 1, 2)
}
CELL_MB.update({(cx, 1): "MB1" for cx in (0, 1, 2)})
CELL_MB.update({(cx, 0): "MB2" for cx in (0, 1, 2)})


def decode_opdet_id(det_id):
    """Decode a 5ZZYYXX opdet ID into cell/SiPM coordinates and VUV/VIS flag."""
    remainder = det_id - 5000000
    is_vis = remainder >= 100000
    if is_vis:
        remainder -= 100000
    yy = remainder // 100
    xx = remainder % 100
    return {
        'cell_x': xx // 4, 'sipm_x': xx % 4,
        'cell_y': yy // 4, 'sipm_y': yy % 4,
        'is_vis': is_vis,
    }


def _intra_index(sipm_x, sipm_y):
    # (TL, BL, TR, BR) = 0, 1, 2, 3
    return 2 * (sipm_x // 2) + (1 - sipm_y // 2)


def opdet_to_channel(det_id):
    """Return channel number 0..35 for a given opdet ID."""
    d = decode_opdet_id(det_id)
    tl_bl_tr_br = CELL_CHANNELS[(d['cell_x'], d['cell_y'])]
    return tl_bl_tr_br[_intra_index(d['sipm_x'], d['sipm_y'])]


def opdet_to_cell(det_id):
    """Return cell index 0..8 (= cell_y*3 + cell_x)."""
    d = decode_opdet_id(det_id)
    return d['cell_y'] * 3 + d['cell_x']


# --- inverse maps, built once at import -------------------------------------

def _build_inverse_maps():
    ch_to_cell_xy = {}
    ch_to_mb = {}
    for (cx, cy), (tl, bl, tr, br) in CELL_CHANNELS.items():
        for ch in (tl, bl, tr, br):
            ch_to_cell_xy[ch] = (cx, cy)
            ch_to_mb[ch] = CELL_MB[(cx, cy)]
    # VUV flag: TL, TR are image-top (sipm_y_half=1) => VUV; BL, BR => VIS.
    ch_is_vuv = {}
    for (tl, bl, tr, br) in CELL_CHANNELS.values():
        ch_is_vuv[tl] = True
        ch_is_vuv[tr] = True
        ch_is_vuv[bl] = False
        ch_is_vuv[br] = False
    return ch_to_cell_xy, ch_to_mb, ch_is_vuv


CHANNEL_TO_CELL_XY, CHANNEL_TO_MB, CHANNEL_IS_VUV = _build_inverse_maps()


# --- trigger scheme ---------------------------------------------------------
# A trigger fires when, within a rolling window of TRIGGER_WINDOW_SIZE
# consecutive trigger-channel indices (0..N_TRIGGER_CHANNELS-1), at least
# TRIGGER_MULTIPLICITY channels cross the summed-ADC threshold.
# Non-trigger channels (32-35) are excluded from the scan.

N_TRIGGER_CHANNELS = N_CHANNELS - len(NON_TRIGGER_CHANNELS)  # 32
TRIGGER_CHANNELS = tuple(ch for ch in range(N_CHANNELS) if ch not in NON_TRIGGER_CHANNELS)
TRIGGER_WINDOW_SIZE = 5   # window covers 5 consecutive channels: 0-4, 1-5, ..., 27-31
TRIGGER_MULTIPLICITY = 2


def trigger_windows():
    """Return a list of channel-index tuples, one per rolling window."""
    return [tuple(range(i, i + TRIGGER_WINDOW_SIZE))
            for i in range(N_TRIGGER_CHANNELS - TRIGGER_WINDOW_SIZE + 1)]


def fires_trigger(channel_peaks, threshold,
                  multiplicity=TRIGGER_MULTIPLICITY,
                  window_size=TRIGGER_WINDOW_SIZE):
    """
    Vectorized trigger evaluator.

    Parameters
    ----------
    channel_peaks : array-like, shape (N, N_CHANNELS) or (N_CHANNELS,)
        Baseline-subtracted summed-ADC peak for each channel. Non-trigger
        channels (32-35) are ignored.
    threshold : int
        Summed-ADC threshold (baseline-subtracted).
    multiplicity : int
        Minimum number of channels within a window that must cross threshold.
    window_size : int
        Rolling window size along the channel axis.

    Returns
    -------
    ndarray of bool, shape (N,) — True if trigger fired in that event.
    If the input is 1D, returns a single bool.
    """
    import numpy as np
    peaks = np.asarray(channel_peaks)
    squeeze = peaks.ndim == 1
    if squeeze:
        peaks = peaks[np.newaxis, :]
    trig = peaks[:, :N_TRIGGER_CHANNELS]
    cross = (trig >= threshold).astype(np.int32)
    # sliding-window sum via cumulative sum
    csum = np.cumsum(cross, axis=1)
    window_counts = csum[:, window_size - 1:].copy()
    window_counts[:, 1:] -= csum[:, :-window_size]
    fired = (window_counts >= multiplicity).any(axis=1)
    return bool(fired[0]) if squeeze else fired


# --- self-test ---------------------------------------------------------------

def _all_opdet_ids():
    for cell_x in range(3):
        for sipm_x in range(4):
            for cell_y in range(3):
                for sipm_y in range(4):
                    xx = cell_x * 4 + sipm_x
                    yy = cell_y * 4 + sipm_y
                    base = 5000000 + 100 * yy + xx
                    # opdetCopyNoOffset: +100000 for sipm_y in {0,1} (VIS),
                    # +0 for sipm_y in {2,3} (VUV).
                    is_vis = sipm_y < 2
                    yield base + (100000 if is_vis else 0)


def _self_test():
    channels_seen = {}
    for det_id in _all_opdet_ids():
        d = decode_opdet_id(det_id)
        ch = opdet_to_channel(det_id)
        assert 0 <= ch < N_CHANNELS, f"channel out of range: {ch}"
        channels_seen.setdefault(ch, []).append(d)

    # 1. Coverage: 36 channels, 4 SiPMs each.
    assert len(channels_seen) == N_CHANNELS, \
        f"expected {N_CHANNELS} channels, got {len(channels_seen)}"
    for ch, sipms in channels_seen.items():
        assert len(sipms) == N_SIPMS_PER_CHANNEL, \
            f"channel {ch} has {len(sipms)} SiPMs, expected {N_SIPMS_PER_CHANNEL}"

    # 2. Orientation invariant: 4 SiPMs of a channel share is_vis.
    for ch, sipms in channels_seen.items():
        flags = {s['is_vis'] for s in sipms}
        assert len(flags) == 1, \
            f"channel {ch} mixes VUV and VIS SiPMs: {sipms}"
        # and matches CHANNEL_IS_VUV
        is_vuv_from_sipms = not next(iter(flags))
        assert CHANNEL_IS_VUV[ch] == is_vuv_from_sipms, \
            f"CHANNEL_IS_VUV[{ch}] disagrees with decoded SiPMs"

    # 3. Table invariant: TL, TR are VUV; BL, BR are VIS.
    for (cx, cy), (tl, bl, tr, br) in CELL_CHANNELS.items():
        assert CHANNEL_IS_VUV[tl] and CHANNEL_IS_VUV[tr], \
            f"cell {(cx, cy)}: TL/TR should be VUV"
        assert not CHANNEL_IS_VUV[bl] and not CHANNEL_IS_VUV[br], \
            f"cell {(cx, cy)}: BL/BR should be VIS"

    n_vuv = sum(1 for v in CHANNEL_IS_VUV.values() if v)

    # 4. Trigger scheme sanity checks.
    import numpy as np
    windows = trigger_windows()
    assert len(windows) == N_TRIGGER_CHANNELS - TRIGGER_WINDOW_SIZE + 1
    assert all(len(w) == TRIGGER_WINDOW_SIZE for w in windows)
    # Single channel over threshold -> no trigger (multiplicity=2)
    peaks = np.zeros((1, N_CHANNELS), dtype=np.int32)
    peaks[0, 3] = 5000
    assert not fires_trigger(peaks, 1000).any(), "single channel should not trigger"
    # Two adjacent channels over threshold -> trigger
    peaks[0, 4] = 5000
    assert fires_trigger(peaks, 1000).any(), "adjacent pair should trigger"
    # Two channels 0 and 6 over threshold (distance 6) -> no trigger
    peaks2 = np.zeros((1, N_CHANNELS), dtype=np.int32)
    peaks2[0, 0] = 5000
    peaks2[0, 6] = 5000
    assert not fires_trigger(peaks2, 1000).any(), "channels 0 and 6 should not co-trigger"
    # Channels 0 and 5 (distance 5) -> trigger (both fit in window {0..5})
    peaks2[0, 6] = 0
    peaks2[0, 5] = 5000
    assert fires_trigger(peaks2, 1000).any(), "channels 0 and 5 should co-trigger"
    # Non-trigger channels (32,33) pair -> no trigger
    peaks3 = np.zeros((1, N_CHANNELS), dtype=np.int32)
    peaks3[0, 32] = 5000
    peaks3[0, 33] = 5000
    assert not fires_trigger(peaks3, 1000).any(), "non-trigger channels should be ignored"

    print(f"[light_config] OK: {N_SIPMS} SiPMs -> {N_CHANNELS} channels "
          f"({n_vuv} VUV, {N_CHANNELS - n_vuv} VIS), "
          f"{len(NON_TRIGGER_CHANNELS)} non-trigger channels. "
          f"Trigger: multiplicity={TRIGGER_MULTIPLICITY}, window={TRIGGER_WINDOW_SIZE}, "
          f"{len(windows)} rolling windows.")


if __name__ == "__main__":
    _self_test()
