# src/pipeline.py
import os
import io
import time
import numpy as np
import pandas as pd
import requests
import urllib.parse
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk

EXO_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# -------------------------
# ID resolution
# -------------------------
def toi_to_tic_id(toi_id, timeout=20):
    """
    Map a TOI id (e.g., 'TOI-1234.01' or 'TOI-1234') to TIC using NASA Exoplanet Archive.
    Returns string TIC or None.
    """
    toi_id = str(toi_id).strip().upper()
    where_clause = f"toi like '{toi_id}'"
    params = f"table=TOI&select=toi,tid,tfopwg_disp&where={urllib.parse.quote_plus(where_clause)}&format=csv"
    url = EXO_API_BASE + params
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if 'tid' in df.columns and df.shape[0] > 0 and not pd.isna(df['tid'].iloc[0]):
            return str(int(df['tid'].iloc[0]))
    except Exception:
        return None
    return None

def generic_resolve_to_tic(obj_id):
    """
    Resolve strings like 'TIC 12345', '12345', 'TOI-1234.01', 'TOI-1234' -> TIC digits string
    """
    s = str(obj_id).strip()
    s_up = s.upper()
    if s_up.startswith('TIC'):
        digits = ''.join([c for c in s if c.isdigit()])
        return digits if digits else None
    if s_up.startswith('TOI'):
        return toi_to_tic_id(s)
    # if pure digits -> assume TIC
    if s.isdigit():
        return s
    # try TOI fallback
    return toi_to_tic_id(s)

# -------------------------
# Lightcurve fetch & preprocess
# -------------------------
def fetch_lightcurve_by_tic(tic_id, mission_prefer='TESS', max_attempts=2):
    """
    Uses lightkurve to search and download lightcurve file(s) for a TIC id.
    Returns a LightCurve (stitched) or None.
    """
    target = f"TIC {tic_id}"
    # try missions in preference order
    missions = [mission_prefer, 'Kepler', 'K2']
    for m in missions:
        try:
            search = lk.search_lightcurvefile(target, mission=m)
            if len(search) == 0:
                continue
            # download all available LC files (may be multiple sectors)
            lcf_collection = search.download_all()
            if not lcf_collection:
                continue
            lcs = []
            for lcf in lcf_collection:
                # prefer PDCSAP_FLUX, fallback to SAP_FLUX
                if hasattr(lcf, 'PDCSAP_FLUX') and lcf.PDCSAP_FLUX is not None:
                    lcs.append(lcf.PDCSAP_FLUX)
                elif hasattr(lcf, 'SAP_FLUX') and lcf.SAP_FLUX is not None:
                    lcs.append(lcf.SAP_FLUX)
            if len(lcs) == 0:
                continue
            stitched = lcs[0]
            for extra in lcs[1:]:
                try:
                    stitched = stitched.append(extra)
                except Exception:
                    # minor mismatch; skip
                    pass
            # try to remove NaNs and normalize
            try:
                stitched = stitched.remove_nans().normalize()
            except Exception:
                pass
            return stitched
        except Exception:
            continue
    return None

def preprocess_lc(lc, window_length=401):
    """
    Detrend / flatten a lightcurve (lightkurve LightCurve)
    """
    try:
        flat = lc.flatten(window_length=window_length)
        return flat.remove_nans()
    except Exception:
        # fallback: simple normalization and NaN removal
        try:
            lc = lc.remove_nans()
            flux = lc.flux / np.nanmedian(lc.flux) - 1.0
            lc2 = lc.copy()
            lc2.flux = flux
            return lc2
        except Exception:
            return lc

# -------------------------
# BLS detection & simple features
# -------------------------
def run_bls(lc, min_period=0.5, max_period=50.0):
    """
    Runs BoxLeastSquares on a detrended lightcurve.
    Returns dict with period, duration, t0, depth, snr and periodogram object.
    """
    t = np.array(lc.time.value)
    y = np.array(lc.flux.value)
    if len(t) < 10:
        return None
    try:
        model = BoxLeastSquares(t * u.day, y)
        # use autopower with a reasonable guess for duration (0.1 day)
        try:
            periodogram = model.autopower(0.2 * u.day)
        except Exception:
            periods = np.linspace(min_period, max_period, 5000) * u.day
            durations = np.linspace(0.01, 0.5, 10) * u.day
            periodogram = model.power(periods, durations)
        best_idx = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx].value)
        best_duration = float(periodogram.duration[best_idx].value)
        best_t0 = float(periodogram.transit_time[best_idx].value)
        # compute stats where possible (BoxLeastSquares has compute_stats in modern astropy)
        try:
            stats = model.compute_stats(periodogram.period[best_idx], periodogram.duration[best_idx], periodogram.transit_time[best_idx])
            depth = float(stats.get('depth', np.nan))
            snr = float(stats.get('snr', np.nan))
        except Exception:
            depth = float(np.nanmedian(y) - np.nanmin(y))
            snr = float(np.nan)
            stats = {}
        return {'period': best_period, 'duration': best_duration, 't0': best_t0, 'depth': depth, 'snr': snr, 'periodogram': periodogram, 'stats': stats}
    except Exception:
        return None

def phase_fold_arrays(lc, period, t0, bins=300):
    """
    Return (phase, flux) arrays folded on period and centered on transit.
    """
    phase = ((lc.time.value - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], lc.flux.value[order]

def build_feature_vector(bls_result):
    """
    Build a simple numerical feature vector for classical ML:
    [period, duration, depth, snr, depth/duration (shape metric)]
    """
    if bls_result is None:
        return [np.nan]*5
    p = bls_result.get('period', np.nan)
    d = bls_result.get('duration', np.nan)
    depth = bls_result.get('depth', np.nan)
    snr = bls_result.get('snr', np.nan)
    shape = (depth / d) if (d and not np.isnan(d) and d != 0) else np.nan
    return [p, d, depth, snr, shape]

# -------------------------
# Dataset builder (small demo)
# -------------------------
def fetch_sample_toi_table(rows=100, timeout=30):
    """
    Fetch a sample TOI table CSV (returns pandas DataFrame). Limited rows.
    """
    params = f"table=TOI&select=toi,tid,tfopwg_disp,toi_disposition&format=csv&rows={rows}"
    url = EXO_API_BASE + params
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Exoplanet archive error {r.status_code}")
    df = pd.read_csv(io.StringIO(r.text))
    return df

def build_dataset_from_toi(sample_rows=200, limit_download=100):
    """
    Builds a labeled dataset by:
     - fetching TOI table snapshot,
     - resolving TIC ids,
     - downloading lightcurves (up to 'limit_download' objects),
     - running BLS and extracting features,
     - mapping dispositions to labels {0:CONFIRMED,1:CANDIDATE,2:FALSE_POSITIVE}
    Returns X (list of feature vectors), y (labels), meta (list of dicts)
    """
    df = fetch_sample_toi_table(rows=sample_rows)
    X = []
    y = []
    meta = []
    count = 0
    for idx, row in df.iterrows():
        if count >= limit_download:
            break
        toi = row.get('toi')
        tid = row.get('tid')
        disp = row.get('tfopwg_disp') if 'tfopwg_disp' in row else row.get('toi_disposition', None)
        if pd.isna(tid):
            continue
        tic = str(int(tid))
        try:
            lc = fetch_lightcurve_by_tic(tic)
            if lc is None:
                continue
            lc = preprocess_lc(lc)
            bls = run_bls(lc)
            fv = build_feature_vector(bls)
            # map disposition -> label
            label = 2  # default false positive
            if isinstance(disp, str):
                d_up = disp.upper()
                if 'CONFIRMED' in d_up or 'CP' in d_up or 'KP' in d_up:
                    label = 0
                elif 'CANDIDATE' in d_up or 'PC' in d_up:
                    label = 1
                else:
                    label = 2
            else:
                label = 2
            X.append(fv)
            y.append(label)
            meta.append({'toi': toi, 'tic': tic, 'disp': disp, 'bls': bls})
            count += 1
            # small sleep to be polite to MAST/API
            time.sleep(0.5)
        except Exception:
            continue
    return X, y, meta
