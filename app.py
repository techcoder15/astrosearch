# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import urllib.parse
import os
from datetime import datetime
import matplotlib.pyplot as plt

# astronomy libs
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

# ML libs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib

# optional: simple CNN
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -------------------------
# Helpers: Exoplanet Archive mapping
# -------------------------
EXO_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"

def toi_to_tic_id(toi_id):
    """Map a TOI (e.g. 'TOI-1234.01' or 'TOI-1234') to TIC ID using the Exoplanet Archive API.
       Returns TIC (string) or None.
    """
    toi_id = toi_id.strip().upper()
    # build a safe URL. We use the TOI table and select the 'toi' and 'tid' (TIC id) columns
    where_clause = f"toi like '{toi_id}'"
    params = f"table=TOI&select=toi,tid&where={urllib.parse.quote_plus(where_clause)}&format=csv"
    url = EXO_API_BASE + params
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] >= 1 and 'tid' in df.columns:
            # tid is the TIC column
            return str(int(df['tid'].values[0]))
    except Exception:
        pass
    return None

def generic_resolve_to_tic(obj_id):
    s = str(obj_id).strip()
    if s.upper().startswith('TIC'):
        # parse digits
        digits = ''.join(ch for ch in s if ch.isdigit())
        return digits
    if s.upper().startswith('TOI'):
        return toi_to_tic_id(s)
    # KOI mapping or other formats can be added here.
    return None

# -------------------------
# Light curve fetch & processing
# -------------------------
def fetch_lightcurve_by_tic(tic_id, mission_prefer='TESS'):
    """Use lightkurve to search/download lightcurve(s) for a TIC ID.
       Returns a LightCurve object (stitched) or None.
    """
    target = f"TIC {tic_id}"
    # try TESS first
    try:
        search = lk.search_lightcurvefile(target, mission='TESS')
        if len(search) > 0:
            lcf_collection = search.download_all()
            # extract PDCSAP_FLUX from each lightcurve and stitch
            lcs = []
            for lcf in lcf_collection:
                try:
                    lc = lcf.PDCSAP_FLUX
                    lcs.append(lc)
                except Exception:
                    try:
                        lc = lcf.SAP_FLUX
                        lcs.append(lc)
                    except Exception:
                        pass
            if len(lcs) == 0:
                return None
            stitched = lcs[0]
            for extra in lcs[1:]:
                try:
                    stitched = stitched.append(extra)
                except Exception:
                    pass
            try:
                return stitched.remove_nans().normalize()
            except Exception:
                return stitched
    except Exception as e:
        # fallback: try Kepler
        pass

    # try Kepler / K2 as fallback
    try:
        search2 = lk.search_lightcurvefile(target, mission='Kepler')
        if len(search2) > 0:
            lcf_collection = search2.download_all()
            lcs = [lcf.PDCSAP_FLUX for lcf in lcf_collection if hasattr(lcf, 'PDCSAP_FLUX')]
            if len(lcs) == 0:
                return None
            stitched = lcs[0]
            for extra in lcs[1:]:
                stitched = stitched.append(extra)
            return stitched.remove_nans().normalize()
    except Exception:
        return None

def preprocess_lc(lc, window_length=401):
    """Return a detrended (flattened) light curve ready for BLS."""
    try:
        flat = lc.flatten(window_length=window_length)
        return flat.remove_nans()
    except Exception:
        # fallback: simple moving median detrend
        y = lc.flux
        # a naive detrend
        y_med = pd.Series(y).rolling(window=101, center=True, min_periods=1).median().fillna(method='bfill').fillna(method='ffill').values
        detrended = lc.copy()
        detrended.flux = lc.flux / y_med - 1.0
        return detrended

# -------------------------
# BLS detection & features
# -------------------------
def run_bls(lc, min_period=0.5, max_period=50., frequency_grid=20000):
    """Run BoxLeastSquares and return best period, duration, depth, SNR and stats dict."""
    t = np.array(lc.time.value)
    y = np.array(lc.flux.value)
    # convert to astropy quantities
    t_q = t * u.day
    model = BoxLeastSquares(t_q, y)
    # autopower uses a conservative duration guess; we pick a q_range (fractional durations)
    # use autopower (efficient)
    try:
        # choose a minimum transit duration fraction (e.g. 0.01 days -> 0.5 - but we pass duration in days to autopower)
        periodogram = model.autopower(0.2 * u.day)  # autopower with a guess duration (will compute grid)
    except Exception:
        # fallback: explicit grid
        periods = np.linspace(min_period, max_period, 5000) * u.day
        durations = np.linspace(0.05, 0.5, 10) * u.day
        periodogram = model.power(periods, durations)

    # find best peak
    best_idx = np.argmax(periodogram.power)
    best_period = periodogram.period[best_idx].value
    best_duration = periodogram.duration[best_idx].value
    # transit time
    best_t0 = periodogram.transit_time[best_idx].value
    # compute stats
    stats = model.compute_stats(periodogram.period[best_idx], periodogram.duration[best_idx], periodogram.transit_time[best_idx])
    # pick depth, snr if available
    depth = stats.get('depth', np.nan)
    snr = stats.get('snr', np.nan)
    # number of transits seen (estimate)
    ntrans = stats.get('n_trans', np.nan) if 'n_trans' in stats else np.sum((t > best_t0) & (t < best_t0 + best_period * 10)) # crude
    return {
        'period': float(best_period),
        'duration': float(best_duration),
        't0': float(best_t0),
        'depth': float(depth),
        'snr': float(snr),
        'stats': stats,
        'periodogram': periodogram
    }

def phase_fold_lc(lc, period, t0, bins=300):
    """Return folded x,y arrays (phase) for plotting or CNN image creation."""
    phase = ((lc.time.value - t0 + 0.5 * period) % period) / period - 0.5
    # sort by phase
    order = np.argsort(phase)
    return phase[order], lc.flux.value[order]

# -------------------------
# ML training & inference (classical)
# -------------------------
def build_feature_vector_from_results(bls_result):
    # simple vector: period, duration, depth, snr
    return [bls_result['period'], bls_result['duration'], bls_result['depth'], bls_result['snr']]

def train_classical_models(X, y, outdir='models'):
    os.makedirs(outdir, exist_ok=True)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {}
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    joblib.dump(ada, os.path.join(outdir, 'ada_model.joblib'))
    models['ada'] = ada

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(outdir, 'rf_model.joblib'))
    models['rf'] = rf

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, os.path.join(outdir, 'knn_model.joblib'))
    models['knn'] = knn

    # report on test set with Ada as example
    y_pred = ada.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    return models, report

def load_models(models_dir='models'):
    out = {}
    for name in ['ada', 'rf', 'knn']:
        p = os.path.join(models_dir, f'{name}_model.joblib')
        if os.path.exists(p):
            out[name] = joblib.load(p)
    return out

# -------------------------
# Simple CNN training for images (optional)
# -------------------------
def make_cnn_model(input_shape=(64,64,1), n_classes=2):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout='wide', page_title='ExoML - demo')

st.title("ExoML — fetch TESS/TOI by ID, run BLS + AdaBoost/RF/KNN (starter)")

st.markdown("""
**How to use:** upload a CSV that contains a single column of IDs (TIC / TOI / KOI).  
This demo will map TOI → TIC (via NASA Exoplanet Archive), download light curves (Lightkurve/MAST), run BLS, extract features and run classical ML models (AdaBoost etc).  
""")

uploaded = st.file_uploader("Upload CSV with IDs (one column of TIC / TOI / KOI)", type=['csv','txt'])
if uploaded:
    ids_df = pd.read_csv(uploaded, header=None)
    ids = ids_df.iloc[:,0].astype(str).tolist()
    st.write(f"Found {len(ids)} IDs. Showing first 10:", ids[:10])
else:
    ids = []
    manual = st.text_input("Or paste a single ID (e.g. TOI-1234.01 or TIC 123456789):")
    if manual:
        ids = [manual.strip()]

run_button = st.button("Fetch & Analyze IDs")

models = load_models('models')
if models:
    st.sidebar.success(f"Found pretrained models: {', '.join(models.keys())}")

if run_button and len(ids) > 0:
    results = []
    for obj in ids:
        st.info(f"Processing {obj} ...")
        with st.spinner(f"Resolving {obj} → TIC and fetching lightcurve..."):
            tic = generic_resolve_to_tic(obj)
            if tic is None:
                st.warning(f"Could not resolve {obj} to TIC automatically.")
                results.append({'id': obj, 'error': 'resolve_failed'})
                continue
            st.write(f"{obj} → TIC {tic}")
            lc = fetch_lightcurve_by_tic(tic)
            if lc is None:
                st.warning(f"No lightcurve found for TIC {tic}")
                results.append({'id': obj, 'tic': tic, 'error': 'lc_not_found'})
                continue
            # preprocess
            lc_clean = preprocess_lc(lc)
            # run bls
            bls = run_bls(lc_clean)
            feature_vec = build_feature_vector_from_results(bls)
            # prediction
            pred_text = "No model"
            explain_text = ""
            if 'ada' in models:
                X = np.array(feature_vec).reshape(1,-1)
                pred = models['ada'].predict(X)[0]
                proba = models['ada'].predict_proba(X)[0]
                pred_text = f"{pred}  (AdaBoost prob={proba.max():.3f})"
                # feature importance
                if hasattr(models['ada'], 'feature_importances_'):
                    fi = models['ada'].feature_importances_
                    explain_text = f"AdaBoost feature importances (period,duration,depth,snr) = {np.round(fi,3).tolist()}"
                else:
                    explain_text = "AdaBoost model available (no feature_importances_)."
            else:
                # fallback heuristic
                if (bls['snr'] is not None and bls['snr']>9) and (bls['depth']>0.0005):
                    pred_text = "Likely planet (heuristic)"
                    explain_text = f"SNR={bls['snr']:.2f}, depth={bls['depth']:.5f}"
                else:
                    pred_text = "Not significant / maybe false positive (heuristic)"
                    explain_text = f"SNR={bls['snr']:.2f}, depth={bls['depth']:.5f}"

            # plots: lightcurve and folded
            st.subheader(f"Results for {obj} (TIC {tic})")
            st.write("Prediction:", pred_text)
            st.write("Why (short):", explain_text)
            # show raw lc
            fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=False)
            ax[0].scatter(lc_clean.time.value, lc_clean.flux.value, s=1)
            ax[0].set_title("Detrended flux")
            ax[0].set_xlabel("Time (BTJD)")
            ax[0].set_ylabel("Normalized flux")
            # phase-fold
            phase, flux = phase_fold_lc(lc_clean, bls['period'], bls['t0'])
            ax[1].scatter(phase, flux, s=3)
            ax[1].set_xlim(-0.1,0.1)
            ax[1].set_title(f"Phase-folded (P={bls['period']:.4f} d, depth={bls['depth']:.2e})")
            ax[1].set_xlabel("Phase")
            st.pyplot(fig)

            # show periodogram
            per = bls['periodogram']
            fig2, ax2 = plt.subplots()
            ax2.plot(per.period, per.power)
            ax2.set_xscale('log')
            ax2.set_xlabel("Period (days)")
            ax2.set_ylabel("Power")
            ax2.set_title("BLS periodogram")
            st.pyplot(fig2)

            results.append({'id':obj,'tic':tic,'bls':bls,'predict':pred_text,'explain':explain_text})

    # show summary table
    rows = []
    for r in results:
        row = {'id': r.get('id'), 'tic': r.get('tic'), 'predict': r.get('predict'), 'explain': r.get('explain')}
        rows.append(row)
    st.dataframe(pd.DataFrame(rows))

# -------------------------
# Training section
# -------------------------
st.sidebar.header("Training / Models")
if st.sidebar.button("Show training guidance"):
    st.sidebar.markdown("""
### To train models:
1. Use Exoplanet Archive (TOI/KOI tables) to download a labeled list (CONFIRMED / CANDIDATE / FALSE POSITIVE).  
   Example API docs: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
2. For each labeled object, map to TIC and download a lightcurve (or many sectors). Extract BLS features (period, duration, depth, snr).
3. Build X (features) and y (labels), then call `train_classical_models(X,y)`. Models are saved to `models/`.
4. Upload `models/` to your server or include in repo and the app will load them automatically.
    """)

if st.sidebar.button("Train on small sample (demo)"):
    st.info("This will try to build a tiny demo training set by downloading up to 20 TOIs with dispositions. This may take time.")
    # small demo: fetch 20 TOIs from Exoplanet Archive where tfopwg_disp is something
    try:
        # fetch a short sample of TOIs (most recent)
        url = EXO_API_BASE + "table=TOI&select=toi,tid,tfopwg_disp&format=csv&order=rowupdate&rows=50"
        r = requests.get(url, timeout=30)
        df = pd.read_csv(io.StringIO(r.text))
        # take those with tfopwg_disp available
        df = df[df['tfopwg_disp'].notnull()].head(20)
        st.write("Sample TOIs:", df.head())
        X = []
        y = []
        for idx, rr in df.iterrows():
            toi = rr['toi']
            disp = rr['tfopwg_disp']
            tic = str(int(rr['tid'])) if not np.isnan(rr['tid']) else None
            if tic is None:
                continue
            lc = fetch_lightcurve_by_tic(tic)
            if lc is None:
                continue
            lc_clean = preprocess_lc(lc)
            bls = run_bls(lc_clean)
            fv = build_feature_vector_from_results(bls)
            X.append(fv)
            # map dispositions -> label
            lab = 2  # default false positive
            if str(disp).upper().strip() in ['CP','KP','CONFIRMED','Kp','CP=confirmed']:
                lab = 0  # confirmed -> 0
            elif str(disp).upper().strip() in ['PC','CANDIDATE','PC=planetary candidate','PC ']:
                lab = 1
            else:
                lab = 2
            y.append(lab)
        if len(X) < 6:
            st.error("Not enough samples downloaded. Try again or increase sample size.")
        else:
            models_trained, report = train_classical_models(X, y, outdir='models')
            st.text("Training report (AdaBoost on test set):")
            st.text(report)
            st.success("Models saved in ./models")
    except Exception as e:
        st.error(f"Training failed: {e}")

st.markdown("---")
st.markdown("**Notes & next steps:** \n- This starter uses a **small** set of features from BLS for classical ML. For production: add many more features (odd/even depth differences, centroid offset flags, secondary eclipse check, local noise metrics, stellar parameters like Tmag/Teff, transit shape diagnostics). \n- For CNN: build a balanced dataset of phase-folded images (same scaling) and train on a GPU or offline. \n- Always cache downloads to avoid repeated MAST queries.")
