# train.py
import os
import numpy as np
import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.pipeline import ensure_dir, build_dataset_from_toi, build_feature_vector
from src.pipeline import build_feature_vector as bfv  # alias
from src.pipeline import run_bls, fetch_lightcurve_by_tic, preprocess_lc
from src.pipeline import phase_fold_arrays
from src.cnn import make_simple_cnn, train_cnn_model

MODELS_DIR = "models"
ensure_dir(MODELS_DIR)

def train_classical(X, y, save_dir=MODELS_DIR):
    X = np.array(X)
    y = np.array(y)
    # simple imputation for NaN features (median)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    joblib.dump(ada, os.path.join(save_dir, "ada_model.joblib"))
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(save_dir, "rf_model.joblib"))
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, os.path.join(save_dir, "knn_model.joblib"))
    print("Classical models trained and saved.")
    # report for Ada
    y_pred = ada.predict(X_test)
    print("AdaBoost classification report:")
    print(classification_report(y_test, y_pred))
    return ada, rf, knn

def build_phase_images(meta_list, img_size=64):
    """
    Given meta list with 'bls' result and 'tic', create simple phase-folded 2D images (resized).
    Returns X_img (N,H,W,1) and y (if 'label' exists in meta entries)
    Note: meta_list should contain entries with keys: 'tic', 'bls', 'label', 'lc' (optional)
    """
    import numpy as np
    from skimage.transform import resize
    X = []
    y = []
    for m in meta_list:
        bls = m.get('bls')
        lc = m.get('lc')
        if bls is None or lc is None:
            continue
        try:
            p = bls.get('period')
            t0 = bls.get('t0')
            phase, flux = phase_fold_arrays(lc, p, t0)
            # make a 1D curve into a 2D image by stacking and resizing
            arr = np.interp(np.linspace(-0.5,0.5,img_size), phase, flux)
            img = np.tile(arr, (img_size,1))  # shape (img_size, img_size)
            # normalize image
            img = (img - np.nanmedian(img)) / (np.nanstd(img) + 1e-12)
            img = resize(img, (img_size, img_size), preserve_range=True)
            X.append(img[..., np.newaxis].astype('float32'))
            if 'label' in m:
                y.append(int(m['label']))
        except Exception:
            continue
    X = np.array(X)
    y = np.array(y) if len(y)>0 else None
    return X, y

def main():
    print("Building dataset (may take time - downloading lightcurves)...")
    X, y, meta = build_dataset_from_toi(sample_rows=300, limit_download=80)
    if len(X) < 10:
        print("Not enough samples downloaded. Increase limit_download or run again.")
        return
    print(f"Downloaded {len(X)} examples.")
    # Train classical models
    ada, rf, knn = train_classical(X, y, save_dir=MODELS_DIR)
    # attach labels & lc to meta for CNN image build (we didn't include lc in build_dataset_from_toi by default,
    # so rebuild meta items with lc where possible - here we'll do a light loop to attach lc for entries that have TIC)
    # Rebuild meta with lightcurves included (best-effort)
    print("Fetching lightcurves for CNN images (best-effort)...")
    meta_with_lc = []
    for i, m in enumerate(meta):
        tic = m.get('tic')
        try:
            lc = fetch_lightcurve_by_tic(tic)
            if lc is None:
                continue
            lc = preprocess_lc(lc)
            m2 = m.copy()
            m2['lc'] = lc
            m2['label'] = y[i]
            meta_with_lc.append(m2)
        except Exception:
            continue
    if len(meta_with_lc) >= 20:
        print(f"Creating images from {len(meta_with_lc)} objects for CNN.")
        X_img, y_img = build_phase_images(meta_with_lc, img_size=64)
        if y_img is not None and len(y_img) >= 20:
            print("Training small CNN (this may be slow without GPU)...")
            model = make_simple_cnn(input_shape=(64,64,1), n_classes=len(np.unique(y_img)))
            train_cnn_model(model, X_img, y_img, epochs=12, batch_size=16, out_path=os.path.join(MODELS_DIR, "cnn_model.h5"))
            print("CNN trained and saved.")
        else:
            print("Not enough labeled images for CNN training.")
    else:
        print("Not enough data with lightcurves to prepare CNN images.")

if __name__ == "__main__":
    main()
