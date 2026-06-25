import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, pickle, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb, lightgbm as lgb, catboost as cb
import shap
from feature_engineering import engineer, LEAKAGE_SQ, LEAKAGE_PP, TARGETS
from date_utils import sort_by_date
import pipeline_validation as pv

ARTIFACTS = os.path.join(os.path.dirname(__file__), 'artifacts')

# Default master data file. Previously 'Outlier_Removed_Data.xlsx' — replaced
# with 'Sinter_Data.xlsx' (same two-sheet schema: Process_Parameters_Clean /
# Sinter_Quality_Clean, just a larger, more current export) as the single
# source of truth for the whole pipeline.
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'Sinter_Data.xlsx')


def _clean_pp_sq(pp: pd.DataFrame, sq: pd.DataFrame):
    """Shared cleaning used by both the file-based and dataframe-based
    training entrypoints: sort chronologically, coerce to numeric, drop
    RDI outliers, drop leakage columns.
    """
    pp = pp.copy(); sq = sq.copy()

    # Sort chronologically FIRST. The source sheets are not guaranteed to be
    # stored in date order (verified false for both sheets in this dataset),
    # and engineer()'s rolling/lag features depend on row order matching
    # calendar order — this also re-sorts defensively inside engineer()
    # itself, but doing it here too means sq_raw/pp_raw (used directly by
    # the Process Monitoring & SPC charts) are stored in correct order.
    if 'Date' in pp.columns:
        pp = sort_by_date(pp, 'Date')
    if 'DATE' in sq.columns:
        sq = sort_by_date(sq, 'DATE')

    for c in pp.columns:
        if c != 'Date': pp[c] = pd.to_numeric(pp[c], errors='coerce')
    for c in sq.columns:
        if c != 'DATE': sq[c] = pd.to_numeric(sq[c], errors='coerce')

    # Outlier filter on RDI — only drop rows where RDI is an actual outlier
    # (>=35), not rows where RDI happens to be missing for that day. A row
    # missing RDI can still be valid training data for TI/RI (per-target
    # dropna happens later in prep_xy); dropping it here purely because RDI
    # is NaN was discarding usable rows.
    if 'RDI' in sq.columns:
        sq = sq[sq['RDI'].isna() | (sq['RDI'] < 35)].copy()

    pp.drop(columns=[c for c in LEAKAGE_PP if c in pp.columns], inplace=True, errors='ignore')
    sq.drop(columns=[c for c in LEAKAGE_SQ if c in sq.columns], inplace=True, errors='ignore')
    return pp, sq


def load_clean_data(path):
    """File-based loader — reads the two named sheets from the master Excel file."""
    pp = pd.read_excel(path, sheet_name='Process_Parameters_Clean')
    sq = pd.read_excel(path, sheet_name='Sinter_Quality_Clean')
    return _clean_pp_sq(pp, sq)


def prep_xy(df, target, thresh=0.50):
    others = [t for t in TARGETS if t != target]
    drop = [c for c in ['DATE','Date']+others if c in df.columns]
    y = df[target].dropna()
    X = df.loc[y.index].drop(columns=drop+[target], errors='ignore')
    X = X.dropna(axis=1, thresh=int((1-thresh)*len(X)))
    return X, y


def _coerce_shap_shape(sv, n_rows, n_feat):
    """Pure shape-fixing logic (kept separate from the try/except below so
    it's directly unit-testable without needing the `shap` package)."""
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:                     # (n_rows, n_features, n_outputs) — some wrappers
        sv = sv[:, :, 0]
    if sv.shape[1] == n_feat + 1:        # extra bias/expected-value column (seen with CatBoost)
        sv = sv[:, :n_feat]
    elif sv.shape[1] == n_feat - 1:      # one short — pad rather than crash
        sv = np.hstack([sv, np.zeros((n_rows, 1))])
    if sv.shape != (n_rows, n_feat):
        raise ValueError(f'SHAP output shape {sv.shape} != expected {(n_rows, n_feat)}')
    return sv


def safe_shap_values(model, X, model_name=''):
    """Compute SHAP values defensively.

    Some library/version combinations (notably CatBoost with certain `shap`
    releases) return an array with one extra column — a bias/expected-value
    term — instead of exactly len(X.columns) columns. Left unhandled, that
    silently misaligns every downstream feature lookup, or throws an
    IndexError the moment the importance ranking indexes past the real
    feature count. This wraps the computation so the rest of the pipeline
    can always assume sv.shape == (n_rows, n_features), and falls back to
    the model's built-in feature_importances_ (broadcast across rows) if
    SHAP genuinely can't be computed for this model at all.
    """
    n_rows, n_feat = len(X), X.shape[1]
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):         # multi-output models
            sv = sv[0]
        return _coerce_shap_shape(sv, n_rows, n_feat)
    except Exception as e:
        print(f'[!] SHAP failed for {model_name}: {e}. Falling back to feature_importances_.')
        if hasattr(model, 'feature_importances_'):
            fi = np.asarray(model.feature_importances_, dtype=float)
            return np.tile(fi, (n_rows, 1))
        return np.zeros((n_rows, n_feat))


def _run_training(pp, sq, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    sq_fe  = engineer(sq, is_pp=False)
    pp_fe  = engineer(pp, is_pp=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_models, all_results, shap_data = {}, {}, {}

    CANDIDATES = {
        'RandomForest': RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=-1),
        'XGBoost':      xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.04,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.1,random_state=42,n_jobs=-1,verbosity=0),
        'LightGBM':     lgb.LGBMRegressor(n_estimators=400,max_depth=6,learning_rate=0.04,num_leaves=35,subsample=0.8,colsample_bytree=0.75,min_child_samples=10,random_state=42,n_jobs=-1,verbose=-1),
        'CatBoost':     cb.CatBoostRegressor(iterations=400,depth=6,learning_rate=0.04,l2_leaf_reg=3.0,random_seed=42,verbose=0),
    }

    for target in TARGETS:
        X, y = prep_xy(sq_fe, target)
        if len(y) == 0 or X.shape[1] == 0:
            raise ValueError(f"No usable rows/features for target '{target}' — check that the "
                              f"Sinter Quality sheet actually contains '{target}' values.")
        imp  = SimpleImputer(strategy='median')
        Xi   = imp.fit_transform(X); Xdf = pd.DataFrame(Xi, columns=X.columns)
        results = {}
        for name, model in CANDIDATES.items():
            yp = cross_val_predict(model, Xdf, y, cv=kf, n_jobs=-1)
            results[name] = {
                'R2':   round(r2_score(y,yp),4),
                'RMSE': round(np.sqrt(mean_squared_error(y,yp)),4),
                'MAE':  round(mean_absolute_error(y,yp),4),
                'MAPE': round(np.mean(np.abs((y.values-yp)/y.values.clip(min=1e-6)))*100,3),
            }
        best_name = max(results, key=lambda m: results[m]['R2'])
        best_m = CANDIDATES[best_name]; best_m.fit(Xdf, y)

        # SHAP — defensive against dimension mismatches / unsupported models
        sv = safe_shap_values(best_m, Xdf, model_name=f'{target}/{best_name}')
        mean_abs = np.abs(sv).mean(axis=0)
        top_n    = min(15, len(mean_abs))
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]

        all_results[target]  = results
        best_models[target]  = {'model':best_m,'imputer':imp,'features':X.columns.tolist(),
                                 'best_algo':best_name,'metrics':results[best_name]}
        shap_data[target]    = {'sv':sv,'feats':X.columns.tolist(),'X':Xdf,
                                 'top_feats':[X.columns[i] for i in top_idx],
                                 'top_vals':mean_abs[top_idx]}

    # Feature importance from best model per target
    feat_imp = {}
    for target in TARGETS:
        info = best_models[target]
        m = info['model']
        if hasattr(m,'feature_importances_'):
            fi = pd.Series(m.feature_importances_, index=info['features']).nlargest(15)
        else:
            fi = pd.Series(shap_data[target]['top_vals'], index=shap_data[target]['top_feats'])
        feat_imp[target] = fi

    # Percentage-contribution table (Top 10, normalised over the FULL
    # importance vector so it's comparable across model families) — used by
    # the dashboard's Feature Importance tab and by the validation framework.
    feat_imp_pct = {
        t: pv.pct_contribution(best_models[t]['model'], best_models[t]['features'], top_n=10)
        for t in TARGETS
    }

    # Correlation matrix
    corr_cols = [c for c in sq_fe.columns if c not in ['DATE','Date']]
    corr = sq_fe[corr_cols].corr()

    artifacts = {
        'best_models':   best_models,
        'all_results':   all_results,
        'shap_data':     shap_data,
        'feat_imp':      feat_imp,
        'feat_imp_pct':  feat_imp_pct,
        'sq_fe':         sq_fe,
        'pp_fe':         pp_fe,
        'sq_raw':        sq,
        'pp_raw':        pp,
        'corr':          corr,
    }

    # Automated validation framework — run every time, surface failures loudly.
    report = pv.run_full_validation(artifacts)
    artifacts['validation_report'] = report

    with open(os.path.join(artifacts_dir,'artifacts.pkl'),'wb') as f:
        pickle.dump(artifacts, f)

    print(f"[✓] Training complete. Artifacts saved to {artifacts_dir}/artifacts.pkl")
    for t in TARGETS:
        r = all_results[t][best_models[t]['best_algo']]
        print(f"  {t}: {best_models[t]['best_algo']}  R²={r['R2']}  RMSE={r['RMSE']}  MAPE={r['MAPE']}%")
    pv.print_report(report)
    if not report['all_ok']:
        print(f"[!] {report['n_fail']} validation check(s) failed — see report above. "
              f"Artifacts were still saved; review before trusting the dashboard output.")
    return artifacts


def train(data_path=None, artifacts_dir=ARTIFACTS):
    """CLI / file-based entrypoint — reads the master Excel file (two sheets:
    Process_Parameters_Clean, Sinter_Quality_Clean)."""
    data_path = data_path or DEFAULT_DATA_PATH
    pp, sq = load_clean_data(data_path)
    return _run_training(pp, sq, artifacts_dir)


def train_from_dataframes(pp_raw: pd.DataFrame, sq_raw: pd.DataFrame, artifacts_dir=ARTIFACTS):
    """Programmatic entrypoint: retrain directly from in-memory raw
    Process Parameters / Sinter Quality dataframes (e.g. produced by
    data_ingestion.process_pp_upload / process_sq_upload + merging) without
    needing a round-trip through an Excel file on disk.
    """
    pp, sq = _clean_pp_sq(pp_raw, sq_raw)
    return _run_training(pp, sq, artifacts_dir)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    train(path)
