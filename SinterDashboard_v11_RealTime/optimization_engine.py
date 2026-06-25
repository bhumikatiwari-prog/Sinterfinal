import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from prediction_engine import predict_from_row, build_input_row

TARGETS = ['TI', 'RDI', 'RI']


def _solution_quality(ti, rdi, ri):
    """Single compromise score used only to pick ONE headline 'best' trial
    out of the Pareto-optimal set for the KPI gauges / recommended-params
    table. Keep this formula identical to pareto_engine.solution_quality —
    both exist because optimization_engine must not import streamlit-
    dependent pareto_engine, and pareto_engine must not import optuna.
    """
    return (ti - 78) + (ri - 68) - max(0, rdi - 25) * 2


def run_optimization(best_models, sq_raw, n_trials=300):
    levers = ['%FeO', '%CaO', '%MgO', '%SiO2', '%Al2O3', 'Avl. lime', 'Basicity  (B2)', 'MgO/Al2O3']
    levers = [c for c in levers if c in sq_raw.columns]
    ranges = sq_raw[levers].quantile([0.05, 0.95]).T; ranges.columns = ['lo', 'hi']

    def objective(trial):
        params = {col: trial.suggest_float(col, float(ranges.loc[col, 'lo']),
                                                  float(ranges.loc[col, 'hi']))
                  for col in levers}
        row = build_input_row(params, sq_raw)
        preds = predict_from_row(row, best_models)
        # Genuine multi-objective return — let Optuna's NSGA-II sampler do
        # the actual Pareto search, instead of pre-collapsing the 3
        # objectives into one scalar weighted sum. A single fixed-weight
        # scalarization (the previous approach) only ever explores trials
        # clustered around ONE point on the trade-off surface, which is why
        # the old Pareto chart looked like a single dense blob rather than
        # a spread-out front — there was no real front to show.
        return preds['TI'], preds['RDI'], preds['RI']

    study = optuna.create_study(
        directions=['maximize', 'minimize', 'maximize'],   # TI↑, RDI↓, RI↑
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    pareto = pd.DataFrame([{
        'TI':  t.values[0] if t.values else np.nan,
        'RDI': t.values[1] if t.values else np.nan,
        'RI':  t.values[2] if t.values else np.nan,
        **t.params,
    } for t in study.trials]).dropna()

    # Pick one headline "best compromise" trial (for KPI gauges + the
    # recommended-parameters table) from Optuna's own Pareto-optimal set.
    pareto_trials = [t for t in study.best_trials if t.values]
    if pareto_trials:
        best = max(pareto_trials, key=lambda t: _solution_quality(*t.values))
        best_vals, best_params = best.values, best.params
    else:
        # Degenerate fallback (e.g. every trial failed) — use the single
        # best-scoring row in the trials dataframe instead of crashing.
        idx = pareto.apply(lambda r: _solution_quality(r['TI'], r['RDI'], r['RI']), axis=1).idxmax()
        row = pareto.loc[idx]
        best_vals = (row['TI'], row['RDI'], row['RI'])
        best_params = {c: row[c] for c in levers}

    return {
        'best_TI':     best_vals[0],
        'best_RDI':    best_vals[1],
        'best_RI':     best_vals[2],
        'best_params': best_params,
        'best':        {'TI': best_vals[0], 'RDI': best_vals[1], 'RI': best_vals[2], 'params': best_params},
        'pareto':      pareto,
        'ranges':      ranges,
        'levers':      levers,
    }
