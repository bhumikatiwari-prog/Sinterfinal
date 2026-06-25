import numpy as np, pandas as pd
from feature_engineering import engineer

TARGETS = ['TI','RDI','RI']
BF_TARGETS = {'TI': 78.0, 'RDI': 25.0, 'RI': 68.0}
BF_DIR     = {'TI': '>=',  'RDI': '<=',  'RI': '>='}

FEAT_LABELS = {
    'pctFeO_r14':'FeO 14d-avg','MgO_Al2O3_r14':'MgO/Al₂O₃ 14d-avg',
    'pctFeO_r7':'FeO 7d-avg','pctFeO_r3':'FeO 3d-avg',
    '%K2O':'%K₂O','Al2O3_x_B2':'Al₂O₃×Basicity','pctAl2O3_r14':'Al₂O₃ 14d-avg',
    'Basicity__B2_r14':'Basicity 14d-avg','%FeO':'%FeO','pctCaO_r14':'CaO 14d-avg',
    'pctMgO_r14':'MgO 14d-avg','MgO_Al2O3_r':'MgO/Al₂O₃','FeO_TFe':'FeO/TFe',
    'B2_calc':'Basicity B2','B2_x_FeO':'Basicity×FeO','Gangue_load':'Gangue Load',
    'B4':'Basicity B4','% P':'%P','%Al2O3':'%Al₂O₃','pctAl2O3_r7':'Al₂O₃ 7d-avg',
    'Basicity__B2_r3':'Basicity 3d-avg','pctCaO_r7':'CaO 7d-avg','pctMgO_r7':'MgO 7d-avg',
    'MgO_Al2O3__r14':'MgO/Al₂O₃ 14d-avg','pctCaO_r3':'CaO 3d-avg',
}

MET_INTERP = {
    'TI': {
        'pctFeO_r14':   ('FeO 14d trend controls magnetite density → structural strength ↑','Maintain FeO at 10.5–11.5%'),
        'MgO_Al2O3_r14':('High MgO/Al₂O₃ → periclase dominance → stronger matrix','Target MgO/Al₂O₃ > 0.65'),
        '%K2O':         ('Alkalis weaken sinter lattice → TI deteriorates','Control K₂O < 0.08% via ore blend'),
        'Al2O3_x_B2':   ('Al₂O₃ at low basicity weakens bonding phases','Pair B2 ≥ 2.0 with Al₂O₃ < 3.3%'),
        'pctAl2O3_r14': ('Sustained high Al₂O₃ degrades calcium ferrites','Blend control to reduce Al₂O₃ trend'),
    },
    'RDI': {
        'Basicity__B2_r14':('High sustained basicity → CaFe₂O₄ → resists low-T degradation','Maintain 14d-avg B2 ≥ 2.05'),
        '%FeO':            ('Dense FeO structure → less cracking during BF reduction','Optimal FeO 10.5–11.5%'),
        'pctCaO_r14':      ('CaO trend drives calcium ferrite formation','Stable CaO; avoid ±0.5% daily swings'),
        'Al2O3_x_B2':      ('Al₂O₃ × low B2 synergistically raises RDI','Avoid high-Al ore when basicity dips'),
        'pctFeO_r14':      ('FeO trend determines oxidation state → RDI sensitivity','Rolling avg more predictive than daily'),
    },
    'RI': {
        'pctFeO_r14':   ('Higher FeO → more magnetite → slower reduction → RI ↓','Keep FeO < 10.5% for best RI'),
        'Al2O3_x_B2':   ('SFCA phases with Al₂O₃ have good reducibility → RI ↑','Moderate Al₂O₃ with balanced basicity'),
        'pctAl2O3_r14': ('Al₂O₃ trend affects SFCA mineralogy → RI','3.0–3.5% Al₂O₃ optimal for RI'),
        'pctMgO_r14':   ('MgO stabilises structure during reduction','MgO 2.0–2.4% optimal for RI'),
        'pctCaO_r14':   ('CaO promotes SFCA → well-reducible phases','Stable CaO dosing critical for RI'),
    }
}

def predict_from_row(row_dict, best_models):
    results = {}
    for target in TARGETS:
        info = best_models[target]
        row  = pd.DataFrame([{f: row_dict.get(f, np.nan) for f in info['features']}])
        Xi   = info['imputer'].transform(row)
        results[target] = round(float(info['model'].predict(Xi)[0]), 3)
    return results

def build_input_row(params: dict, sq_fe: pd.DataFrame) -> dict:
    """Build a feature-rich input row from basic user-provided parameters."""
    row = params.copy()
    feo    = params.get('%FeO', sq_fe['%FeO'].median())
    cao    = params.get('%CaO', sq_fe['%CaO'].median())
    mgo    = params.get('%MgO', sq_fe['%MgO'].median())
    sio2   = params.get('%SiO2', sq_fe['%SiO2'].median())
    al2o3  = params.get('%Al2O3', sq_fe['%Al2O3'].median())
    tfe    = sq_fe['%T(Fe)'].median()
    b2     = cao / sio2 if sio2 > 0 else 2.1
    row.update({
        'B2_calc':     b2,
        'B3':          (cao+mgo)/sio2 if sio2>0 else 2.3,
        'B4':          (cao+mgo)/(sio2+al2o3) if (sio2+al2o3)>0 else 1.5,
        'MgO_Al2O3_r': mgo/al2o3 if al2o3>0 else 0.6,
        'Gangue_load': al2o3+sio2,
        'Al2O3_SiO2':  al2o3/sio2 if sio2>0 else 0.55,
        'FeO_TFe':     feo/tfe if tfe>0 else 0.2,
        'B2_x_FeO':    b2*feo,
        'Al2O3_x_B2':  al2o3*b2,
        'MgO_x_Al2O3': mgo*al2o3,
        'pctFeO_r3':   feo,'pctFeO_r7':  feo,'pctFeO_r14': feo,
        'pctCaO_r3':   cao,'pctCaO_r7':  cao,'pctCaO_r14': cao,
        'pctMgO_r3':   mgo,'pctMgO_r7':  mgo,'pctMgO_r14': mgo,
        'pctAl2O3_r3':al2o3,'pctAl2O3_r7':al2o3,'pctAl2O3_r14':al2o3,
        'Basicity__B2_r3':b2,'Basicity__B2_r7':b2,'Basicity__B2_r14':b2,
        'MgO_Al2O3__r3':mgo/al2o3 if al2o3>0 else 0.6,
        'MgO_Al2O3__r7':mgo/al2o3 if al2o3>0 else 0.6,
        'MgO_Al2O3__r14':mgo/al2o3 if al2o3>0 else 0.6,
        'pctFeO_lag1':feo,'pctFeO_lag3':feo,'pctFeO_lag7':feo,
        'pctCaO_lag1':cao,'pctCaO_lag3':cao,'pctCaO_lag7':cao,
        'pctAl2O3_lag1':al2o3,'pctAl2O3_lag3':al2o3,'pctAl2O3_lag7':al2o3,
        'Basicity__B2_lag1':b2,'Basicity__B2_lag3':b2,'Basicity__B2_lag7':b2,
        'MgO_Al2O3__lag1':mgo/al2o3 if al2o3>0 else 0.6,
        'MgO_Al2O3__lag3':mgo/al2o3 if al2o3>0 else 0.6,
        'MgO_Al2O3__lag7':mgo/al2o3 if al2o3>0 else 0.6,
    })
    return row

def bf_suitability(preds):
    scores = {}
    for t in TARGETS:
        v, tgt = preds[t], BF_TARGETS[t]
        if t == 'RDI':
            scores[t] = max(0, min(100, 100 - (v - tgt) * 20)) if v > tgt else 100
        else:
            scores[t] = max(0, min(100, 100 - (tgt - v) * 20)) if v < tgt else 100
    return round(np.mean(list(scores.values())), 1), scores
