import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd, numpy as np, pickle, os, sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from prediction_engine import predict_from_row, build_input_row, bf_suitability, BF_TARGETS, FEAT_LABELS, MET_INTERP
from optimization_engine import run_optimization
from data_ingestion import render as di_render
from realtime_engine import (
    render_sidebar_controls, inject_autorefresh,
    load_artifacts_cached, get_artifact_mtime,
    render_live_status_badge, HAS_AUTOREFRESH
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sinter Quality Intelligence Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #ffffff; color: #1a1a2e; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a5c 0%, #0f2347 100%);
    border-right: 2px solid #446183;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label { color: #c8d9e8 !important; font-size:0.82rem; }

/* Top nav */
.nav-bar {
    background: #ffffff;
    border-bottom: 3px solid #446183;
    padding: 10px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 8px rgba(68,97,131,0.12);
}
.nav-title { font-size:1.35rem; font-weight:700; color:#1a3a5c; letter-spacing:0.5px; }
.nav-sub   { font-size:0.75rem; color:#446183; letter-spacing:1px; text-transform:uppercase; }
.nav-badge { background:#f0f4f8; border:1px solid #446183; border-radius:6px;
             padding:4px 12px; font-size:0.72rem; color:#1a3a5c; font-weight:600; }

/* KPI Cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #d1dce8;
    border-radius: 12px;
    padding: 16px 18px;
    position: relative; overflow: hidden;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(68,97,131,0.08);
}
.kpi-card:hover { border-color:#446183; box-shadow:0 4px 16px rgba(68,97,131,0.18); }
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: var(--accent,#446183);
}
.kpi-label { font-size:0.7rem; color:#6b7a8d; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.kpi-value { font-size:2rem; font-weight:700; color:#1a3a5c; font-family:'JetBrains Mono',monospace; line-height:1; }
.kpi-sub   { font-size:0.75rem; margin-top:4px; }
.kpi-ok    { color:#16a34a; }
.kpi-warn  { color:#d97706; }
.kpi-bad   { color:#dc2626; }
.kpi-delta { font-size:0.8rem; font-weight:600; margin-top:6px; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #e8f0f7, transparent);
    border-left: 3px solid #446183;
    padding: 8px 16px;
    margin: 16px 0 12px 0;
    font-size: 0.85rem; font-weight:600; color:#1a3a5c;
    text-transform: uppercase; letter-spacing: 1px;
}
.insight-box {
    background: #f8fafc; border:1px solid #d1dce8;
    border-radius:8px; padding:12px 16px; margin:6px 0;
}
.insight-box h4 { color:#446183; font-size:0.82rem; margin:0 0 4px 0; }
.insight-box p  { color:#4a5568; font-size:0.8rem; margin:0; line-height:1.5; }

/* Tables */
.stDataFrame { background:#ffffff !important; }

/* Slider / input labels */
div[data-testid="stNumberInput"] label,
div[data-testid="column"] label { color:#1a3a5c !important; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .bg { fill: #0d1f3c !important; }

/* Metric override */
[data-testid="metric-container"] {
    background: #0d1f3c; border:1px solid #1e3a5f;
    border-radius:8px; padding:12px;
}
[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
[data-testid="stMetricLabel"] { color:#64748b !important; }
[data-testid="stMetricDelta"] { font-size:0.8rem !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628; border-bottom:1px solid #1e3a5f; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color:#64748b;
    border-radius:6px 6px 0 0; padding:8px 18px;
    font-size:0.82rem; font-weight:500;
}
.stTabs [aria-selected="true"] {
    background: #0d1f3c !important; color:#e2e8f0 !important;
    border-bottom: 2px solid #2563eb !important;
}
div[data-baseweb="tab-panel"] { background:#060d1a; padding-top:16px; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    color: white; border:none; border-radius:8px;
    padding:10px 24px; font-weight:600; font-size:0.85rem;
    width:100%; transition:all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg,#2563eb,#3b82f6); transform:translateY(-1px); }

/* Status badge */
.status-ok   { background:#052e16; border:1px solid #22c55e; color:#22c55e; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.status-warn { background:#1c1200; border:1px solid #f59e0b; color:#f59e0b; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.status-bad  { background:#1c0202; border:1px solid #ef4444; color:#ef4444; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }

hr { border-color:#1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
    font=dict(family='Inter',color='#94b4d4',size=11),
    margin=dict(l=50,r=20,t=40,b=40),
    xaxis=dict(gridcolor='#1e3a5f',showgrid=True,zeroline=False),
    yaxis=dict(gridcolor='#1e3a5f',showgrid=True,zeroline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor='#1e3a5f',font=dict(size=10)),
)
TC = {'TI':'#3b82f6','RDI':'#ef4444','RI':'#22c55e'}
TARGETS = ['TI','RDI','RI']

def clayout(**kw): d=CHART_LAYOUT.copy(); d.update(kw); return d

def status_badge(ok): return '🟢' if ok else '🔴'
def kpi_class(ok): return 'kpi-ok' if ok else 'kpi-bad'

def gauge(val, target, direction, title, color):
    ok = (val >= target) if direction == '>=' else (val <= target)
    bar_color = color if ok else '#ef4444'
    if direction == '>=':
        lo  = min(val, target) * 0.92
        hi  = max(val, target) * 1.06
        ref = target
    else:
        lo  = min(val, target) * 0.94
        hi  = max(val, target) * 1.06
        ref = target

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        delta={
            'reference': ref,
            'valueformat': '.2f',
            'font': {'size': 13},
            'increasing': {'color': '#22c55e'} if direction == '>=' else {'color': '#ef4444'},
            'decreasing': {'color': '#ef4444'} if direction == '>=' else {'color': '#22c55e'},
        },
        number={
            'font': {'size': 20, 'color': '#e2e8f0', 'family': 'JetBrains Mono'},
            'valueformat': '.2f',
            'suffix': '',
        },
        title={
            'text': f'<b>{title}</b>',
            'font': {'size': 11, 'color': '#94b4d4'},
            'align': 'center',
        },
        domain={'x': [0.05, 0.95], 'y': [0.05, 1]},
        gauge={
            'axis': {
                'range': [lo, hi],
                'tickwidth': 1,
                'tickcolor': '#1e3a5f',
                'tickfont': {'size': 8},
                'nticks': 5,
            },
            'bar': {'color': bar_color, 'thickness': 0.6},
            'bgcolor': '#0d1f3c',
            'borderwidth': 1,
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [lo, ref], 'color': '#0f2347'},
                {'range': [ref, hi], 'color': '#0a1628'},
            ],
            'threshold': {
                'line': {'color': '#fbbf24', 'width': 2},
                'thickness': 0.8,
                'value': ref,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor='#0d1f3c',
        font=dict(color='#94b4d4'),
        margin=dict(l=25, r=25, t=40, b=50),   # extra bottom space for number+delta
        height=230,
    )
    return fig

# ── LOAD ARTIFACTS ──────────────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# Smart cached loader — auto-invalidates when artifacts.pkl mtime changes
_art_mtime = get_artifact_mtime(DASHBOARD_DIR)
art = load_artifacts_cached(os.path.join(DASHBOARD_DIR,'artifacts'), _art_mtime)
if art is None:
    st.error("Artifacts not found. Run: python model_training.py"); st.stop()
bm, sq_fe, sq_raw, pp_raw = art['best_models'], art['sq_fe'], art['sq_raw'], art['pp_raw']
ar, sd, fi, corr = art['all_results'], art['shap_data'], art['feat_imp'], art['corr']

# ── Auto-refresh (polls mtime; zero-config on first run) ─────────────
_rt_interval = st.session_state.get('_refresh_interval', '5 minutes')
_rt_seconds  = {'Off':0,'1 minute':60,'5 minutes':300,'10 minutes':600,'30 minutes':1800}.get(_rt_interval, 300)
inject_autorefresh(_rt_seconds)
fi_pct            = art.get('feat_imp_pct', {})
validation_report = art.get('validation_report')

# ── Live data override (set by Data Upload page) ──────────────────────
if st.session_state.get('data_updated'):
    sq_raw = st.session_state.get('live_sq', sq_raw)
    pp_raw = st.session_state.get('live_pp', pp_raw)
    sq_fe  = st.session_state.get('live_sq_fe', sq_fe)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:12px 0 8px"><img src="data:image/png;base64,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" style="height:44px;margin-bottom:6px;object-fit:contain"><br><span style="font-size:0.82rem;font-weight:700;color:#ffffff;letter-spacing:1px">SINTER INTELLIGENCE</span><br><span style="font-size:0.68rem;color:#a8c4d8;letter-spacing:1px">PLATFORM v2.0</span></div>', unsafe_allow_html=True)
    if validation_report is not None:
        total = validation_report['n_pass'] + validation_report['n_fail']
        ok    = validation_report['all_ok']
        icon  = '✅' if ok else '⚠️'
        color = '#22c55e' if ok else '#f59e0b'
        st.markdown(
            f'<div style="text-align:center;font-size:0.68rem;color:{color};'
            f'margin-bottom:6px">{icon} Validation: {validation_report["n_pass"]}/{total} checks passed</div>',
            unsafe_allow_html=True)
    st.markdown('---')
    page = st.selectbox('📍 NAVIGATION', [
        '📂  Data Upload & Processing',
        '🏠  Executive Dashboard',
        '📈  Quality Monitoring',
        '⚙️  Process Monitoring',
        '🔗  Correlation Analysis',
        '🔍  Root Cause Analysis',
        '🎯  Quality Prediction',
    ])
    st.markdown('---')
    # ── Real-time controls ───────────────────────────────
    _rt_info = render_sidebar_controls(DASHBOARD_DIR)
    # Reload if artifacts.pkl was updated by a retrain
    if _rt_info['should_reload']:
        st.cache_resource.clear()
        st.rerun()
    st.markdown('---')
    # Date filter
    date_col = 'DATE' if 'DATE' in sq_raw.columns else 'Date'
    dates = pd.to_datetime(sq_raw[date_col], errors='coerce').dropna()
    d_min, d_max = dates.min().date(), dates.max().date()
    st.markdown('<div style="font-size:0.75rem;color:#4a90d9;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">📅 DATE FILTER</div>', unsafe_allow_html=True)
    date_range = st.date_input('', value=(d_min, d_max), min_value=d_min, max_value=d_max, label_visibility='collapsed')
    st.markdown('---')
    st.markdown(f'<div style="font-size:0.72rem;color:#334155;text-align:center">Records: {len(sq_raw)} clean | Features: {sq_fe.shape[1]}<br>Models: RF · XGB · LGB · CAT<br>Last trained: current session</div>', unsafe_allow_html=True)

# Apply date filter
date_col = 'DATE' if 'DATE' in sq_fe.columns else 'Date'
sq_fe[date_col] = pd.to_datetime(sq_fe[date_col], errors='coerce')
try:
    d_from = pd.Timestamp(date_range[0]); d_to = pd.Timestamp(date_range[1])
except: d_from,d_to = sq_fe[date_col].min(),sq_fe[date_col].max()
mask = (sq_fe[date_col] >= d_from) & (sq_fe[date_col] <= d_to)
df = sq_fe[mask].copy()

# ── TOP NAV BAR ────────────────────────────────────────────────────────────
last_ti  = sq_raw['TI'].dropna().iloc[-1]  if 'TI'  in sq_raw.columns else 79.0
last_rdi = sq_raw['RDI'].dropna().iloc[-1] if 'RDI' in sq_raw.columns else 23.2
last_ri  = sq_raw['RI'].dropna().iloc[-1]  if 'RI'  in sq_raw.columns else 70.1

ti_ok  = last_ti  >= 78; rdi_ok = last_rdi <= 25; ri_ok  = last_ri  >= 68
overall= '🟢 NORMAL' if (ti_ok and rdi_ok and ri_ok) else ('🟡 CAUTION' if sum([ti_ok,rdi_ok,ri_ok])>=2 else '🔴 ALERT')

st.markdown(f"""
<div class="nav-bar">
  <div style="width:140px;min-width:100px"></div>
  <div style="text-align:center;flex:1">
    <div class="nav-title">⚙️ SINTER QUALITY INTELLIGENCE PLATFORM</div>
    <div class="nav-sub">Integrated Steel Plant — Industry 4.0 Analytics</div>
    <div style="display:flex;gap:8px;align-items:center;justify-content:center;margin-top:6px">
      <div class="nav-badge">TI: {last_ti:.2f} {'✅' if ti_ok else '⚠️'}</div>
      <div class="nav-badge">RDI: {last_rdi:.2f} {'✅' if rdi_ok else '⚠️'}</div>
      <div class="nav-badge">RI: {last_ri:.2f} {'✅' if ri_ok else '⚠️'}</div>
      <div class="nav-badge" style="border-color:#{'22c55e' if '🟢' in overall else ('d97706' if '🟡' in overall else 'dc2626')}">{overall}</div>
      {render_live_status_badge(_rt_info if "_rt_info" in dir() else {"data_age_str":"—","current_mtime":0}, st.session_state.get("data_updated",False))}
    </div>
  </div>
  <div style="width:140px;min-width:100px;display:flex;align-items:center;justify-content:flex-end">
    <img src="data:image/png;base64,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" style="height:52px;object-fit:contain;" alt="Jindal Steel">
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if '🏠' in page:
    st.markdown('<div class="section-header">🎯 QUALITY KPIs — CURRENT vs TARGET</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    avg_ti  = df['TI'].dropna().mean()
    avg_rdi = df['RDI'].dropna().mean()
    avg_ri  = df['RI'].dropna().mean()
    bf_score, bf_scores = bf_suitability({'TI':avg_ti,'RDI':avg_rdi,'RI':avg_ri})
    quality_score = round((bf_scores['TI']+bf_scores['RDI']+bf_scores['RI'])/3,1)
    ti_in  = (df['TI'].dropna() >= 78).mean()*100
    rdi_in = (df['RDI'].dropna() <= 25).mean()*100
    ri_in  = (df['RI'].dropna() >= 68).mean()*100

    with c1:
        st.plotly_chart(gauge(avg_ti, 78.0,'>=','Tumbler Index (TI)',TC['TI']), use_container_width=True)
    with c2:
        st.plotly_chart(gauge(avg_rdi, 25.0,'<=','Reduction Degradation (RDI)',TC['RDI']), use_container_width=True)
    with c3:
        st.plotly_chart(gauge(avg_ri, 68.0,'>=','Reducibility Index (RI)',TC['RI']), use_container_width=True)
    with c4:
        st.plotly_chart(gauge(bf_score, 85.0,'>=','BF Suitability Score','#a855f7'), use_container_width=True)

    st.markdown('<div class="section-header">📊 PERFORMANCE INDICATORS</div>', unsafe_allow_html=True)
    p1,p2,p3,p4,p5,p6 = st.columns(6)
    metrics = [
        (p1,'TI Achievement',f'{ti_in:.1f}%','≥78 target',ti_in>=80),
        (p2,'RDI Achievement',f'{rdi_in:.1f}%','≤25 target',rdi_in>=80),
        (p3,'RI Achievement',f'{ri_in:.1f}%','≥68 target',ri_in>=80),
        (p4,'BF Score',f'{bf_score:.1f}','/ 100',bf_score>=85),
        (p5,'Quality Score',f'{quality_score:.1f}','/ 100',quality_score>=80),
        (p6,'Data Records',f'{len(df):,}','in range',True),
    ]
    for col,label,val,sub,ok in metrics:
        col.metric(label, val, sub)

    st.markdown('<div class="section-header">📈 QUALITY TREND OVERVIEW</div>', unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(['TI Trend','RDI Trend','RI Trend'])
    for tab,target in zip([tab1,tab2,tab3],TARGETS):
        with tab:
            d2 = df[[date_col,target]].dropna()
            if len(d2) < 3: st.info("Insufficient data in selected range."); continue
            roll = d2[target].rolling(7,min_periods=1).mean()
            mn,mx = d2[target].mean()-2*d2[target].std(), d2[target].mean()+2*d2[target].std()
            tgt = BF_TARGETS[target]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d2[date_col],y=d2[target],mode='markers',
                name=target,marker=dict(color=TC[target],size=4,opacity=0.5),showlegend=True))
            fig.add_trace(go.Scatter(x=d2[date_col],y=roll,mode='lines',
                name='7d Moving Avg',line=dict(color=TC[target],width=2.5)))
            fig.add_hline(y=tgt,line=dict(color='#fbbf24',width=1.5,dash='dash'),
                          annotation_text=f'Target {">" if target!="RDI" else "<"}{tgt}',
                          annotation_font=dict(color='#fbbf24',size=10))
            fig.add_hrect(y0=mn,y1=mx,fillcolor=TC[target],opacity=0.04,line_width=0)
            fig.update_layout(**clayout(title=f'{target} Trend with 7-Day Moving Average',height=320))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🔔 QUALITY ALERTS — LAST 7-DAY MOVING AVERAGE</div>', unsafe_allow_html=True)
    a1,a2,a3 = st.columns(3)
    for col,target,direction in zip([a1,a2,a3],TARGETS,['>=','<=','>=']):
        d_sorted  = df[[date_col,target]].dropna().sort_values(date_col)
        last7     = d_sorted.tail(7)
        recent    = last7[target].mean()
        date_from = last7[date_col].iloc[0].strftime('%d %b %Y') if len(last7) else '—'
        date_to   = last7[date_col].iloc[-1].strftime('%d %b %Y') if len(last7) else '—'
        tgt       = BF_TARGETS[target]
        ok        = recent>=tgt if direction=='>=' else recent<=tgt
        dir_sym   = '≥' if direction=='>=' else '≤'
        badge_cls = 'status-ok'  if ok else 'status-bad'
        badge_txt = '✅ ON TARGET' if ok else '⚠️ OFF TARGET'
        col.markdown(f"""
        <div class="insight-box" style="text-align:center;padding:18px 14px">
          <div style="font-size:0.72rem;color:#4a90d9;font-weight:600;letter-spacing:0.5px;margin-bottom:4px">
            {target} — 7-Day Moving Average
          </div>
          <div style="font-size:0.67rem;color:#475569;margin-bottom:10px">
            📅 {date_from} → {date_to}
          </div>
          <div style="font-size:1.9rem;font-weight:700;color:#e2e8f0;
                      font-family:'JetBrains Mono',monospace;line-height:1.1;margin-bottom:4px">
            {recent:.3f}
          </div>
          <div style="font-size:0.71rem;color:#64748b;margin-bottom:10px">
            Target: {dir_sym}{tgt}
          </div>
          <span class="{badge_cls}">{badge_txt}</span>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — QUALITY MONITORING
# ═══════════════════════════════════════════════════════════════════════════
elif '📈' in page:
    st.markdown('<div class="section-header">📈 QUALITY MONITORING — TREND ANALYSIS WITH CONTROL LIMITS</div>', unsafe_allow_html=True)
    for target in TARGETS:
        d2 = df[[date_col,target]].dropna().copy()
        if len(d2)<5: continue
        mu,sigma = d2[target].mean(), d2[target].std()
        ucl,lcl  = mu+3*sigma, mu-3*sigma
        roll7    = d2[target].rolling(7,min_periods=1).mean()
        tgt      = BF_TARGETS[target]
        fig = make_subplots(rows=2,cols=1,row_heights=[0.75,0.25],shared_xaxes=True,
                            vertical_spacing=0.04)
        # Raw
        colors_pts = [TC[target] if ((v>=tgt) if target!='RDI' else (v<=tgt)) else '#ef4444' for v in d2[target]]
        fig.add_trace(go.Scatter(x=d2[date_col],y=d2[target],mode='markers',
            marker=dict(color=colors_pts,size=5,opacity=0.65),name='Daily',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=d2[date_col],y=roll7,mode='lines',
            line=dict(color=TC[target],width=2.5),name='7d Avg'),row=1,col=1)
        # Control limits
        for y,name,color,dash in [(ucl,'UCL','#f59e0b','dash'),(lcl,'LCL','#f59e0b','dash'),(mu,'Mean','#64748b','dot')]:
            fig.add_hline(y=y,line=dict(color=color,width=1,dash=dash),
                          annotation_text=f'{name}:{y:.2f}',row=1,col=1,
                          annotation_font=dict(color=color,size=9))
        fig.add_hline(y=tgt,line=dict(color='#22c55e',width=1.5,dash='dashdot'),
                      annotation_text=f'BF Target: {tgt}',row=1,col=1,
                      annotation_font=dict(color='#22c55e',size=9))
        # Distribution histogram
        fig.add_trace(go.Histogram(x=d2[target],marker_color=TC[target],opacity=0.7,
            nbinsx=30,name='Dist'),row=2,col=1)
        fig.update_layout(**clayout(
            title=f'{target} — Statistical Process Control Chart  |  σ={sigma:.3f}  CV={sigma/mu*100:.2f}%',
            height=420))
        fig.update_yaxes(title_text=target,row=1,col=1,title_font=dict(color='#94b4d4'))
        fig.update_yaxes(title_text='Freq',row=2,col=1,title_font=dict(color='#94b4d4'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('---')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROCESS MONITORING
# ═══════════════════════════════════════════════════════════════════════════
elif '⚙️' in page:
    from process_monitor import render as pm_render
    from date_utils import parse_dates_robust
    pp_date_col = 'Date'
    if pp_date_col in pp_raw.columns:
        # NOTE: this dataset stores PP dates as DD.MM.YYYY text. Plain
        # pd.to_datetime(..., errors='coerce') defaults to month-first and
        # silently turned >half the rows to NaT (verified on Sinter_Data.xlsx:
        # 427/712 rows lost) — parse_dates_robust picks whichever of
        # month-first/day-first parses more rows successfully.
        pp_raw[pp_date_col] = parse_dates_robust(pp_raw[pp_date_col])
        pp_mask = (pp_raw[pp_date_col] >= d_from) & (pp_raw[pp_date_col] <= d_to)
        pp_filt = pp_raw[pp_mask].copy()
    else:
        pp_filt = pp_raw.copy()
    try:
        pm_render(pp_filt, pp_date_col, d_from, d_to)
    except Exception as e:
        st.error(f'⚠️ Process Monitoring could not render: {e}')
        st.caption('This usually means a column has an unexpected type after a data upload. '
                   'Try "Reset to Original Data" on the Data Upload page, or re-check the source file.')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif '🔗' in page:
    st.markdown('<div class="section-header">🔗 CORRELATION ANALYSIS</div>', unsafe_allow_html=True)
    chem_cols = [c for c in ['%T(Fe)','%FeO','%CaO','%MgO','%SiO2','%Al2O3','%K2O','% P',
                              'Avl. lime','Basicity  (B2)','MgO/Al2O3','B2_calc','B3','B4',
                              'MgO_Al2O3_r','Gangue_load','FeO_TFe','B2_x_FeO','Al2O3_x_B2']
                 if c in df.columns]
    thresh = st.slider('Correlation threshold', 0.1, 0.9, 0.3, 0.05)
    c_df   = df[chem_cols+TARGETS].corr()

    fig = px.imshow(c_df, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    text_auto='.2f', aspect='auto')
    fig.update_traces(textfont_size=9)
    fig.update_layout(**clayout(title='Pearson Correlation Heatmap — Chemistry × Quality Targets',height=600))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🔝 TOP CORRELATIONS WITH QUALITY TARGETS</div>', unsafe_allow_html=True)
    ct1,ct2,ct3 = st.columns(3)
    for col,target in zip([ct1,ct2,ct3],TARGETS):
        with col:
            st.markdown(f'<div style="color:{TC[target]};font-weight:700;font-size:0.9rem;margin-bottom:8px">{target}</div>',unsafe_allow_html=True)
            tc = c_df[target].drop(TARGETS).abs().nlargest(10)
            raw_vals = c_df[target].drop(TARGETS).loc[tc.index]
            fig2 = go.Figure(go.Bar(
                x=raw_vals.values, y=raw_vals.index,
                orientation='h',
                marker_color=['#22c55e' if v>0 else '#ef4444' for v in raw_vals.values],
                marker_opacity=0.8,
            ))
            fig2.update_layout(**clayout(title=f'Correlation with {target}',height=350,
                                          margin=dict(l=120,r=20,t=35,b=20)))
            st.plotly_chart(fig2,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE IMPORTANCE & SHAP  (Redesigned)
# ═══════════════════════════════════════════════════════════════════════════
elif '🔍' in page:
    st.markdown('<div class="section-header">🔍 ROOT CAUSE ANALYSIS</div>', unsafe_allow_html=True)
    rca_target = st.selectbox('Select Quality Issue',['TI decreased','RDI increased','RI decreased'])
    target = rca_target.split()[0]

    d2 = df[[date_col,target]].dropna()
    if len(d2) < 10:
        st.warning("Insufficient data."); st.stop()

    recent_avg = d2[target].tail(7).mean()
    hist_avg   = d2[target].mean()
    delta      = recent_avg - hist_avg
    tgt        = BF_TARGETS[target]
    direction  = '>=' if target != 'RDI' else '<='
    is_issue   = (recent_avg < hist_avg and target!='RDI') or (recent_avg > hist_avg and target=='RDI')

    col1,col2 = st.columns([1,2])
    with col1:
        st.plotly_chart(gauge(recent_avg, tgt, direction, f'Recent {target} (7d avg)', TC[target]), use_container_width=True)
        status_txt = '⚠️ DETERIORATING' if is_issue else '✅ WITHIN NORM'
        st.markdown(f'<div class="insight-box" style="text-align:center"><h4>{status_txt}</h4><p>7d avg: <b>{recent_avg:.3f}</b><br>Historical avg: <b>{hist_avg:.3f}</b><br>Δ = <b style="color:{"#ef4444" if is_issue else "#22c55e"}">{delta:+.3f}</b></p></div>',unsafe_allow_html=True)
    with col2:
        # Top contributing features via SHAP
        data = sd[target]; feats = data['top_feats'][:8]
        shap_means = data['top_vals'][:8]
        labels = [FEAT_LABELS.get(f,f) for f in feats]
        severity = ['🔴 Critical' if v > np.percentile(shap_means,66) else ('🟡 Moderate' if v > np.percentile(shap_means,33) else '🟢 Minor') for v in shap_means]
        contrib_pct = (shap_means/shap_means.sum()*100).round(1)
        fig = go.Figure(go.Bar(
            x=contrib_pct[::-1], y=labels[::-1], orientation='h',
            marker=dict(color=['#ef4444','#ef4444','#ef4444','#f59e0b','#f59e0b','#f59e0b','#22c55e','#22c55e'][::-1],opacity=0.8),
            text=[f'{p:.1f}%' for p in contrib_pct[::-1]], textposition='inside',
        ))
        fig.update_layout(**clayout(title=f'Root Cause Analysis — {target} Deterioration Drivers',
                                     height=360,margin=dict(l=160,r=20,t=40,b=20)))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-header">💊 RECOMMENDED CORRECTIVE ACTIONS</div>', unsafe_allow_html=True)
    interp = MET_INTERP.get(target,{})
    for i,(feat,pct,sev) in enumerate(zip(feats[:5],contrib_pct[:5],severity[:5])):
        label = FEAT_LABELS.get(feat,feat)
        mech,rec = interp.get(feat,('Monitor this variable','Adjust according to operating limits'))
        st.markdown(f'<div class="insight-box"><h4>{sev} | {label} — {pct:.1f}% contribution</h4><p>🔬 {mech}<br>✅ <b>Action:</b> {rec}</p></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — QUALITY PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif '🎯' in page:
    st.markdown('<div class="section-header">🎯 REAL-TIME QUALITY PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:#64748b;margin-bottom:16px">Adjust process parameters below to predict TI, RDI, and RI instantly.</div>',unsafe_allow_html=True)

    med = lambda c: float(sq_raw[c].median()) if c in sq_raw.columns else 0.0
    mn  = lambda c: float(sq_raw[c].quantile(0.05)) if c in sq_raw.columns else 0.0
    mx  = lambda c: float(sq_raw[c].quantile(0.95)) if c in sq_raw.columns else 100.0

    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown('<div style="color:#4a90d9;font-size:0.8rem;font-weight:600;margin-bottom:8px">CHEMISTRY</div>',unsafe_allow_html=True)
        feo   = st.slider('%FeO',     mn('%FeO'),   mx('%FeO'),   med('%FeO'),   0.01, key='feo')
        cao   = st.slider('%CaO',     mn('%CaO'),   mx('%CaO'),   med('%CaO'),   0.01, key='cao')
        mgo   = st.slider('%MgO',     mn('%MgO'),   mx('%MgO'),   med('%MgO'),   0.01, key='mgo')
        sio2  = st.slider('%SiO₂',    mn('%SiO2'),  mx('%SiO2'),  med('%SiO2'),  0.01, key='sio2')
        al2o3 = st.slider('%Al₂O₃',   mn('%Al2O3'), mx('%Al2O3'), med('%Al2O3'), 0.01, key='al2o3')
        avl   = st.slider('Avl. Lime',mn('Avl. lime') if 'Avl. lime' in sq_raw.columns else 5.0,
                           mx('Avl. lime') if 'Avl. lime' in sq_raw.columns else 10.0,
                           med('Avl. lime') if 'Avl. lime' in sq_raw.columns else 7.0, 0.01)

    with col2:
        st.markdown('<div style="color:#4a90d9;font-size:0.8rem;font-weight:600;margin-bottom:8px">PROCESS</div>',unsafe_allow_html=True)
        b2_col_raw = 'Basicity  (B2)' if 'Basicity  (B2)' in sq_raw.columns else 'Basicity(B2)'
        b2    = st.slider('Basicity (B2)', mn(b2_col_raw) if b2_col_raw in sq_raw.columns else 1.8,
                           mx(b2_col_raw) if b2_col_raw in sq_raw.columns else 2.4,
                           med(b2_col_raw) if b2_col_raw in sq_raw.columns else 2.1, 0.01)
        mga   = st.slider('MgO/Al₂O₃', 0.3, 1.2, float(sq_raw['MgO/Al2O3'].median()) if 'MgO/Al2O3' in sq_raw.columns else 0.6, 0.01)
        tfe   = st.slider('%T(Fe)', mn('%T(Fe)'), mx('%T(Fe)'), med('%T(Fe)'), 0.01)
        kp    = st.slider('%K₂O', mn('%K2O'), mx('%K2O'), med('%K2O'), 0.001)
        pp_col = st.slider('% P', mn('% P'), mx('% P'), med('% P'), 0.001)

    params = {
        '%FeO':feo,'%CaO':cao,'%MgO':mgo,'%SiO2':sio2,'%Al2O3':al2o3,
        'Avl. lime':avl,'Basicity  (B2)':b2,'MgO/Al2O3':mga,
        '%T(Fe)':tfe,'%K2O':kp,'% P':pp_col,
    }
    row  = build_input_row(params, sq_fe)
    pred = predict_from_row(row, bm)
    bf,bfs = bf_suitability(pred)

    st.markdown('<div class="section-header">🔮 PREDICTION RESULTS</div>', unsafe_allow_html=True)
    r1,r2,r3,r4 = st.columns(4)
    for col,target in zip([r1,r2,r3],TARGETS):
        v = pred[target]; tgt = BF_TARGETS[target]
        ok = v>=tgt if target!='RDI' else v<=tgt
        delta = v - tgt
        col.metric(f'Predicted {target}', f'{v:.3f}', f'{"+" if delta>=0 else ""}{delta:.3f} vs target',
                   delta_color='normal' if (ok and target!='RDI') or (not ok and target=='RDI') else 'inverse')
    r4.metric('BF Suitability', f'{bf:.1f} / 100', 'Score')

    pcols = st.columns(3)
    for col,target,direction in zip(pcols,TARGETS,['>=','<=','>=']):
        col.plotly_chart(gauge(pred[target],BF_TARGETS[target],direction,f'Predicted {target}',TC[target]),use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 — WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif '📂' in page:
    # ── Retrain from Data Upload page ─────────────────────────────────
    import os as _os
    _sd_path = _os.path.join(DASHBOARD_DIR, 'Sinter_Data.xlsx')
    _sd_exists = _os.path.exists(_sd_path)
    _retrain_cols = st.columns([2,1])
    with _retrain_cols[1]:
        if _sd_exists and st.button('🚀 Retrain on Sinter_Data.xlsx',
                                    key='_di_retrain_btn', use_container_width=True,
                                    help='Re-run full ML pipeline on Sinter_Data.xlsx'):
            from realtime_engine import trigger_retrain
            trigger_retrain(_sd_path, DASHBOARD_DIR)
            st.rerun()
        elif not _sd_exists:
            st.warning('Sinter_Data.xlsx not found in dashboard folder.')
        # Show retrain progress
        rs = st.session_state.get('retrain_status')
        if rs:
            if rs['status'] == 'running':
                st.progress(rs['pct']/100, text=rs['msg'])
            elif rs['status'] == 'done':
                st.success(rs['msg'])
            elif rs['status'] == 'error':
                st.error(rs['msg'])
    di_render(sq_raw, pp_raw, sq_fe, art)

 
