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
.stApp { background: #060d1a; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label { color: #94b4d4 !important; font-size:0.82rem; }

/* Top nav */
.nav-bar {
    background: linear-gradient(90deg, #0a1628 0%, #0f2347 50%, #0a1628 100%);
    border-bottom: 2px solid #1e4080;
    padding: 12px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; justify-content: space-between;
}
.nav-title { font-size:1.35rem; font-weight:700; color:#e2e8f0; letter-spacing:0.5px; }
.nav-sub   { font-size:0.75rem; color:#4a90d9; letter-spacing:1px; text-transform:uppercase; }
.nav-badge { background:#1e4080; border:1px solid #2563eb; border-radius:6px;
             padding:4px 12px; font-size:0.72rem; color:#7eb3f7; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0f2a4a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 18px;
    position: relative; overflow: hidden;
    transition: all 0.2s ease;
}
.kpi-card:hover { border-color:#2563eb; box-shadow:0 0 20px rgba(37,99,235,0.15); }
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: var(--accent,#2563eb);
}
.kpi-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.kpi-value { font-size:2rem; font-weight:700; color:#e2e8f0; font-family:'JetBrains Mono',monospace; line-height:1; }
.kpi-sub   { font-size:0.75rem; margin-top:4px; }
.kpi-ok    { color:#22c55e; }
.kpi-warn  { color:#f59e0b; }
.kpi-bad   { color:#ef4444; }
.kpi-delta { font-size:0.8rem; font-weight:600; margin-top:6px; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #0d1f3c, transparent);
    border-left: 3px solid #2563eb;
    padding: 8px 16px;
    margin: 16px 0 12px 0;
    font-size: 0.85rem; font-weight:600; color:#94b4d4;
    text-transform: uppercase; letter-spacing: 1px;
}
.insight-box {
    background: #0d1f3c; border:1px solid #1e3a5f;
    border-radius:8px; padding:12px 16px; margin:6px 0;
}
.insight-box h4 { color:#4a90d9; font-size:0.82rem; margin:0 0 4px 0; }
.insight-box p  { color:#94b4d4; font-size:0.8rem; margin:0; line-height:1.5; }

/* Tables */
.stDataFrame { background:#0d1f3c !important; }

/* Slider / input labels */
div[data-testid="stNumberInput"] label,
div[data-testid="column"] label { color:#94b4d4 !important; }

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
    st.markdown('<div style="text-align:center;padding:16px 0 8px"><span style="font-size:2rem">🏭</span><br><span style="font-size:0.95rem;font-weight:700;color:#e2e8f0">SINTER INTELLIGENCE</span><br><span style="font-size:0.7rem;color:#4a90d9;letter-spacing:1px">PLATFORM v2.0</span></div>', unsafe_allow_html=True)
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
        '🧠  Feature Importance & SHAP',
        '🔍  Root Cause Analysis',
        '🎯  Quality Prediction',
        '🔮  What-If Analysis',
        '⚡  Optimization Engine',
        '🔥  BF Suitability',
        '💡  Management Insights',
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
  <div>
    <div class="nav-title">⚙️ SINTER QUALITY INTELLIGENCE PLATFORM</div>
    <div class="nav-sub">Integrated Steel Plant — Industry 4.0 Analytics</div>
  </div>
  <div style="display:flex;gap:12px;align-items:center">
    <div class="nav-badge">TI: {last_ti:.2f} {'✅' if ti_ok else '⚠️'}</div>
    <div class="nav-badge">RDI: {last_rdi:.2f} {'✅' if rdi_ok else '⚠️'}</div>
    <div class="nav-badge">RI: {last_ri:.2f} {'✅' if ri_ok else '⚠️'}</div>
    <div class="nav-badge" style="border-color:#{'22c55e' if '🟢' in overall else ('f59e0b' if '🟡' in overall else 'ef4444')}">{overall}</div>
    {render_live_status_badge(_rt_info if "_rt_info" in dir() else {"data_age_str":"—","current_mtime":0}, st.session_state.get("data_updated",False))}
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
elif '🧠' in page:
    from prediction_engine import FEAT_LABELS, MET_INTERP

    # ── KPI DRIVER CARDS ──────────────────────────────────────────────────
    ti_top_feat  = FEAT_LABELS.get(sd['TI']['top_feats'][0],  sd['TI']['top_feats'][0])
    rdi_top_feat = FEAT_LABELS.get(sd['RDI']['top_feats'][0], sd['RDI']['top_feats'][0])
    ri_top_feat  = FEAT_LABELS.get(sd['RI']['top_feats'][0],  sd['RI']['top_feats'][0])
    ti_top_val   = sd['TI']['top_vals'][0]
    rdi_top_val  = sd['RDI']['top_vals'][0]
    ri_top_val   = sd['RI']['top_vals'][0]
    common_feats = set(sd['TI']['top_feats'][:10]) & set(sd['RDI']['top_feats'][:10]) & set(sd['RI']['top_feats'][:10])
    n_common = len(common_feats)
    common_labels = ', '.join([FEAT_LABELS.get(f,f) for f in list(common_feats)[:3]])

    st.markdown("""
    <div style="font-size:0.7rem;color:#4a90d9;letter-spacing:1px;
                text-transform:uppercase;margin-bottom:14px;font-weight:600">
      🧠 FEATURE IMPORTANCE & SHAP ANALYSIS
      <span style="color:#334155;font-weight:400;margin-left:8px">
        — Which process variables drive TI, RDI & RI?
      </span>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    for col,label,feat,val,tc,hint in [
        (k1,'Top TI Driver',ti_top_feat,ti_top_val,'#58A6FF','Cold strength driver'),
        (k2,'Top RDI Driver',rdi_top_feat,rdi_top_val,'#F85149','Degradation driver'),
        (k3,'Top RI Driver',ri_top_feat,ri_top_val,'#3FB950','Reducibility driver'),
        (k4,f'Common Drivers',f'{n_common} features',None,'#a855f7',common_labels),
    ]:
        val_html = (f'<div style="font-size:1rem;font-weight:700;color:{tc};'
                    f'font-family:JetBrains Mono,monospace;margin:4px 0;line-height:1.2">'
                    f'{feat}</div>'
                    f'<div style="font-size:0.68rem;color:#64748b">'
                    f'{"SHAP: "+str(round(val,4)) if val is not None else "across all targets"}</div>'
                    if val is not None else
                    f'<div style="font-size:1.4rem;font-weight:800;color:{tc};'
                    f'font-family:JetBrains Mono,monospace;margin:4px 0">{feat}</div>'
                    f'<div style="font-size:0.65rem;color:#64748b">{hint}</div>')
        col.markdown(
            f'<div style="background:#0d1f3c;border:1px solid #1e3a5f;'
            f'border-top:3px solid {tc};border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:6px">{label}</div>'
            f'{val_html}</div>',
            unsafe_allow_html=True)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # ── AI KEY FINDINGS BOX (dynamic — looks up a curated mechanism for
    #    whichever feature actually tops SHAP this run, instead of naming a
    #    fixed feature in hardcoded prose that goes stale after retraining) ──
    def _metallurgical_blurb(target):
        for feat in sd[target]['top_feats'][:5]:
            if feat in MET_INTERP.get(target, {}):
                mechanism, action = MET_INTERP[target][feat]
                label = FEAT_LABELS.get(feat, feat)
                return f'<b>{label}</b> is the strongest driver with a curated mechanism: {mechanism}. <i>{action}.</i>'
        feat = sd[target]['top_feats'][0]
        label = FEAT_LABELS.get(feat, feat)
        val   = sd[target]['top_vals'][0]
        return (f'<b>{label}</b> is currently the strongest SHAP driver '
                f'(mean |SHAP|={val:.3f}). No curated metallurgical note exists for '
                f'this feature yet — add one to MET_INTERP in prediction_engine.py.')

    ti_blurb  = _metallurgical_blurb('TI')
    rdi_blurb = _metallurgical_blurb('RDI')
    ri_blurb  = _metallurgical_blurb('RI')

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1f3c,#0a1628);
                border:1px solid #1e4080;border-radius:10px;
                padding:16px 20px;margin-bottom:18px">
      <div style="font-size:0.75rem;font-weight:700;color:#a855f7;
                  letter-spacing:0.8px;margin-bottom:10px">
        🤖 AI KEY FINDINGS — METALLURGICAL INTERPRETATION
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.76rem">
        <div style="background:#060d1a;border-radius:6px;padding:10px 12px;border-left:3px solid #58A6FF">
          <span style="color:#58A6FF;font-weight:600">TI:</span>
          <span style="color:#94b4d4"> {ti_blurb}</span>
        </div>
        <div style="background:#060d1a;border-radius:6px;padding:10px 12px;border-left:3px solid #F85149">
          <span style="color:#F85149;font-weight:600">RDI:</span>
          <span style="color:#94b4d4"> {rdi_blurb}</span>
        </div>
        <div style="background:#060d1a;border-radius:6px;padding:10px 12px;border-left:3px solid #3FB950">
          <span style="color:#3FB950;font-weight:600">RI:</span>
          <span style="color:#94b4d4"> {ri_blurb}</span>
        </div>
        <div style="background:#060d1a;border-radius:6px;padding:10px 12px;border-left:3px solid #a855f7">
          <span style="color:#a855f7;font-weight:600">Common:</span>
          <span style="color:#94b4d4"> {common_labels} appear in top-10 for all three targets —
          controlling these simultaneously optimises all quality indices.</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── MAIN TABS ─────────────────────────────────────────────────────────
    tab_fi, tab_shap, tab_bee, tab_dep, tab_wf, tab_met = st.tabs([
        '📊 Feature Importance',
        '🔵 SHAP Summary',
        '🐝 SHAP Dependence',
        '📈 Dependence Plot',
        '🌊 Waterfall',
        '🔬 Metallurgical Insights',
    ])

    show_all = st.sidebar.toggle('View All Features (Top 15)', value=False)
    n_show   = 15 if show_all else 5

    # ── TAB 1: Feature Importance ─────────────────────────────────────────
    with tab_fi:
        st.markdown('<div style="font-size:0.72rem;color:#64748b;margin-bottom:12px">'
                    'How frequently each variable was used by the model to split data. '
                    'Showing top ' + str(n_show) + ' features per target.</div>',
                    unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3, gap='medium')
        for col,target in zip([c1,c2,c3], TARGETS):
            with col:
                tc = TC[target]
                imp    = fi[target].head(n_show)
                labels = [FEAT_LABELS.get(f,f) for f in imp.index]
                vals   = imp.values
                fig = go.Figure(go.Bar(
                    x=vals[::-1], y=labels[::-1],
                    orientation='h',
                    marker=dict(
                        color=[tc]*len(vals),
                        opacity=[0.45 + 0.55*(i/max(len(vals)-1,1)) for i in range(len(vals))][::-1],
                        line=dict(color='rgba(0,0,0,0)', width=0),
                    ),
                    text=[f'{v:.4f}' for v in vals[::-1]],
                    textposition='outside',
                    textfont=dict(size=9, color='#94b4d4'),
                    hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>',
                ))
                fig.add_vline(x=0, line=dict(color='#1e3a5f', width=1))
                fig.update_layout(
                    paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
                    font=dict(family='Inter', color='#94b4d4', size=10),
                    title=dict(
                        text=f'<b style="color:{tc}">{target}</b>  —  Feature Importance<br>'
                             f'<span style="font-size:10px;color:#475569">{bm[target]["best_algo"]} model</span>',
                        font=dict(size=13), x=0, xanchor='left'),
                    margin=dict(l=10, r=70, t=60, b=10),
                    height=80 + n_show * 34,
                    xaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9),
                               title=dict(text='Importance Score', font=dict(size=9,color='#64748b'))),
                    yaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9.5),
                               automargin=True),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Model name + R² / MAE / RMSE caption for this target's best model
                metrics = bm[target].get('metrics', {})
                st.markdown(
                    f'<div style="font-size:0.68rem;color:#64748b;margin:-6px 0 8px;text-align:center">'
                    f'<b style="color:{tc}">{bm[target]["best_algo"]}</b> &nbsp;'
                    f'R²={metrics.get("R2","–")} &nbsp;|&nbsp; '
                    f'MAE={metrics.get("MAE","–")} &nbsp;|&nbsp; '
                    f'RMSE={metrics.get("RMSE","–")}</div>', unsafe_allow_html=True)

                # Top-10 Features with normalised Percentage Contribution
                pct_df = fi_pct.get(target)
                if pct_df is not None and len(pct_df):
                    show_df = pct_df.head(10).copy()
                    show_df['feature'] = show_df['feature'].map(lambda f: FEAT_LABELS.get(f, f))
                    show_df = show_df.rename(columns={'feature':'Feature','importance':'Importance Score',
                                                        'pct_contribution':'% Contribution'})
                    st.dataframe(
                        show_df.style.format({'Importance Score':'{:.4f}','% Contribution':'{:.2f}%'}),
                        use_container_width=True, hide_index=True, height=320)

    # ── TAB 2: SHAP Summary ───────────────────────────────────────────────
    with tab_shap:
        st.markdown('<div style="font-size:0.72rem;color:#64748b;margin-bottom:12px">'
                    'Mean absolute SHAP value — how much each feature shifts the prediction '
                    'away from the baseline. Showing top ' + str(n_show) + ' features.</div>',
                    unsafe_allow_html=True)
        s1,s2,s3 = st.columns(3, gap='medium')
        for col,target in zip([s1,s2,s3], TARGETS):
            with col:
                tc   = TC[target]
                data = sd[target]
                vals  = data['top_vals'][:n_show]
                feats = data['top_feats'][:n_show]
                labels= [FEAT_LABELS.get(f,f) for f in feats]
                fig = go.Figure(go.Bar(
                    x=vals[::-1], y=labels[::-1],
                    orientation='h',
                    marker=dict(
                        color=[tc]*len(vals),
                        opacity=[0.45 + 0.55*(i/max(len(vals)-1,1)) for i in range(len(vals))][::-1],
                        line=dict(color='rgba(0,0,0,0)', width=0),
                    ),
                    text=[f'{v:.4f}' for v in vals[::-1]],
                    textposition='outside',
                    textfont=dict(size=9, color='#94b4d4'),
                    hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>',
                ))
                fig.add_vline(x=0, line=dict(color='#1e3a5f', width=1))
                fig.update_layout(
                    paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
                    font=dict(family='Inter', color='#94b4d4', size=10),
                    title=dict(
                        text=f'<b style="color:{tc}">{target}</b>  —  SHAP Importance<br>'
                             f'<span style="font-size:10px;color:#475569">Mean |SHAP value| across all predictions</span>',
                        font=dict(size=13), x=0, xanchor='left'),
                    margin=dict(l=10, r=70, t=60, b=10),
                    height=80 + n_show * 34,
                    xaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9),
                               title=dict(text='Mean |SHAP Value|', font=dict(size=9,color='#64748b'))),
                    yaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9.5),
                               automargin=True),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: SHAP Beeswarm ─────────────────────────────────────────────
    with tab_bee:
        target_sel = st.selectbox('Select Quality Target', TARGETS,
                                   format_func=lambda t: {'TI':'TI — Tumbler Index (Cold Strength)',
                                                           'RDI':'RDI — Reduction Degradation Index',
                                                           'RI':'RI — Reducibility Index'}[t],
                                   key='bee_target')
        st.markdown(
            '<div style="font-size:0.72rem;color:#64748b;margin-bottom:10px">'
            'Each dot = one day\'s prediction. '
            '<span style="color:#3b82f6">Blue</span> = low feature value, '
            '<span style="color:#ef4444">Red</span> = high feature value. '
            'Position left/right shows whether the feature pushed the prediction down or up.</div>',
            unsafe_allow_html=True)
        data   = sd[target_sel]
        sv,X   = data['sv'], data['X']
        feats  = data['top_feats'][:n_show]
        labels = [FEAT_LABELS.get(f,f) for f in feats]
        tc     = TC[target_sel]
        fig    = go.Figure()
        for i,feat in enumerate(feats[::-1]):
            if feat not in X.columns: continue
            feat_vals = X[feat].values
            shap_vals = sv[:, X.columns.get_loc(feat)]
            norm      = (feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-9)
            colors    = [f'hsl({int(240-240*n)},80%,55%)' for n in norm]
            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=[labels[::-1][i]] * len(shap_vals),
                mode='markers',
                marker=dict(color=colors, size=4, opacity=0.45,
                            line=dict(width=0)),
                name=labels[::-1][i], showlegend=False,
                hovertemplate=f'{feat}<br>SHAP: %{{x:.4f}}<extra></extra>',
            ))
        fig.add_vline(x=0, line=dict(color='#94b4d4', width=1, dash='dot'),
                      annotation_text='Baseline', annotation_font=dict(color='#64748b',size=9))
        fig.update_layout(
            paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
            font=dict(family='Inter', color='#94b4d4', size=10),
            title=dict(
                text=f'<b style="color:{tc}">{target_sel}</b>  —  SHAP Beeswarm<br>'
                     f'<span style="font-size:10px;color:#475569">'
                     f'Feature value (colour) vs prediction impact (x-axis)</span>',
                font=dict(size=13), x=0, xanchor='left'),
            margin=dict(l=10, r=30, t=60, b=40),
            height=100 + n_show * 38,
            xaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9),
                       title=dict(text=f'SHAP Value  →  Impact on {target_sel} Prediction',
                                  font=dict(size=10, color='#64748b'))),
            yaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9.5),
                       automargin=True),
            hovermode='closest',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: Dependence Plot ────────────────────────────────────────────
    with tab_dep:
        st.markdown('<div style="font-size:0.72rem;color:#64748b;margin-bottom:10px">'
                    'Shows how a single feature value affects the SHAP contribution — '
                    'reveals non-linear relationships and interaction effects.</div>',
                    unsafe_allow_html=True)
        dc1,dc2 = st.columns(2)
        dep_target = dc1.selectbox('Target', TARGETS, key='dep_target')
        data       = sd[dep_target]
        X          = data['X']
        # Key metallurgical levers requested explicitly: FeO, Basicity, MgO,
        # Al2O3, MgO/Al2O3 — guarantee these are selectable even when they
        # don't happen to rank in this run's top-10 SHAP list.
        KEY_PARAMS = ['%FeO', 'B2_calc', '%MgO', '%Al2O3', 'MgO_Al2O3_r']
        candidate_raw = list(data['top_feats'][:10])
        for kp in KEY_PARAMS:
            if kp in X.columns and kp not in candidate_raw:
                candidate_raw.append(kp)
        dep_feats  = [FEAT_LABELS.get(f,f) for f in candidate_raw]
        dep_sel    = dc2.selectbox('Feature', dep_feats, key='dep_feat')
        dep_feat_raw = candidate_raw[dep_feats.index(dep_sel)]

        sv = data['sv']
        tc   = TC[dep_target]
        if dep_feat_raw in X.columns:
            feat_vals = X[dep_feat_raw].values
            shap_vals = sv[:, X.columns.get_loc(dep_feat_raw)]
            norm      = (feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-9)
            colors    = [f'hsl({int(240-240*n)},80%,55%)' for n in norm]
            fig = go.Figure(go.Scatter(
                x=feat_vals, y=shap_vals,
                mode='markers',
                marker=dict(color=colors, size=6, opacity=0.6,
                            colorscale='RdBu', showscale=True,
                            colorbar=dict(title='Feature<br>Value',
                                          tickfont=dict(size=8, color='#94b4d4'),
                                          titlefont=dict(size=9, color='#94b4d4'))),
                hovertemplate=f'{dep_sel}: %{{x:.3f}}<br>SHAP Impact: %{{y:.4f}}<extra></extra>',
            ))
            fig.add_hline(y=0, line=dict(color='#94b4d4', width=1, dash='dot'))
            fig.update_layout(
                paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
                font=dict(family='Inter', color='#94b4d4', size=10),
                title=dict(
                    text=f'<b style="color:{tc}">{dep_target}</b>  —  Dependence: {dep_sel}<br>'
                         f'<span style="font-size:10px;color:#475569">'
                         f'How {dep_sel} value changes its SHAP contribution to {dep_target}</span>',
                    font=dict(size=13), x=0, xanchor='left'),
                margin=dict(l=40, r=40, t=65, b=50),
                height=400,
                xaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9),
                           title=dict(text=f'{dep_sel}  (Feature Value)',
                                      font=dict(size=10, color='#64748b'))),
                yaxis=dict(gridcolor='#1e3a5f', zeroline=False, tickfont=dict(size=9),
                           title=dict(text=f'SHAP Contribution to {dep_target}',
                                      font=dict(size=10, color='#64748b'))),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: SHAP Waterfall (per-prediction breakdown) ────────────────────
    with tab_wf:
        st.markdown('<div style="font-size:0.72rem;color:#64748b;margin-bottom:10px">'
                    'Shows how each feature pushes ONE specific prediction up or down '
                    'from the baseline (average model output) to the final predicted value.</div>',
                    unsafe_allow_html=True)
        wc1, wc2 = st.columns([1,2])
        wf_target = wc1.selectbox('Target', TARGETS, key='wf_target')
        data   = sd[wf_target]
        X, sv  = data['X'], data['sv']
        n_rows = len(X)
        if n_rows == 0:
            st.info('No rows available for this target.')
        else:
            wf_row = wc2.slider('Observation # (row in training data)', 0, max(n_rows-1,0),
                                 0, key='wf_row')
            model    = bm[wf_target]['model']
            row_X    = X.iloc[[wf_row]]
            pred     = float(model.predict(row_X)[0])
            row_shap = np.asarray(sv[wf_row], dtype=float)
            # SHAP additivity: prediction ≈ baseline + sum(shap values for the row).
            # We don't persist the explainer's expected_value separately, so derive
            # the baseline implicitly from this identity rather than assuming it.
            base_value = pred - row_shap.sum()

            feats   = data['feats']
            contrib = pd.Series(row_shap, index=feats)
            ranked  = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
            top_contrib = ranked.head(12)
            other_sum   = ranked.drop(top_contrib.index).sum()

            labels = [FEAT_LABELS.get(f,f) for f in top_contrib.index]
            values = list(top_contrib.values)
            if abs(other_sum) > 1e-9 and len(ranked) > len(top_contrib):
                labels.append(f'Other ({len(ranked)-len(top_contrib)} features)')
                values.append(other_sum)

            tc = TC[wf_target]
            fig = go.Figure(go.Waterfall(
                orientation='v',
                measure=['absolute'] + ['relative']*len(values) + ['total'],
                x=['Baseline'] + labels + ['Prediction'],
                y=[base_value] + values + [pred],
                text=[f'{base_value:.2f}'] + [f'{v:+.3f}' for v in values] + [f'{pred:.2f}'],
                textposition='outside',
                textfont=dict(size=9, color='#94b4d4'),
                connector=dict(line=dict(color='#1e3a5f', width=1)),
                increasing=dict(marker=dict(color='#3FB950')),
                decreasing=dict(marker=dict(color='#F85149')),
                totals=dict(marker=dict(color='#58A6FF')),
            ))
            fig.update_layout(
                paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
                font=dict(family='Inter', color='#94b4d4', size=10),
                title=dict(text=f'<b style="color:{tc}">{wf_target}</b>  —  Prediction breakdown for observation #{wf_row}',
                           font=dict(size=13), x=0, xanchor='left'),
                margin=dict(l=40, r=40, t=60, b=130),
                height=480, showlegend=False,
                xaxis=dict(tickfont=dict(size=8.5), tickangle=-40),
                yaxis=dict(gridcolor='#1e3a5f', zeroline=False,
                           title=dict(text=wf_target, font=dict(size=10, color='#64748b'))),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f'Baseline (avg. model output): {base_value:.3f}  →  '
                       f'Final prediction: {pred:.3f}  (Δ = {pred-base_value:+.3f})')

    # ── TAB 6: Metallurgical Insights ─────────────────────────────────────
    with tab_met:
        for target in TARGETS:
            tc     = TC[target]
            interp = MET_INTERP.get(target, {})
            data   = sd[target]
            st.markdown(
                f'<div style="background:#0d1f3c;border-left:4px solid {tc};'
                f'border-radius:0 8px 8px 0;padding:10px 16px;margin:12px 0 8px">'
                f'<span style="color:{tc};font-size:0.95rem;font-weight:700">🎯 {target}</span>'
                f'<span style="color:#64748b;font-size:0.72rem;margin-left:10px">'
                f'{"Tumbler Index — Cold Strength" if target=="TI" else "Reduction Degradation Index" if target=="RDI" else "Reducibility Index"}'
                f'</span></div>',
                unsafe_allow_html=True)
            m1,m2 = st.columns(2, gap='medium')
            for i,(feat,val) in enumerate(zip(data['top_feats'][:n_show],
                                              data['top_vals'][:n_show])):
                label = FEAT_LABELS.get(feat, feat)
                mech,rec = interp.get(feat, ('Monitor this variable closely.',
                                             'Adjust within operating limits.'))
                rank_badge = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟'][i] if i < 10 else f'#{i+1}'
                card = (
                    f'<div style="background:#060d1a;border:1px solid #1e3a5f;'
                    f'border-radius:8px;padding:12px 14px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:6px">'
                    f'<span style="font-size:0.78rem;font-weight:700;color:{tc}">'
                    f'{rank_badge} {label}</span>'
                    f'<span style="font-size:0.68rem;background:#0d1f3c;'
                    f'border:1px solid #1e3a5f;border-radius:12px;'
                    f'padding:2px 8px;color:#94b4d4">SHAP: {val:.4f}</span>'
                    f'</div>'
                    f'<div style="font-size:0.73rem;color:#94b4d4;margin-bottom:4px">'
                    f'🔬 {mech}</div>'
                    f'<div style="font-size:0.71rem;color:#22c55e">'
                    f'✅ {rec}</div>'
                    f'</div>'
                )
                (m1 if i % 2 == 0 else m2).markdown(card, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — ROOT CAUSE ANALYSIS
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
elif '🔮' in page:
    st.markdown('<div class="section-header">🔮 WHAT-IF SCENARIO ANALYSIS</div>', unsafe_allow_html=True)
    b2c = 'Basicity  (B2)' if 'Basicity  (B2)' in sq_raw.columns else 'Basicity(B2)'
    med = lambda c: float(sq_raw[c].median()) if c in sq_raw.columns else 0.0

    # Baseline
    base_params = {
        '%FeO':med('%FeO'),'%CaO':med('%CaO'),'%MgO':med('%MgO'),
        '%SiO2':med('%SiO2'),'%Al2O3':med('%Al2O3'),'Avl. lime':med('Avl. lime'),
        'Basicity  (B2)':med(b2c),'MgO/Al2O3':med('MgO/Al2O3') if 'MgO/Al2O3' in sq_raw.columns else 0.6,
        '%T(Fe)':med('%T(Fe)'),'%K2O':med('%K2O'),'% P':med('% P'),
    }
    base_row  = build_input_row(base_params, sq_fe)
    base_pred = predict_from_row(base_row, bm)

    st.markdown('<div style="color:#94b4d4;font-size:0.8rem;margin-bottom:12px">Compare a modified scenario against the median/baseline. Use the sliders to modify variables.</div>',unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('**🔵 Baseline (Median)**')
        for t in TARGETS:
            st.markdown(f'<div class="insight-box"><b>{t}</b>: {base_pred[t]:.3f}</div>',unsafe_allow_html=True)
    with col2:
        st.markdown('**🟡 Scenario (Modified)**')
        feo2  = st.slider('Δ %FeO',  -2.0, 2.0, 0.0, 0.05)
        b2_2  = st.slider('Δ Basicity', -0.3, 0.3, 0.0, 0.01)
        mgo2  = st.slider('Δ %MgO',  -1.0, 1.0, 0.0, 0.05)
        al2   = st.slider('Δ %Al₂O₃',-1.0, 1.0, 0.0, 0.05)

    mod_params = base_params.copy()
    mod_params['%FeO']          += feo2
    mod_params['Basicity  (B2)']= mod_params.get('Basicity  (B2)',2.1) + b2_2
    mod_params['%MgO']          += mgo2
    mod_params['%Al2O3']        += al2
    mod_row  = build_input_row(mod_params, sq_fe)
    mod_pred = predict_from_row(mod_row, bm)

    st.markdown('<div class="section-header">📊 DELTA ANALYSIS</div>', unsafe_allow_html=True)
    d_cols = st.columns(3)
    for col,target in zip(d_cols,TARGETS):
        base_v = base_pred[target]; mod_v = mod_pred[target]
        delta  = mod_v - base_v
        tgt    = BF_TARGETS[target]
        ok_base= base_v>=tgt if target!='RDI' else base_v<=tgt
        ok_mod = mod_v>=tgt  if target!='RDI' else mod_v<=tgt
        status = ('✅ Improved' if ((delta>0 and target!='RDI') or (delta<0 and target=='RDI'))
                  else '❌ Worsened')
        col.metric(f'{target}',f'{mod_v:.3f}',f'{delta:+.3f} ({status})')

    fig = go.Figure()
    categories = TARGETS + [TARGETS[0]]
    base_norm = [(base_pred[t]-BF_TARGETS[t]) for t in TARGETS] + [(base_pred[TARGETS[0]]-BF_TARGETS[TARGETS[0]])]
    mod_norm  = [(mod_pred[t]-BF_TARGETS[t]) for t in TARGETS] + [(mod_pred[TARGETS[0]]-BF_TARGETS[TARGETS[0]])]
    fig.add_trace(go.Scatterpolar(r=base_norm,theta=categories,fill='toself',name='Baseline',
                                   line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatterpolar(r=mod_norm, theta=categories,fill='toself',name='Scenario',
                                   line=dict(color='#f59e0b')))
    fig.update_layout(polar=dict(bgcolor='#0d1f3c',radialaxis=dict(gridcolor='#1e3a5f'),
                                  angularaxis=dict(gridcolor='#1e3a5f')),
                      paper_bgcolor='#0d1f3c',font=dict(color='#94b4d4'),
                      title='Scenario vs Baseline — Quality Delta',height=380)
    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 9 — OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
elif '⚡' in page:
    st.markdown('<div class="section-header">⚡ MULTI-OBJECTIVE OPTIMIZATION ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:#64748b;margin-bottom:12px">Runs Optuna NSGA-II multi-objective optimization to find process settings that maximize TI, minimize RDI, and maximize RI simultaneously — producing a genuine Pareto front rather than trials clustered around one fixed trade-off.</div>',unsafe_allow_html=True)

    n_trials = st.slider('Optimization Trials', 100, 500, 300, 50)
    if st.button('🚀 Run Optimization'):
        with st.spinner('Running multi-objective optimization...'):
            result = run_optimization(bm, sq_raw, n_trials=n_trials)
        st.session_state['opt_result'] = result
        st.success('Optimization complete!')

    if 'opt_result' in st.session_state:
        res = st.session_state['opt_result']
        st.markdown('<div class="section-header">🏆 OPTIMAL QUALITY PREDICTIONS</div>', unsafe_allow_html=True)
        oc1,oc2,oc3 = st.columns(3)
        for col,t,dir in zip([oc1,oc2,oc3],TARGETS,['>=','<=','>=']):
            col.plotly_chart(gauge(res[f'best_{t}'],BF_TARGETS[t],dir,f'Optimal {t}',TC[t]),use_container_width=True)

        st.markdown('<div class="section-header">🔧 RECOMMENDED OPERATING PARAMETERS</div>', unsafe_allow_html=True)
        bp = res['best_params']
        rec_data = [(k, round(sq_raw[k].median() if k in sq_raw.columns else 0,4),
                     round(v,4)) for k,v in bp.items()]
        rec_df = pd.DataFrame(rec_data, columns=['Parameter','Current (Median)','Recommended'])
        rec_df['Δ Change'] = (rec_df['Recommended'] - rec_df['Current (Median)']).round(4)
        rec_df['Direction'] = rec_df['Δ Change'].apply(lambda x: '▲ Increase' if x>0 else '▼ Decrease')
        st.dataframe(rec_df.style.map(lambda v: 'color:#22c55e' if isinstance(v,str) and '▲' in v
                                       else ('color:#ef4444' if isinstance(v,str) and '▼' in v else ''),
                                       subset=['Direction']), use_container_width=True)

        st.markdown('<div class="section-header">🗺️ PARETO TRADE-OFF SPACE</div>', unsafe_allow_html=True)
        try:
            import pareto_engine
            # Normalise the result dict to the shape pareto_engine expects
            _pareto_cache = {
                'best':     res.get('best', {'TI':res.get('best_TI',79),
                                             'RDI':res.get('best_RDI',23),
                                             'RI':res.get('best_RI',70),
                                             'params':res.get('best_params',{})}),
                'pareto':   res.get('pareto', pd.DataFrame()),
                'ranges':   res.get('ranges', pd.DataFrame()),
                'levers':   res.get('levers', []),
                'best_quality': {'TI':res.get('best_TI',79),
                                 'RDI':res.get('best_RDI',23),
                                 'RI':res.get('best_RI',70)},
                'best_compromise': res.get('best', {}),
                'n_pareto': int(res.get('pareto',pd.DataFrame()).get(
                    'is_pareto', pd.Series([False])).sum()) if isinstance(
                    res.get('pareto'), pd.DataFrame) else 0,
                'n_trials': len(res.get('pareto', pd.DataFrame())),
            }
            pareto_engine.render(_pareto_cache, sq_raw)
        except Exception as e:
            st.error(f'⚠️ Pareto visualization could not render: {e}')
            import traceback; st.code(traceback.format_exc())
    else:
        st.info('Click "Run Optimization" to find optimal operating conditions.')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 10 — BF SUITABILITY
# ═══════════════════════════════════════════════════════════════════════════
elif '🔥' in page:
    st.markdown('<div class="section-header">🔥 BLAST FURNACE SUITABILITY PANEL</div>', unsafe_allow_html=True)
    avg_ti  = df['TI'].dropna().mean()
    avg_rdi = df['RDI'].dropna().mean()
    avg_ri  = df['RI'].dropna().mean()
    bf,bfs  = bf_suitability({'TI':avg_ti,'RDI':avg_rdi,'RI':avg_ri})

    b1,b2,b3,b4 = st.columns(4)
    b1.plotly_chart(gauge(avg_ti,78,'>=','TI — Cold Strength',TC['TI']),use_container_width=True)
    b2.plotly_chart(gauge(avg_rdi,25,'<=','RDI — Degradation',TC['RDI']),use_container_width=True)
    b3.plotly_chart(gauge(avg_ri,68,'>=','RI — Reducibility',TC['RI']),use_container_width=True)
    b4.plotly_chart(gauge(bf,85,'>=','Overall BF Score','#a855f7'),use_container_width=True)

    st.markdown('<div class="section-header">📋 BF IMPACT ASSESSMENT</div>', unsafe_allow_html=True)
    bf_impacts = [
        ('TI', avg_ti, 78, '>=',
         'Cold strength — affects transport, handling, and burden distribution',
         'Low TI → excessive fines generation at BF top → poor permeability',
         'Maintain TI > 78 | Adjust basicity & FeO | Reduce Al₂O₃'),
        ('RDI', avg_rdi, 25, '<=',
         'Low-temperature reduction degradation — critical for BF shaft',
         'High RDI → sinter degrades to fines in BF stack → flooding risk',
         'Control basicity ≥ 2.0 | Maintain FeO 10.5–11.5% | MgO > 2.0%'),
        ('RI', avg_ri, 68, '>=',
         'Reducibility — ease of oxygen removal in BF',
         'Low RI → higher coke consumption | higher CO₂ per tonne iron',
         'Reduce FeO < 10.5% | Optimise Al₂O₃ 3.0–3.5% | Stable basicity'),
    ]
    for target, val, tgt, direction, desc, risk, action in bf_impacts:
        ok = val>=tgt if direction=='>=' else val<=tgt
        risk_level = '🟢 LOW' if ok else '🔴 HIGH'
        color_hex = '052e16' if ok else '1c0202'
        border_hex = '22c55e' if ok else 'ef4444'
        st.markdown(f"""
        <div style="background:#{color_hex};border:1px solid #{border_hex};border-radius:10px;padding:14px 18px;margin:8px 0">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="color:{TC[target]};font-size:1rem;font-weight:700">{target} = {val:.3f}</span>
            <span style="color:{'#22c55e' if ok else '#ef4444'};font-size:0.8rem;font-weight:600">Risk: {risk_level}</span>
          </div>
          <p style="color:#94b4d4;font-size:0.8rem;margin:0 0 6px 0">📌 <b>Description:</b> {desc}</p>
          <p style="color:#f59e0b;font-size:0.8rem;margin:0 0 6px 0">⚠️ <b>BF Risk:</b> {risk}</p>
          <p style="color:#22c55e;font-size:0.8rem;margin:0">✅ <b>Recommended:</b> {action}</p>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 11 — MANAGEMENT INSIGHTS
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

elif '💡' in page:
    st.markdown('<div class="section-header">💡 MANAGEMENT INSIGHTS & AUTOMATED RECOMMENDATIONS</div>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(['Quality Summary','Top Drivers','Operating Windows','Action Plan'])

    with tab1:
        st.markdown('### 📊 Quality Performance Summary')
        perf = {}
        for target in TARGETS:
            d = df[target].dropna()
            tgt = BF_TARGETS[target]; direction = '>=' if target!='RDI' else '<='
            in_pct = ((d>=tgt).mean() if direction=='>=' else (d<=tgt).mean())*100
            perf[target] = {'Mean':d.mean(),'Std':d.std(),'CV%':d.std()/d.mean()*100,
                             'In-Target%':in_pct,'Min':d.min(),'Max':d.max()}
        perf_df = pd.DataFrame(perf).T.round(3)
        st.dataframe(perf_df,use_container_width=True)

        fig = go.Figure()
        for target in TARGETS:
            d = df[[date_col,target]].dropna()
            fig.add_trace(go.Scatter(x=d[date_col],y=d[target].rolling(14,min_periods=1).mean(),
                mode='lines',name=f'{target} (14d avg)',line=dict(color=TC[target],width=2)))
        fig.update_layout(**clayout(title='14-Day Rolling Average — All Quality Targets',height=340))
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        st.markdown('### 🔝 Top 5 Variables Affecting Each Target')
        for target in TARGETS:
            st.markdown(f'<div style="color:{TC[target]};font-weight:700;font-size:0.95rem;margin:12px 0 6px">{target}</div>',unsafe_allow_html=True)
            feats = sd[target]['top_feats'][:5]; vals = sd[target]['top_vals'][:5]
            interp = MET_INTERP.get(target,{})
            for rank,(f,v) in enumerate(zip(feats,vals),1):
                label = FEAT_LABELS.get(f,f)
                mech  = interp.get(f,('',''))[0]
                st.markdown(f'<div class="insight-box"><h4>#{rank} {label} — SHAP: {v:.4f}</h4><p>{mech}</p></div>',unsafe_allow_html=True)

    with tab3:
        st.markdown('### 🟩 Best vs Worst Operating Windows')
        for target in TARGETS:
            d = df[target].dropna(); tgt=BF_TARGETS[target]
            q25,q75 = d.quantile(0.25),d.quantile(0.75)
            best_mask  = d >= q75; worst_mask = d <= q25
            st.markdown(f'<div style="color:{TC[target]};font-weight:700;margin:10px 0 4px">{target}</div>',unsafe_allow_html=True)
            chem_c = [c for c in ['%FeO','%CaO','%MgO','%SiO2','%Al2O3','Basicity  (B2)','MgO/Al2O3'] if c in df.columns]
            best_chem = df.loc[best_mask.index[best_mask], chem_c].mean().round(3)
            worst_chem= df.loc[worst_mask.index[worst_mask], chem_c].mean().round(3)
            ow_df = pd.DataFrame({'Best Quartile':best_chem,'Worst Quartile':worst_chem})
            ow_df['Δ']= (ow_df['Best Quartile']-ow_df['Worst Quartile']).round(3)
            st.dataframe(ow_df,use_container_width=True)

    with tab4:
        st.markdown('### 📋 Recommended Action Plan')
        actions = [
            ('Chemistry Control','Maintain FeO 10.5–11.5%','High','Metallurgical — dense magnetite phase improves TI & RDI'),
            ('Chemistry Control','Target Basicity (B2) 2.05–2.15','High','Promotes SFCA formation — key for TI↑ RDI↓'),
            ('Chemistry Control','Increase MgO to 2.0–2.4%','High','Periclase inhibits reduction degradation → RDI↓'),
            ('Chemistry Control','Control Al₂O₃ < 3.3%','Medium','Excess Al₂O₃ weakens calcium ferrite bonds → TI↓'),
            ('Chemistry Control','Control K₂O < 0.08%','Medium','Alkalis weaken sinter lattice significantly'),
            ('Process Control','Stable machine speed (avoid rapid changes)','Medium','Speed variation → incomplete burn-through → TI↓'),
            ('Process Control','Maintain BTP temperature consistency','Medium','Temperature stability → uniform mineralogy'),
            ('Data Practice','Daily calibration of laboratory instruments','High','Measurement accuracy critical for process control'),
            ('Data Practice','Increase frequency of RI measurement','Low','RI has highest missing data — improve sampling'),
        ]
        for category,action,priority,basis in actions:
            p_color = '#ef4444' if priority=='High' else ('#f59e0b' if priority=='Medium' else '#22c55e')
            st.markdown(f'<div class="insight-box"><h4>📌 [{category}] {action} <span style="color:{p_color};font-size:0.75rem">● {priority}</span></h4><p>{basis}</p></div>',unsafe_allow_html=True)
