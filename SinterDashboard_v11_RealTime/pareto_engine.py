"""
pareto_engine.py — Redesigned Pareto Optimization Visualization
SCADA-grade industrial UX with true Pareto front, 3D surface, insights panel.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Theme ──────────────────────────────────────────────────────────────────
BG    = '#0d1f3c'
PLOT  = '#060d1a'
TEXT  = '#e2e8f0'
GRID  = '#1e3a5f'
C_OPT = '#22c55e'   # green  = optimal / pareto front
C_TRD = '#f59e0b'   # yellow = trade-off zone
C_AVG = '#3b82f6'   # blue   = average / current
C_BAD = '#ef4444'   # red    = poor solutions
C_REC = '#a855f7'   # purple = recommended point

BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=PLOT,
    font=dict(family='Inter', color=TEXT, size=11),
    hoverlabel=dict(bgcolor='#0a1628', bordercolor=GRID, font=dict(size=11)),
    legend=dict(bgcolor='rgba(6,13,26,0.85)', bordercolor=GRID,
                font=dict(size=10), itemsizing='constant'),
)

# ── Pareto-optimality check ─────────────────────────────────────────────────
def pareto_front(df, maximize_ti=True, minimize_rdi=True, maximize_ri=True):
    """Return a boolean mask of Pareto-optimal rows: a row is optimal if no
    other row is at least as good in every objective and strictly better in
    at least one (i.e. no other row dominates it).

    Verified against synthetic data: every single-objective optimum (max TI,
    min RDI, max RI) is always flagged Pareto-optimal, every flagged row is
    genuinely undominated, and every unflagged row is genuinely dominated by
    something in the set.
    """
    n = len(df)
    # Convert every objective to a common "lower is better" cost space.
    costs = np.column_stack([
        -df['TI'].values  if maximize_ti  else  df['TI'].values,
         df['RDI'].values if minimize_rdi else -df['RDI'].values,
        -df['RI'].values  if maximize_ri  else  df['RI'].values,
    ])
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        c = costs[i]
        # Rows j with c <= costs[j] everywhere and c < costs[j] somewhere
        # are dominated BY i — those (and only those) get marked inefficient.
        dominated_by_i = np.all(c <= costs, axis=1) & np.any(c < costs, axis=1)
        is_efficient[dominated_by_i] = False
        is_efficient[i] = True   # i can't dominate itself
    return is_efficient

def solution_quality(ti, rdi, ri):
    """Score a solution: ≥0 = meets targets. Works for scalars AND for
    vectorized pandas Series/ndarrays — np.maximum (not the builtin max())
    is required for the latter, since max() on a Series raises 'truth value
    of a Series is ambiguous'."""
    return (ti - 78) + (ri - 68) - np.maximum(0, rdi - 25) * 2

def render(opt_cache, sq_raw):
    if opt_cache is None:
        st.info('Run optimization first (⚡ Optimization Engine page).')
        return

    pareto     = opt_cache['pareto'].copy()
    best       = opt_cache.get('best', {})
    ranges     = opt_cache.get('ranges', pd.DataFrame())
    levers     = opt_cache.get('levers', [])
    best_q     = opt_cache.get('best_quality', {})
    best_comp  = opt_cache.get('best_compromise', {})

    if len(pareto) == 0:
        st.warning('No optimization results found.')
        return

    # Classify solutions
    pareto['_score']   = solution_quality(pareto['TI'], pareto['RDI'], pareto['RI'])
    pareto['_pareto']  = pareto_front(pareto)
    pareto['_quality'] = pd.cut(
        pareto['_score'],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=['Poor', 'Marginal', 'Good', 'Optimal']
    )

    # Current (mean of clean data)
    cur = {
        'TI':  float(sq_raw['TI'].dropna().mean()),
        'RDI': float(sq_raw['RDI'].dropna().mean()),
        'RI':  float(sq_raw['RI'].dropna().mean()),
    }

    # Recommended = best compromise from Pareto front
    pf_df  = pareto[pareto['_pareto']]
    if len(pf_df) > 0:
        pf_df  = pf_df.copy()
        pf_df['_comp_score'] = solution_quality(pf_df['TI'], pf_df['RDI'], pf_df['RI'])
        rec_row = pf_df.loc[pf_df['_comp_score'].idxmax()]
    else:
        rec_row = pareto.loc[pareto['_score'].idxmax()]

    rec = {'TI': rec_row['TI'], 'RDI': rec_row['RDI'], 'RI': rec_row['RI']}

    # ── PAGE HEADER ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1628,#0f2347);
                border:1px solid #1e3a5f;border-radius:12px;
                padding:18px 24px;margin-bottom:18px">
      <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-bottom:4px">
        ⚡ Multi-Objective Pareto Optimization Engine
      </div>
      <div style="font-size:0.76rem;color:#64748b;line-height:1.6">
        400 Optuna TPE trials &nbsp;·&nbsp;
        Objectives: <b style="color:#58A6FF">Maximize TI</b> &nbsp;|&nbsp;
        <b style="color:#ef4444">Minimize RDI</b> &nbsp;|&nbsp;
        <b style="color:#22c55e">Maximize RI</b> &nbsp;·&nbsp;
        Pareto-optimal front computed via non-dominated sorting
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI SUMMARY CARDS ─────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_rows = [
        (k1, 'Best TI',         f"{pareto['TI'].max():.3f}",  '#58A6FF', f'Target ≥78'),
        (k2, 'Lowest RDI',      f"{pareto['RDI'].min():.3f}", '#F85149', f'Target ≤25'),
        (k3, 'Best RI',         f"{pareto['RI'].max():.3f}",  '#3FB950', f'Target ≥68'),
        (k4, 'Pareto Solutions',f"{pareto['_pareto'].sum()}",  '#a855f7', f'of {len(pareto)} trials'),
        (k5, 'Rec. TI / RDI',   f"{rec['TI']:.2f} / {rec['RDI']:.2f}", '#f97316', 'Best compromise'),
        (k6, 'Rec. RI',         f"{rec['RI']:.3f}",           '#22c55e', 'Best compromise'),
    ]
    for col, label, val, color, sub in kpi_rows:
        col.markdown(
            f'<div style="background:#0d1f3c;border:1px solid #1e3a5f;'
            f'border-top:3px solid {color};border-radius:8px;'
            f'padding:12px 14px;text-align:center">'
            f'<div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:0.6px;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:{color};'
            f'font-family:JetBrains Mono,monospace">{val}</div>'
            f'<div style="font-size:0.62rem;color:#475569;margin-top:3px">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # ── COLOR MAPPER ──────────────────────────────────────────────────────
    def point_color(row):
        if row['_pareto']:           return C_OPT
        if row['_score'] >= 0:       return C_TRD
        if row['_score'] >= -1:      return C_AVG
        return C_BAD

    colors_all = pareto.apply(point_color, axis=1).tolist()
    sizes_all  = pareto['_pareto'].map({True:12, False:6}).tolist()
    opacity_all= pareto['_pareto'].map({True:1.0, False:0.35}).tolist()

    def hover_text(row):
        lines = [
            f"<b>TI={row['TI']:.3f}</b>  RDI={row['RDI']:.3f}  RI={row['RI']:.3f}",
        ]
        for lv in levers:
            if lv in row:
                lines.append(f"{lv}: {row[lv]:.3f}")
        if row['_pareto']:
            lines.append('<b>★ PARETO OPTIMAL</b>')
        return '<br>'.join(lines)

    hover_all = pareto.apply(hover_text, axis=1).tolist()

    # ─────────────────────────────────────────────────────────────────────
    # PLOT 1 — TI vs RDI (coloured by RI)
    # PLOT 2 — RI vs TI (coloured by RDI)
    # ─────────────────────────────────────────────────────────────────────
    p_tab, p3d_tab, insight_tab = st.tabs([
        '📊 2D Trade-off Plots',
        '🌐 3D Pareto Surface',
        '💡 Optimization Insights',
    ])

    with p_tab:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                '<b>Plot 1:</b> TI vs RDI  (colour = RI)',
                '<b>Plot 2:</b> RI vs TI  (colour = RDI)',
            ],
            horizontal_spacing=0.10,
        )

        # ── PLOT 1: TI vs RDI ─────────────────────────────────────────────
        # Non-pareto background
        non_pf = pareto[~pareto['_pareto']]
        fig.add_trace(go.Scatter(
            x=non_pf['TI'], y=non_pf['RDI'],
            mode='markers',
            marker=dict(color=non_pf['RI'], colorscale='RdYlGn',
                        cmin=pareto['RI'].quantile(0.1),
                        cmax=pareto['RI'].quantile(0.9),
                        size=5, opacity=0.28, showscale=False,
                        line=dict(width=0)),
            name='All trials', showlegend=True,
            hovertemplate='TI=%{x:.2f} RDI=%{y:.2f}<br>RI=%{marker.color:.2f}<extra></extra>',
        ), row=1, col=1)

        # Pareto front — sorted by TI for line
        pf_df  = pareto[pareto['_pareto']].sort_values('TI')
        fig.add_trace(go.Scatter(
            x=pf_df['TI'], y=pf_df['RDI'],
            mode='lines+markers',
            line=dict(color=C_OPT, width=2.5),
            marker=dict(color=C_OPT, size=10,
                        line=dict(color='white', width=1.5)),
            name='Pareto Front', showlegend=True,
            hovertemplate='<b>PARETO</b><br>TI=%{x:.3f}<br>RDI=%{y:.3f}<extra></extra>',
        ), row=1, col=1)

        # Target zone band
        fig.add_hrect(y0=0, y1=25, fillcolor='rgba(34,197,94,0.06)',
                      line=dict(color='rgba(34,197,94,0.3)', width=1, dash='dot'),
                      row=1, col=1)
        fig.add_vline(x=78, line=dict(color='#58A6FF', width=1.5, dash='dash'),
                      row=1, col=1)
        fig.add_hline(y=25, line=dict(color='#F85149', width=1.5, dash='dash'),
                      row=1, col=1)

        # Current operating point
        fig.add_trace(go.Scatter(
            x=[cur['TI']], y=[cur['RDI']],
            mode='markers', marker=dict(color='#f97316', size=14,
                symbol='diamond', line=dict(color='white',width=2)),
            name='Current', showlegend=True,
            hovertemplate=f'<b>CURRENT</b><br>TI={cur["TI"]:.2f}<br>RDI={cur["RDI"]:.2f}<extra></extra>',
        ), row=1, col=1)

        # Recommended
        fig.add_trace(go.Scatter(
            x=[rec['TI']], y=[rec['RDI']],
            mode='markers', marker=dict(color=C_REC, size=16,
                symbol='star', line=dict(color='white',width=2)),
            name='Recommended', showlegend=True,
            hovertemplate=f'<b>RECOMMENDED</b><br>TI={rec["TI"]:.3f}<br>RDI={rec["RDI"]:.3f}<extra></extra>',
        ), row=1, col=1)

        # ── PLOT 2: RI vs TI ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=non_pf['RI'], y=non_pf['TI'],
            mode='markers',
            marker=dict(color=non_pf['RDI'], colorscale='RdYlGn_r',
                        cmin=pareto['RDI'].quantile(0.1),
                        cmax=pareto['RDI'].quantile(0.9),
                        size=5, opacity=0.28, showscale=False,
                        line=dict(width=0)),
            name='All trials (P2)', showlegend=False,
            hovertemplate='RI=%{x:.2f} TI=%{y:.2f}<extra></extra>',
        ), row=1, col=2)

        pf_df2 = pareto[pareto['_pareto']].sort_values('RI')
        fig.add_trace(go.Scatter(
            x=pf_df2['RI'], y=pf_df2['TI'],
            mode='lines+markers',
            line=dict(color=C_OPT, width=2.5),
            marker=dict(color=C_OPT, size=10,
                        line=dict(color='white', width=1.5)),
            name='Pareto Front (P2)', showlegend=False,
            hovertemplate='<b>PARETO</b><br>RI=%{x:.3f}<br>TI=%{y:.3f}<extra></extra>',
        ), row=1, col=2)

        fig.add_vline(x=68, line=dict(color='#3FB950', width=1.5, dash='dash'), row=1, col=2)
        fig.add_hline(y=78, line=dict(color='#58A6FF', width=1.5, dash='dash'), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[cur['RI']], y=[cur['TI']],
            mode='markers', marker=dict(color='#f97316', size=14,
                symbol='diamond', line=dict(color='white',width=2)),
            name='Current (P2)', showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[rec['RI']], y=[rec['TI']],
            mode='markers', marker=dict(color=C_REC, size=16,
                symbol='star', line=dict(color='white',width=2)),
            name='Recommended (P2)', showlegend=False,
        ), row=1, col=2)

        fig.update_layout(**BASE,
            height=520,
            title=dict(text='', x=0),
            margin=dict(l=55,r=55,t=55,b=55),
        )
        fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10),
                         title_font=dict(size=11, color='#64748b'))
        fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10),
                         title_font=dict(size=11, color='#64748b'))
        fig.update_xaxes(title_text='TI  (Tumbler Index)', row=1, col=1)
        fig.update_yaxes(title_text='RDI  (Reduction Degradation)', row=1, col=1)
        fig.update_xaxes(title_text='RI  (Reducibility Index)', row=1, col=2)
        fig.update_yaxes(title_text='TI  (Tumbler Index)', row=1, col=2)

        # Annotation: Target Zone
        fig.add_annotation(
            x=79.5, y=24.2, text='✅ Target Zone',
            font=dict(color='#22c55e', size=10), showarrow=False,
            xref='x', yref='y', row=1, col=1)

        # Color legend explanation
        for col_idx, (clr, lbl) in enumerate([
            ('#22c55e','— Pareto Front'),
            ('#f97316','◆ Current Operating Point'),
            ('#a855f7','★ Recommended Point'),
            ('#58A6FF','-- TI Target ≥78'),
            ('#F85149','-- RDI Target ≤25'),
        ], 1):
            fig.add_annotation(
                x=0.01 + (col_idx-1)*0.20, y=-0.07,
                text=f'<span style="color:{clr}">{lbl}</span>',
                xref='paper', yref='paper', showarrow=False,
                font=dict(size=9, color=clr), align='left')

        st.plotly_chart(fig, use_container_width=True)

    # ── 3D PARETO SURFACE ──────────────────────────────────────────────────
    with p3d_tab:
        fig3d = go.Figure()

        # Non-pareto — semi-transparent grey
        fig3d.add_trace(go.Scatter3d(
            x=non_pf['TI'], y=non_pf['RDI'], z=non_pf['RI'],
            mode='markers',
            marker=dict(size=3, color='#334155', opacity=0.25),
            name='Sub-optimal', showlegend=True,
            hovertemplate='TI=%{x:.2f}<br>RDI=%{y:.2f}<br>RI=%{z:.2f}<extra></extra>',
        ))

        # Color all trials by score
        good = pareto[pareto['_score'] >= 0]
        if len(good):
            fig3d.add_trace(go.Scatter3d(
                x=good['TI'], y=good['RDI'], z=good['RI'],
                mode='markers',
                marker=dict(size=5, color=good['_score'],
                            colorscale='RdYlGn', opacity=0.55,
                            colorbar=dict(title=dict(text='Score', font=dict(size=10, color=TEXT)),
                                          x=1.02, tickfont=dict(size=9, color=TEXT)),
                            showscale=True),
                name='Feasible Solutions', showlegend=True,
                hovertemplate='TI=%{x:.2f}<br>RDI=%{y:.2f}<br>RI=%{z:.2f}<extra></extra>',
            ))

        # Pareto front
        fig3d.add_trace(go.Scatter3d(
            x=pf_df['TI'], y=pf_df['RDI'], z=pf_df['RI'],
            mode='markers+lines',
            marker=dict(size=9, color=C_OPT,
                        line=dict(color='white', width=1.5)),
            line=dict(color=C_OPT, width=3),
            name='Pareto Front', showlegend=True,
            hovertemplate='<b>PARETO</b><br>TI=%{x:.3f}<br>RDI=%{y:.3f}<br>RI=%{z:.3f}<extra></extra>',
        ))

        # Target planes
        ti_range = [pareto['TI'].min(), pareto['TI'].max()]
        rdi_range= [pareto['RDI'].min(), pareto['RDI'].max()]
        ri_range = [pareto['RI'].min(), pareto['RI'].max()]

        # Current point
        fig3d.add_trace(go.Scatter3d(
            x=[cur['TI']], y=[cur['RDI']], z=[cur['RI']],
            mode='markers',
            marker=dict(size=12, color='#f97316', symbol='diamond',
                        line=dict(color='white', width=2)),
            name='Current', showlegend=True,
        ))

        # Recommended
        fig3d.add_trace(go.Scatter3d(
            x=[rec['TI']], y=[rec['RDI']], z=[rec['RI']],
            mode='markers',
            marker=dict(size=14, color=C_REC, symbol='diamond',
                        line=dict(color='white', width=2.5)),
            name='Recommended', showlegend=True,
            hovertemplate=(f'<b>RECOMMENDED</b><br>'
                           f'TI={rec["TI"]:.3f}<br>'
                           f'RDI={rec["RDI"]:.3f}<br>'
                           f'RI={rec["RI"]:.3f}<extra></extra>'),
        ))

        fig3d.update_layout(
            **{k:v for k,v in BASE.items() if k not in ['plot_bgcolor']},
            height=620,
            scene=dict(
                bgcolor=PLOT,
                xaxis=dict(title='TI (Tumbler Index)', gridcolor=GRID,
                           backgroundcolor=PLOT, color=TEXT),
                yaxis=dict(title='RDI (Reduction Degradation)', gridcolor=GRID,
                           backgroundcolor=PLOT, color=TEXT),
                zaxis=dict(title='RI (Reducibility Index)', gridcolor=GRID,
                           backgroundcolor=PLOT, color=TEXT),
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
            ),
            title=dict(
                text='3D Pareto Surface — TI × RDI × RI Optimization Space',
                font=dict(size=13, color=TEXT), x=0, xanchor='left'),
            margin=dict(l=0,r=0,t=50,b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown(
            '<div style="font-size:0.72rem;color:#64748b;text-align:center;margin-top:-8px">'
            '🖱️ Drag to rotate  ·  Scroll to zoom  ·  '
            '🟢 Green = Pareto optimal  ·  🟣 Purple = Recommended  ·  🟠 Orange = Current'
            '</div>', unsafe_allow_html=True)

    # ── OPTIMIZATION INSIGHTS PANEL ────────────────────────────────────────
    with insight_tab:
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        # Delta computation
        d_ti  = rec['TI']  - cur['TI']
        d_rdi = rec['RDI'] - cur['RDI']
        d_ri  = rec['RI']  - cur['RI']

        # Quality summary table
        insight_data = [
            ('Best TI achievable',     f"{pareto['TI'].max():.3f}",  f"Current: {cur['TI']:.2f}", '#58A6FF'),
            ('Lowest RDI achievable',  f"{pareto['RDI'].min():.3f}", f"Current: {cur['RDI']:.2f}",'#F85149'),
            ('Best RI achievable',     f"{pareto['RI'].max():.3f}",  f"Current: {cur['RI']:.2f}", '#3FB950'),
            ('Recommended TI',         f"{rec['TI']:.3f}",  f"Δ {d_ti:+.3f} ({d_ti/cur['TI']*100:+.1f}%)", '#58A6FF'),
            ('Recommended RDI',        f"{rec['RDI']:.3f}", f"Δ {d_rdi:+.3f} ({d_rdi/cur['RDI']*100:+.1f}%)", '#F85149'),
            ('Recommended RI',         f"{rec['RI']:.3f}",  f"Δ {d_ri:+.3f} ({d_ri/cur['RI']*100:+.1f}%)", '#3FB950'),
        ]

        i1,i2 = st.columns(2)
        for i,(label,val,sub,color) in enumerate(insight_data):
            col = i1 if i%2==0 else i2
            good = (('TI' in label and d_ti>=0) or
                    ('RDI' in label and d_rdi<=0) or
                    ('RI' in label and d_ri>=0) or
                    'achievable' in label)
            bg   = '#052e16' if good else '#1c0202'
            brd  = '#22c55e' if good else '#ef4444'
            col.markdown(
                f'<div style="background:{bg};border:1px solid {brd};'
                f'border-radius:8px;padding:12px 16px;margin-bottom:8px">'
                f'<div style="font-size:0.66rem;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:4px">{label}</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:{color};'
                f'font-family:JetBrains Mono,monospace">{val}</div>'
                f'<div style="font-size:0.68rem;color:#94b4d4;margin-top:3px">{sub}</div>'
                f'</div>', unsafe_allow_html=True)

        # Recommended parameter settings
        st.markdown('<div style="background:#0a1628;border-left:3px solid #a855f7;'
                    'padding:8px 14px;font-size:0.78rem;font-weight:700;color:#c084fc;'
                    'letter-spacing:0.5px;border-radius:0 6px 6px 0;margin:14px 0 10px">'
                    '🔧 RECOMMENDED PARAMETER SETTINGS</div>', unsafe_allow_html=True)

        param_rows = []
        for lv in levers:
            if lv in rec_row.index and lv in sq_raw.columns:
                curr_val = float(sq_raw[lv].median())
                rec_val  = float(rec_row[lv]) if lv in rec_row.index else np.nan
                if np.isnan(rec_val): continue
                delta    = rec_val - curr_val
                direction= '▲ Increase' if delta > 0 else '▼ Decrease'
                param_rows.append({
                    'Parameter': lv,
                    'Current (Median)': round(curr_val, 4),
                    'Recommended':      round(rec_val, 4),
                    'Δ Change':         round(delta, 4),
                    'Direction':        direction,
                })
        if param_rows:
            rec_df = pd.DataFrame(param_rows)
            st.dataframe(
                rec_df.style.map(
                    lambda v: 'color:#22c55e;font-weight:bold' if isinstance(v,str) and '▲' in v
                              else ('color:#ef4444;font-weight:bold' if isinstance(v,str) and '▼' in v else ''),
                    subset=['Direction']
                ).set_properties(**{
                    'background-color':'#0d1f3c','color':'#e2e8f0','font-size':'0.8rem'
                }),
                use_container_width=True,
            )

        # Expected improvement percentages
        improvements = {
            'TI':  d_ti  / cur['TI']  * 100,
            'RDI': -d_rdi/ cur['RDI'] * 100,
            'RI':  d_ri  / cur['RI']  * 100,
        }
        st.markdown('<div style="background:#0a1628;border-left:3px solid #f97316;'
                    'padding:8px 14px;font-size:0.78rem;font-weight:700;color:#fb923c;'
                    'letter-spacing:0.5px;border-radius:0 6px 6px 0;margin:14px 0 10px">'
                    '📈 EXPECTED IMPROVEMENT vs CURRENT</div>', unsafe_allow_html=True)

        i_cols = st.columns(3)
        for col, (t, pct) in zip(i_cols, improvements.items()):
            tc   = {'TI':'#58A6FF','RDI':'#F85149','RI':'#3FB950'}[t]
            good2= pct > 0
            col.markdown(
                f'<div style="background:#0d1f3c;border:1px solid #1e3a5f;'
                f'border-radius:8px;padding:12px;text-align:center">'
                f'<div style="font-size:0.68rem;color:#64748b;margin-bottom:4px">'
                f'{t} Improvement</div>'
                f'<div style="font-size:1.5rem;font-weight:800;'
                f'color:{"#22c55e" if good2 else "#ef4444"};'
                f'font-family:JetBrains Mono,monospace">'
                f'{pct:+.2f}%</div>'
                f'<div style="font-size:0.65rem;color:{tc};margin-top:2px">'
                f'{"✅ Meets target" if ((t=="TI" and rec["TI"]>=78) or (t=="RDI" and rec["RDI"]<=25) or (t=="RI" and rec["RI"]>=68)) else "⚠️ Near target"}'
                f'</div></div>', unsafe_allow_html=True)
