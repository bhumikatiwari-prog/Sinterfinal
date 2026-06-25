"""
Process Parameter Monitoring — Redesigned SCADA-grade Module
Industrial UX for sinter plant engineers & operations managers
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Color palette ──────────────────────────────────────────────────────────
CLR = dict(
    raw        = 'rgba(120,180,255,0.55)',
    ma7        = '#f97316',
    ma30       = '#a855f7',
    normal_fill= 'rgba(34,197,94,0.07)',
    warn_fill  = 'rgba(245,158,11,0.05)',
    ucl        = '#f59e0b',
    lcl        = '#f59e0b',
    mean_line  = '#475569',
    max_dot    = '#a855f7',
    min_dot    = '#06b6d4',
    latest_dot = '#ffffff',
    anomaly    = '#ef4444',
    grid       = '#1e3a5f',
    panel      = '#0d1f3c',
    bg         = '#060d1a',
    text       = '#e2e8f0',
    muted      = '#64748b',
    compare    = ['#3b82f6','#f97316','#22c55e','#a855f7','#06b6d4','#f59e0b'],
)

# ── Operating limits ────────────────────────────────────────────────────────
OP_LIMITS = {
    'Machine speed':    dict(lo=1.8,  hi=3.2,  warn_lo=2.0,  warn_hi=3.0,  unit='m/min',  desc='Strand speed — controls burn-through time & sinter quality'),
    'WGF Speed':        dict(lo=750,  hi=1050, warn_lo=800,  warn_hi=1000, unit='rpm',    desc='Waste gas fan — controls suction pressure & bed permeability'),
    'ESP Inlet Temp.':  dict(lo=130,  hi=175,  warn_lo=135,  warn_hi=170,  unit='°C',     desc='Electrostatic precipitator inlet temperature'),
    'ESP Inlet Press.': dict(lo=90,   hi=155,  warn_lo=95,   warn_hi=150,  unit='mmWC',   desc='ESP inlet pressure — reflects bed resistance'),
    'Feed':             dict(lo=350,  hi=450,  warn_lo=370,  warn_hi=440,  unit='t/h',    desc='Feed rate to sinter strand'),
    'Moisture':         dict(lo=5.0,  hi=7.5,  warn_lo=5.2,  warn_hi=7.0,  unit='%',      desc='Mix moisture — affects bed permeability & productivity'),
    'BTP Temperature':  dict(lo=380,  hi=465,  warn_lo=395,  warn_hi=460,  unit='°C',     desc='Burn-through point — key sintering completeness indicator'),
    'Cooler Speed':     dict(lo=0.55, hi=1.05, warn_lo=0.6,  warn_hi=1.0,  unit='m/min',  desc='Sinter cooler speed — affects product temperature'),
    'Production':       dict(lo=2200, hi=4000, warn_lo=2400, warn_hi=3900, unit='t/day',  desc='Daily sinter production rate'),
    'Basicity(B2)':     dict(lo=1.95, hi=2.20, warn_lo=2.00, warn_hi=2.15, unit='ratio',  desc='CaO/SiO₂ — controls bonding phase mineralogy'),
    '%FeO':             dict(lo=9.8,  hi=10.8, warn_lo=10.0, warn_hi=10.7, unit='%',      desc='Iron oxide state — affects RDI and TI'),
    '%MgO':             dict(lo=1.9,  hi=2.4,  warn_lo=2.0,  warn_hi=2.35, unit='%',      desc='Magnesia — reduces RDI, improves sinter structure'),
    '%CaO':             dict(lo=12.0, hi=14.0, warn_lo=12.3, warn_hi=13.8, unit='%',      desc='Lime content — drives basicity & SFCA formation'),
    '%SiO2':            dict(lo=5.6,  hi=6.7,  warn_lo=5.8,  warn_hi=6.5,  unit='%',      desc='Silica — inversely affects basicity'),
    '%Al2O3':           dict(lo=2.9,  hi=3.5,  warn_lo=3.0,  warn_hi=3.4,  unit='%',      desc='Alumina — excess weakens calcium ferrite bonds'),
}

# ── Helper: defensive numeric coercion ──────────────────────────────────────
def _numeric(series):
    """Coerce to numeric, turning any stray non-numeric cell (stray text,
    blank-but-not-NaN, unit suffixes left over from a manual edit) into NaN
    instead of letting it silently make the whole column object-dtype and
    crash the first .quantile()/.std()/.mean() call downstream."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors='coerce')


# ── Helper: health status ───────────────────────────────────────────────────
def param_health(series, col):
    series = _numeric(series)
    if col not in OP_LIMITS or len(series.dropna()) == 0:
        return 'normal', '🟢'
    lim    = OP_LIMITS[col]
    recent = series.dropna().tail(7).mean()
    if recent < lim['lo'] or recent > lim['hi']:
        return 'critical', '🔴'
    if recent < lim['warn_lo'] or recent > lim['warn_hi']:
        return 'warning', '🟡'
    return 'normal', '🟢'

# ── Helper: anomaly detection (IQR 2×) ────────────────────────────────────
def detect_anomalies(series):
    series = _numeric(series)
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr     = q3 - q1
    return (series < q1 - 2.0 * iqr) | (series > q3 + 2.0 * iqr)

# ── Helper: stats dictionary ────────────────────────────────────────────────
def param_stats(series, col):
    s = _numeric(series).dropna()
    if len(s) == 0:
        return {}
    sigma = s.std()
    return dict(
        current = s.iloc[-1],
        ma7     = s.tail(7).mean(),
        ma30    = s.tail(30).mean(),
        std     = 0.0 if pd.isna(sigma) else sigma,
        mn      = s.min(),
        mx      = s.max(),
        cv      = (sigma / s.mean() * 100) if (s.mean() != 0 and not pd.isna(sigma)) else 0,
        unit    = OP_LIMITS.get(col, {}).get('unit', ''),
        anomaly_count = int(detect_anomalies(s).sum()),
    )

# ── Stat card HTML ──────────────────────────────────────────────────────────
def stat_card_html(label, col, stats, health, badge):
    lim      = OP_LIMITS.get(col, {})
    unit     = stats.get('unit', '')
    cur      = stats.get('current', 0)
    ma7      = stats.get('ma7', 0)
    ma30     = stats.get('ma30', 0)
    std      = stats.get('std', 0)
    cv       = stats.get('cv', 0)
    mn       = stats.get('mn', 0)
    mx       = stats.get('mx', 0)
    anom     = stats.get('anomaly_count', 0)
    lo       = lim.get('lo', '—')
    hi       = lim.get('hi', '—')
    desc     = lim.get('desc', '')
    delta    = cur - ma7
    delta_str= f'{"▲" if delta > 0 else "▼"} {abs(delta):.3f} vs 7d avg'
    delta_col= '#22c55e' if delta > 0 else '#ef4444'
    bdr_col  = {'normal': '#1e4080', 'warning': '#92400e', 'critical': '#7f1d1d'}[health]
    top_col  = {'normal': '#3b82f6', 'warning': '#f59e0b', 'critical': '#ef4444'}[health]
    anom_col = '#ef4444' if anom > 0 else '#22c55e'
    anom_txt = f'{anom} anomaly detected' if anom > 0 else '0 anomalies'

    return f"""
    <div style="background:#0d1f3c;border:1px solid {bdr_col};
                border-top:3px solid {top_col};border-radius:10px;
                padding:14px 16px;margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
        <div>
          <div style="font-size:0.72rem;color:#4a90d9;font-weight:700;
                      letter-spacing:0.8px;text-transform:uppercase;margin-bottom:2px">{label}</div>
          <div style="font-size:0.65rem;color:#475569;line-height:1.4">{desc}</div>
        </div>
        <div style="font-size:0.85rem;flex-shrink:0;margin-left:8px">{badge}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px">
        <div style="text-align:center;background:#060d1a;border-radius:6px;padding:8px 4px">
          <div style="font-size:0.6rem;color:#64748b;margin-bottom:3px;letter-spacing:0.5px">CURRENT</div>
          <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;
                      font-family:'JetBrains Mono',monospace;line-height:1">{cur:.3f}</div>
          <div style="font-size:0.6rem;color:#64748b;margin-top:2px">{unit}</div>
        </div>
        <div style="text-align:center;background:#060d1a;border-radius:6px;padding:8px 4px">
          <div style="font-size:0.6rem;color:#64748b;margin-bottom:3px;letter-spacing:0.5px">7D MOVING AVG</div>
          <div style="font-size:1.1rem;font-weight:700;color:#f97316;
                      font-family:'JetBrains Mono',monospace;line-height:1">{ma7:.3f}</div>
          <div style="font-size:0.6rem;color:{delta_col};margin-top:2px">{delta_str}</div>
        </div>
        <div style="text-align:center;background:#060d1a;border-radius:6px;padding:8px 4px">
          <div style="font-size:0.6rem;color:#64748b;margin-bottom:3px;letter-spacing:0.5px">30D MONTHLY AVG</div>
          <div style="font-size:1.1rem;font-weight:700;color:#a855f7;
                      font-family:'JetBrains Mono',monospace;line-height:1">{ma30:.3f}</div>
          <div style="font-size:0.6rem;color:#64748b;margin-top:2px">{unit}</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;font-size:0.65rem">
        <div style="background:#0a1628;border-radius:4px;padding:5px 6px;text-align:center">
          <div style="color:#64748b;margin-bottom:1px">σ (Std Dev)</div>
          <div style="color:#e2e8f0;font-weight:600">{std:.3f}</div>
        </div>
        <div style="background:#0a1628;border-radius:4px;padding:5px 6px;text-align:center">
          <div style="color:#64748b;margin-bottom:1px">CV%</div>
          <div style="color:#e2e8f0;font-weight:600">{cv:.1f}%</div>
        </div>
        <div style="background:#0a1628;border-radius:4px;padding:5px 6px;text-align:center">
          <div style="color:#64748b;margin-bottom:1px">Min – Max</div>
          <div style="color:#e2e8f0;font-weight:600">{mn:.2f}–{mx:.2f}</div>
        </div>
        <div style="background:#0a1628;border-radius:4px;padding:5px 6px;text-align:center">
          <div style="color:#64748b;margin-bottom:1px">Op. Limits</div>
          <div style="color:#22c55e;font-weight:600">{lo}–{hi}</div>
        </div>
      </div>
      <div style="margin-top:8px;font-size:0.63rem;color:{anom_col};text-align:right">
        ⚡ {anom_txt}
      </div>
    </div>"""

# ── Main chart builder ──────────────────────────────────────────────────────
def trend_chart(x, y, col, label, chart_type='trend'):
    s       = pd.Series(y.values if hasattr(y,'values') else y,
                        index=x.values  if hasattr(x,'values') else x)
    s       = _numeric(s)
    s_clean = s.dropna()
    if len(s_clean) < 3:
        return go.Figure()

    lim   = OP_LIMITS.get(col, {})
    lo    = lim.get('lo',    s_clean.min() * 0.95)
    hi    = lim.get('hi',    s_clean.max() * 1.05)
    wlo   = lim.get('warn_lo', lo)
    whi   = lim.get('warn_hi', hi)
    unit  = lim.get('unit', '')
    mu    = s_clean.mean()
    sigma = s_clean.std()
    ucl   = mu + 3 * sigma
    lcl   = mu - 3 * sigma
    ma7   = s_clean.rolling(7,  min_periods=1).mean()
    ma30  = s_clean.rolling(30, min_periods=1).mean()
    anom  = detect_anomalies(s_clean)
    xi    = s_clean.index
    yi    = s_clean.values

    i_max  = s_clean.idxmax()
    i_min  = s_clean.idxmin()
    i_last = xi[-1]
    anom_x = xi[anom]
    anom_y = yi[anom]

    BASE_LAYOUT = dict(
        paper_bgcolor='#0d1f3c',
        plot_bgcolor ='#060d1a',
        font         = dict(family='Inter', color='#94b4d4', size=10),
        margin       = dict(l=55, r=85, t=55, b=45),
        height       = 290,
        hovermode    = 'x unified',
        hoverlabel   = dict(bgcolor='#0d1f3c', bordercolor='#1e3a5f', font=dict(size=10)),
        legend       = dict(bgcolor='rgba(6,13,26,0.85)', bordercolor='#1e3a5f',
                            font=dict(size=9), orientation='h',
                            yanchor='bottom', y=1.01, xanchor='left', x=0),
        title=dict(
            text=f'<b>{label}</b>  <span style="font-size:10px;color:#475569">— {unit}</span>',
            font=dict(size=12, color='#e2e8f0'), x=0, xanchor='left'),
        xaxis = dict(gridcolor='#1e3a5f', showgrid=True, zeroline=False,
                     tickformat='%b %Y', dtick='M1',
                     tickfont=dict(size=9), tickangle=-30,
                     showline=True, linecolor='#1e3a5f',
                     title=dict(text='Date', font=dict(size=9, color='#64748b'))),
        yaxis = dict(gridcolor='#1e3a5f', showgrid=True, zeroline=False,
                     title=dict(text=f'{label} ({unit})', font=dict(size=9, color='#64748b')),
                     tickfont=dict(size=9),
                     showline=True, linecolor='#1e3a5f'),
    )

    # ── TREND CHART ────────────────────────────────────────────────────────
    if chart_type == 'trend':
        fig = go.Figure()

        # Green normal operating zone
        fig.add_hrect(y0=wlo, y1=whi,
                      fillcolor='rgba(34,197,94,0.06)',
                      line=dict(color='rgba(34,197,94,0.25)', width=1, dash='dot'),
                      annotation_text='Normal Zone',
                      annotation_position='top left',
                      annotation_font=dict(color='#22c55e', size=8))

        # Yellow warning bands
        if lo < wlo:
            fig.add_hrect(y0=lo, y1=wlo,
                          fillcolor='rgba(245,158,11,0.04)', line_width=0)
        if whi < hi:
            fig.add_hrect(y0=whi, y1=hi,
                          fillcolor='rgba(245,158,11,0.04)', line_width=0)

        # Raw daily — light blue, low opacity, NO markers
        fig.add_trace(go.Scatter(
            x=xi, y=yi, mode='lines',
            line=dict(color='rgba(120,180,255,0.55)', width=1),
            name='Daily', showlegend=True,
            hovertemplate=f'%{{x|%d %b %Y}}<br>{label}: %{{y:.3f}} {unit}<extra></extra>',
        ))

        # 30-day MA — thin purple dotted
        fig.add_trace(go.Scatter(
            x=xi, y=ma30.values, mode='lines',
            line=dict(color='#a855f7', width=1.3, dash='dot'),
            name='30d MA', opacity=0.75,
            hovertemplate='30d MA: %{y:.3f}<extra></extra>',
        ))

        # 7-day MA — thick orange PRIMARY FOCUS
        fig.add_trace(go.Scatter(
            x=xi, y=ma7.values, mode='lines',
            line=dict(color='#f97316', width=1.2),
            name='7d Moving Avg',
            hovertemplate='7d MA: %{y:.3f}<extra></extra>',
        ))

        # Control lines: UCL, LCL, Mean
        for yval, name, color, dash in [
            (ucl, f'UCL {ucl:.2f}', '#f59e0b', 'dash'),
            (lcl, f'LCL {lcl:.2f}', '#f59e0b', 'dash'),
            (mu,  f'μ {mu:.2f}',    '#475569', 'dot'),
        ]:
            fig.add_hline(y=yval,
                          line=dict(color=color, width=1, dash=dash),
                          annotation_text=name,
                          annotation_position='top right',
                          annotation_font=dict(color=color, size=8))

        # MAX marker — purple triangle up
        fig.add_trace(go.Scatter(
            x=[i_max], y=[s_clean[i_max]],
            mode='markers+text',
            marker=dict(color='#a855f7', size=10, symbol='triangle-up',
                        line=dict(color='white', width=1)),
            text=[f'▲ {s_clean[i_max]:.2f}'],
            textposition='top center',
            textfont=dict(size=8, color='#a855f7'),
            name='Max', showlegend=False,
            hovertemplate=f'MAX: %{{y:.3f}} {unit}<extra></extra>',
        ))

        # MIN marker — cyan triangle down
        fig.add_trace(go.Scatter(
            x=[i_min], y=[s_clean[i_min]],
            mode='markers+text',
            marker=dict(color='#06b6d4', size=10, symbol='triangle-down',
                        line=dict(color='white', width=1)),
            text=[f'▼ {s_clean[i_min]:.2f}'],
            textposition='bottom center',
            textfont=dict(size=8, color='#06b6d4'),
            name='Min', showlegend=False,
            hovertemplate=f'MIN: %{{y:.3f}} {unit}<extra></extra>',
        ))

        # LATEST marker — white circle
        fig.add_trace(go.Scatter(
            x=[i_last], y=[s_clean[i_last]],
            mode='markers+text',
            marker=dict(color='white', size=10, symbol='circle',
                        line=dict(color='#3b82f6', width=2.5)),
            text=[f' {s_clean[i_last]:.2f}'],
            textposition='middle right',
            textfont=dict(size=8.5, color='white',
                          family='JetBrains Mono'),
            name='Latest', showlegend=False,
            hovertemplate=f'Latest: %{{y:.3f}} {unit}<extra></extra>',
        ))

        # Anomaly markers removed per user request

        fig.update_layout(**BASE_LAYOUT)
        fig.update_layout(
            yaxis=dict(
                range=[min(lcl * 0.995, s_clean.min() * 0.995),
                       max(ucl * 1.005, s_clean.max() * 1.005)],
                gridcolor='#1e3a5f', zeroline=False,
                title=dict(text=unit, font=dict(size=9, color='#64748b')),
                tickfont=dict(size=9),
            )
        )
        return fig

    # ── SPC CONTROL CHART + HISTOGRAM ─────────────────────────────────────
    elif chart_type == 'control':
        fig = make_subplots(rows=1, cols=2,
                            column_widths=[0.77, 0.23],
                            horizontal_spacing=0.03)

        fig.add_hrect(y0=wlo, y1=whi,
                      fillcolor='rgba(34,197,94,0.06)',
                      line=dict(color='rgba(34,197,94,0.2)', width=1, dash='dot'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(
            x=xi, y=yi, mode='lines',
            line=dict(color='rgba(120,180,255,0.55)', width=1),
            name='Daily', showlegend=False),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=xi, y=ma7.values, mode='lines',
            line=dict(color='#f97316', width=1.2),
            name='7d MA', showlegend=False),
            row=1, col=1)

        for yval, color, dash in [
            (ucl, '#f59e0b', 'dash'),
            (lcl, '#f59e0b', 'dash'),
            (mu,  '#475569', 'dot'),
        ]:
            fig.add_hline(y=yval,
                          line=dict(color=color, width=1, dash=dash),
                          row=1, col=1)

        # Anomaly markers removed per user request

        # Distribution histogram
        fig.add_trace(go.Histogram(
            y=yi, marker_color='#3b82f6', opacity=0.65,
            nbinsy=25, showlegend=False,
            hovertemplate='Count: %{x}<extra></extra>'),
            row=1, col=2)

        for yval, color, dash in [
            (mu,  '#475569', 'dot'),
            (ucl, '#f59e0b', 'dash'),
            (lcl, '#f59e0b', 'dash'),
        ]:
            fig.add_hline(y=yval,
                          line=dict(color=color, width=1, dash=dash),
                          row=1, col=2)

        fig.update_layout(**BASE_LAYOUT)
        fig.update_layout(
            xaxis =dict(gridcolor='#1e3a5f', tickformat='%b %Y', dtick='M1',
                        tickfont=dict(size=9), tickangle=-30),
            yaxis =dict(gridcolor='#1e3a5f', tickfont=dict(size=9),
                        title=dict(text=f'{label} ({unit})', font=dict(size=9, color='#64748b'))),
            xaxis2=dict(gridcolor='#1e3a5f', tickfont=dict(size=9),
                        title=dict(text='Frequency', font=dict(size=9))),
            yaxis2=dict(gridcolor='#1e3a5f', tickfont=dict(size=9), matches='y'),
        )
        return fig

    # ── BOX PLOT (monthly variability) ────────────────────────────────────
    elif chart_type == 'boxplot':
        df_box = pd.DataFrame({'date': xi, 'val': yi})
        df_box['month_dt']  = pd.to_datetime(df_box['date'])
        df_box['month_lbl'] = df_box['month_dt'].dt.strftime('%b %Y')
        df_box['sort_key']  = df_box['month_dt'].dt.to_period('M').astype(str)
        months_ordered = df_box.sort_values('sort_key')['month_lbl'].unique()

        fig = go.Figure()
        fig.add_hrect(y0=wlo, y1=whi,
                      fillcolor='rgba(34,197,94,0.07)', line_width=0)
        fig.add_hline(y=hi, line=dict(color='#ef4444', width=1, dash='dash'))
        fig.add_hline(y=lo, line=dict(color='#ef4444', width=1, dash='dash'))
        fig.add_hline(y=mu, line=dict(color='#475569', width=1, dash='dot'))

        for mth in months_ordered:
            grp = df_box[df_box['month_lbl'] == mth]['val']
            fig.add_trace(go.Box(
                y=grp, name=mth,
                marker_color='#3b82f6',
                line_color='#60a5fa',
                fillcolor='rgba(59,130,246,0.15)',
                boxmean=True,
                showlegend=False,
                hovertemplate=f'{mth}<br>%{{y:.3f}} {unit}<extra></extra>',
            ))

        fig.update_layout(**BASE_LAYOUT)
        fig.update_layout(
            title=dict(text=f'<b>{label}</b> — Monthly Distribution  ({unit})',
                       font=dict(size=12, color='#e2e8f0'), x=0, xanchor='left'),
            xaxis=dict(gridcolor='#1e3a5f', tickfont=dict(size=9), tickangle=-30,
                       title=dict(text='Month', font=dict(size=9, color='#64748b'))),
            yaxis=dict(gridcolor='#1e3a5f', tickfont=dict(size=9),
                       title=dict(text=f'{label} ({unit})', font=dict(size=9, color='#64748b'))),
        )
        return fig

    return go.Figure()

# ── AI Insights generator ───────────────────────────────────────────────────
def ai_insights(pp_filt, col_map):
    insights = []
    for label, col in col_map.items():
        if col not in pp_filt.columns:
            continue
        try:
            s = _numeric(pp_filt[col]).dropna()
            if len(s) < 14:
                continue
            lim  = OP_LIMITS.get(col, {})
            unit = lim.get('unit', '')
            lo   = lim.get('lo', -np.inf)
            hi   = lim.get('hi',  np.inf)

            ma7_recent = s.tail(7).mean()
            ma7_prior  = s.tail(14).head(7).mean()
            trend_delta= ma7_recent - ma7_prior
            trend_pct  = abs(trend_delta) / (abs(ma7_prior) + 1e-9) * 100

            cv_recent  = s.tail(30).std() / (s.tail(30).mean() + 1e-9) * 100
            cv_full    = s.std() / (s.mean() + 1e-9) * 100

            exceedances= int(((s.tail(30) > hi) | (s.tail(30) < lo)).sum())
        except Exception:
            continue

        if exceedances > 0:
            insights.append(dict(
                level='critical', icon='🔴',
                text=f'**{label}** exceeded operating limits **{exceedances}× in last 30 days**.',
                action=f'Review {label} — {lim.get("desc","check equipment and inputs")}.',
            ))
        elif trend_pct > 3 and trend_delta > 0:
            insights.append(dict(
                level='warning', icon='🟡',
                text=f'**{label}** trending upward (+{trend_delta:.3f} {unit} vs prior 7d).',
                action='Monitor closely — adjust process inputs if trend continues.',
            ))
        elif trend_pct > 3 and trend_delta < 0:
            insights.append(dict(
                level='warning', icon='🟡',
                text=f'**{label}** declining ({trend_delta:.3f} {unit} vs prior 7d).',
                action='Check upstream feed consistency and equipment settings.',
            ))
        elif cv_recent > cv_full * 1.5:
            insights.append(dict(
                level='warning', icon='🟡',
                text=f'**{label}** variability increased — CV {cv_recent:.1f}% vs historical {cv_full:.1f}%.',
                action='Investigate raw mix or equipment inconsistency.',
            ))
        else:
            insights.append(dict(
                level='normal', icon='🟢',
                text=f'**{label}** stable — 7d avg {ma7_recent:.3f} {unit}, CV {cv_recent:.1f}%.',
                action='',
            ))

    order = {'critical': 0, 'warning': 1, 'normal': 2}
    return sorted(insights, key=lambda x: order[x['level']])

# ── MAIN RENDER ─────────────────────────────────────────────────────────────
def render(pp_filt, pp_date, d_from, d_to):
    PROC_COLS = {
        'Machine Speed':    'Machine speed',
        'WGF Speed':        'WGF Speed',
        'ESP Inlet Temp.':  'ESP Inlet Temp.',
        'ESP Inlet Press.': 'ESP Inlet Press.',
        'Feed Rate':        'Feed',
        'Moisture':         'Moisture',
        'BTP Temperature':  'BTP Temperature',
        'Cooler Speed':     'Cooler Speed',
        'Production':       'Production',
        'Basicity (B2)':    'Basicity(B2)',
        '%FeO':             '%FeO',
        '%MgO':             '%MgO',
        '%CaO':             '%CaO',
        '%SiO₂':            '%SiO2',
        '%Al₂O₃':           '%Al2O3',
    }
    available = {k: v for k, v in PROC_COLS.items() if v in pp_filt.columns}

    # ── PLANT HEALTH SCORE ─────────────────────────────────────────────────
    n_crit = n_warn = n_ok = 0
    for k, v in available.items():
        s = pp_filt[v].dropna()
        h, _ = param_health(s, v)
        if   h == 'critical': n_crit += 1
        elif h == 'warning':  n_warn += 1
        else:                 n_ok   += 1
    total        = max(len(available), 1)
    health_score = round((n_ok * 100 + n_warn * 50) / total, 1)
    plant_color  = ('#22c55e' if health_score >= 80
                    else '#f59e0b' if health_score >= 60
                    else '#ef4444')
    date_label   = (f'{d_from.strftime("%d %b")} → {d_to.strftime("%d %b %Y")}')

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0a1628,#0d1f3c);
                border:1px solid #1e3a5f;border-radius:12px;
                padding:18px 24px;margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:0.7rem;color:#4a90d9;letter-spacing:1px;
                      text-transform:uppercase;margin-bottom:4px">
            ⚙️ PLANT PROCESS HEALTH SCORE
          </div>
          <div style="font-size:2.6rem;font-weight:800;color:{plant_color};
                      font-family:'JetBrains Mono',monospace;line-height:1">
            {health_score:.0f}
            <span style="font-size:1rem;color:#64748b;font-weight:400"> / 100</span>
          </div>
          <div style="font-size:0.7rem;color:#475569;margin-top:3px">
            {total} parameters monitored &nbsp;·&nbsp; {date_label}
          </div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap">
          <div style="background:#1c0202;border:1px solid #ef4444;
                      border-radius:8px;padding:10px 22px;text-align:center">
            <div style="font-size:1.8rem;font-weight:800;color:#ef4444;
                        font-family:'JetBrains Mono',monospace">{n_crit}</div>
            <div style="font-size:0.67rem;color:#ef4444;letter-spacing:0.5px">🔴 CRITICAL</div>
          </div>
          <div style="background:#1c1200;border:1px solid #f59e0b;
                      border-radius:8px;padding:10px 22px;text-align:center">
            <div style="font-size:1.8rem;font-weight:800;color:#f59e0b;
                        font-family:'JetBrains Mono',monospace">{n_warn}</div>
            <div style="font-size:0.67rem;color:#f59e0b;letter-spacing:0.5px">🟡 WARNING</div>
          </div>
          <div style="background:#052e16;border:1px solid #22c55e;
                      border-radius:8px;padding:10px 22px;text-align:center">
            <div style="font-size:1.8rem;font-weight:800;color:#22c55e;
                        font-family:'JetBrains Mono',monospace">{n_ok}</div>
            <div style="font-size:0.67rem;color:#22c55e;letter-spacing:0.5px">🟢 NORMAL</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── LEGEND ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:18px;flex-wrap:wrap;background:#0a1628;
                border:1px solid #1e3a5f;border-radius:8px;
                padding:9px 16px;margin-bottom:14px;font-size:0.72rem;color:#94b4d4">
      <span style="color:rgba(120,180,255,0.75)">━ Daily (raw)</span>
      <span style="color:#f97316;font-weight:600">━━ 7d Moving Avg (primary)</span>
      <span style="color:#a855f7">┄ 30d Monthly Avg</span>
      <span style="color:#22c55e">▓ Normal Zone</span>
      <span style="color:#f59e0b">- - UCL / LCL</span>
      <span style="color:#a855f7">▲ Max</span>
      <span style="color:#06b6d4">▼ Min</span>
      <span style="color:#ffffff">● Latest</span>
      <span style="color:#ef4444">✕ Anomaly</span>
    </div>""", unsafe_allow_html=True)

    # ── CONTROLS ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2.5, 1.2, 0.8])
    with c1:
        sel = st.multiselect(
            'Select parameters',
            list(available.keys()),
            default=list(available.keys())[:8],
            placeholder='Choose parameters to monitor...',
            label_visibility='collapsed',
        )
    with c2:
        chart_type = st.selectbox(
            'Chart type',
            ['Trend + Control Lines', 'SPC Control + Distribution', 'Monthly Box Plot'],
            label_visibility='collapsed',
        )
        ctype = {'Trend + Control Lines': 'trend',
                 'SPC Control + Distribution': 'control',
                 'Monthly Box Plot': 'boxplot'}[chart_type]
    with c3:
        compare_mode = st.toggle('Compare Mode', value=False)

    if not sel:
        st.info('Select at least one parameter to monitor.')
        return

    # ── COMPARE MODE ───────────────────────────────────────────────────────
    if compare_mode and len(sel) >= 2:
        fig = go.Figure()
        corr_series = {}
        for i, label in enumerate(sel):
            col = available[label]
            if col not in pp_filt.columns: continue
            d2  = pp_filt[[pp_date, col]].dropna() if pp_date in pp_filt.columns \
                  else pp_filt[[col]].dropna()
            if len(d2):
                d2 = d2.assign(**{col: _numeric(d2[col])}).dropna(subset=[col])
            if len(d2) < 3: continue
            s   = d2[col].reset_index(drop=True)
            x   = (d2[pp_date].reset_index(drop=True)
                   if pp_date in d2.columns
                   else pd.RangeIndex(len(d2)))
            rng = float(s.max() - s.min())
            s_n = (s - s.min()) / (rng + 1e-9) * 100
            ma7 = s_n.rolling(7, min_periods=1).mean()
            color = CLR['compare'][i % len(CLR['compare'])]
            fig.add_trace(go.Scatter(
                x=x, y=s_n.values, mode='lines',
                line=dict(color=color, width=0.8),
                opacity=0.18, name=label, showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=x, y=ma7.values, mode='lines',
                line=dict(color=color, width=2.2),
                name=f'{label} 7d MA', showlegend=False,
                hovertemplate=f'{label}: %{{y:.1f}}%<extra></extra>',
            ))
            corr_series[label] = s

        # Correlation annotation
        if len(corr_series) >= 2:
            keys  = list(corr_series.keys())
            vals  = [corr_series[k] for k in keys]
            cdf   = pd.DataFrame(dict(zip(keys, vals))).corr()
            pairs = [(keys[i], keys[j], cdf.iloc[i, j])
                     for i in range(len(keys))
                     for j in range(i+1, len(keys))
                     if abs(cdf.iloc[i, j]) > 0.3]
            if pairs:
                corr_txt = '  |  '.join(
                    [f'r({a[:5]}, {b[:5]}) = {c:.2f}' for a, b, c in pairs])
                st.markdown(
                    f'<div style="background:#0a1628;border-left:3px solid #3b82f6;'
                    f'padding:7px 14px;font-size:0.73rem;color:#94b4d4;'
                    f'border-radius:0 6px 6px 0;margin-bottom:10px">'
                    f'📊 Significant Correlations: {corr_txt}</div>',
                    unsafe_allow_html=True)

        fig.update_layout(
            paper_bgcolor='#0d1f3c', plot_bgcolor='#060d1a',
            font=dict(family='Inter', color='#94b4d4', size=10),
            margin=dict(l=50, r=20, t=40, b=40), height=360,
            title=dict(text='Parameter Comparison — Normalised 0–100%',
                       font=dict(size=11, color='#e2e8f0'), x=0),
            xaxis=dict(gridcolor='#1e3a5f', tickformat='%b %Y', dtick='M1',
                       tickfont=dict(size=9), tickangle=-30),
            yaxis=dict(gridcolor='#1e3a5f',
                       title=dict(text='Normalised Value (%)',
                                  font=dict(size=9, color='#64748b')),
                       tickfont=dict(size=9)),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # ── PARAMETER CARDS — 2-column grid ────────────────────────────────────
    for i in range(0, len(sel), 2):
        pair = sel[i:i+2]
        col_pair = st.columns(2, gap='medium')
        for j, label in enumerate(pair):
            col_name = available.get(label)
            with col_pair[j]:
                try:
                    if col_name is None or col_name not in pp_filt.columns:
                        st.info(f'⚠️ "{label}" — column not found in the current data.')
                        continue
                    if pp_date in pp_filt.columns:
                        d2 = pp_filt[[pp_date, col_name]].dropna()
                        if len(d2):
                            d2 = d2.assign(**{col_name: _numeric(d2[col_name])}).dropna(subset=[col_name])
                        x  = d2[pp_date]
                    else:
                        d2 = pp_filt[[col_name]].dropna()
                        if len(d2):
                            d2 = d2.assign(**{col_name: _numeric(d2[col_name])}).dropna(subset=[col_name])
                        x  = pd.RangeIndex(len(d2))
                    if len(d2) < 3:
                        st.info(f'⚠️ "{label}" — not enough valid data points to plot '
                                f'({len(d2)} available, need ≥3).')
                        continue
                    s = d2[col_name]
                    fig = trend_chart(x, s, col_name, label, ctype)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f'⚠️ "{label}" could not be plotted: {e}')

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── BOTTOM: AI INSIGHTS + ANOMALY TABLE ────────────────────────────────
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    bot1, bot2 = st.columns([1, 1], gap='medium')

    with bot1:
        st.markdown(
            '<div style="background:#0a1628;border-left:3px solid #a855f7;'
            'padding:8px 14px;font-size:0.78rem;font-weight:700;color:#c084fc;'
            'letter-spacing:0.5px;border-radius:0 6px 6px 0;margin-bottom:10px">'
            '🤖 AI PROCESS INSIGHTS</div>', unsafe_allow_html=True)
        col_map = {k: available[k] for k in sel if k in available}
        insights = ai_insights(pp_filt, col_map)
        shown = 0
        for ins in insights:
            if shown >= 8: break
            bg  = {'critical':'#1c0202','warning':'#1c1200','normal':'#052e16'}[ins['level']]
            brd = {'critical':'#7f1d1d','warning':'#92400e','normal':'#14532d'}[ins['level']]
            tc2 = {'critical':'#fca5a5','warning':'#fde68a','normal':'#86efac'}[ins['level']]
            act = (f'<div style="font-size:0.67rem;color:#64748b;margin-top:4px">'
                   f'→ {ins["action"]}</div>') if ins['action'] else ''
            st.markdown(
                f'<div style="background:{bg};border:1px solid {brd};'
                f'border-radius:7px;padding:9px 13px;margin-bottom:7px">'
                f'<span style="font-size:0.74rem;color:{tc2}">'
                f'{ins["icon"]} {ins["text"]}</span>{act}</div>',
                unsafe_allow_html=True)
            shown += 1

    with bot2:
        st.markdown(
            '<div style="background:#0a1628;border-left:3px solid #ef4444;'
            'padding:8px 14px;font-size:0.78rem;font-weight:700;color:#fca5a5;'
            'letter-spacing:0.5px;border-radius:0 6px 6px 0;margin-bottom:10px">'
            '⚠️ RECENT ANOMALIES LOG</div>', unsafe_allow_html=True)
        anom_rows = []
        for label in sel:
            col_name = available.get(label)
            if not col_name or col_name not in pp_filt.columns:
                continue
            if pp_date in pp_filt.columns:
                d2 = pp_filt[[pp_date, col_name]].dropna()
                dates = d2[pp_date]
            else:
                d2 = pp_filt[[col_name]].dropna()
                dates = pd.Series(range(len(d2)))
            s         = d2[col_name]
            anom_mask = detect_anomalies(s)
            lim       = OP_LIMITS.get(col_name, {})
            hi_lim    = lim.get('hi',  np.inf)
            lo_lim    = lim.get('lo', -np.inf)
            for date_val, val in zip(dates[anom_mask], s[anom_mask]):
                sev = ('Critical' if (val > hi_lim or val < lo_lim) else 'Warning')
                dt  = (pd.to_datetime(date_val).strftime('%d %b %Y')
                       if hasattr(date_val, 'strftime') else str(date_val))
                anom_rows.append({'Date': dt, 'Parameter': label,
                                  'Value': round(val, 3), 'Severity': sev})
        if anom_rows:
            adf = (pd.DataFrame(anom_rows)
                   .sort_values('Date', ascending=False)
                   .head(15)
                   .reset_index(drop=True))
            st.dataframe(
                adf.style
                   .map(
                       lambda v: ('color:#ef4444;font-weight:bold' if v == 'Critical'
                                  else 'color:#f59e0b' if v == 'Warning' else ''),
                       subset=['Severity'])
                   .set_properties(**{
                       'background-color': '#0d1f3c',
                       'color': '#e2e8f0',
                       'font-size': '0.78rem',
                   }),
                use_container_width=True,
                height=340,
            )
        else:
            st.markdown(
                '<div style="background:#052e16;border:1px solid #22c55e;'
                'border-radius:8px;padding:20px;text-align:center;'
                'color:#22c55e;font-size:0.82rem">'
                '✅ No anomalies detected in selected parameters</div>',
                unsafe_allow_html=True)
