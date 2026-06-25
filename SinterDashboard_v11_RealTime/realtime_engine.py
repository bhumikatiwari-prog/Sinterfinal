"""
realtime_engine.py
─────────────────────────────────────────────────────────────
Real-time data polling, auto-refresh, and live model reloading.

Architecture:
  - Polls artifacts.pkl mtime every N minutes (configurable in sidebar)
  - If mtime changed  → clears st.cache_resource → reloads models seamlessly
  - If new Excel file uploaded → triggers background retrain → reloads
  - Shows live "data age" indicator in sidebar and top nav
  - Uses streamlit-autorefresh for the polling loop; falls back to
    a manual "Refresh Now" button if the package isn't installed
"""
import os, time, pickle, hashlib, threading
import pandas as pd, numpy as np
import streamlit as st

# ── Try to import streamlit-autorefresh (optional dependency) ─────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# ── Constants ─────────────────────────────────────────────────────────────
REFRESH_OPTIONS = {
    'Off':        0,
    '1 minute':   60,
    '5 minutes':  300,
    '10 minutes': 600,
    '30 minutes': 1800,
}
DEFAULT_REFRESH = '5 minutes'


# ── Artifact mtime tracking ────────────────────────────────────────────────
def get_artifact_mtime(dashboard_dir: str) -> float:
    """Return mtime of artifacts.pkl, or 0 if it doesn't exist."""
    path = os.path.join(dashboard_dir, 'artifacts', 'artifacts.pkl')
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def get_data_age_str(mtime: float) -> str:
    """Human-readable age of the last training run."""
    if mtime == 0:
        return 'Never trained'
    age_s = time.time() - mtime
    if age_s < 120:      return f'{int(age_s)}s ago'
    if age_s < 3600:     return f'{int(age_s/60)}m ago'
    if age_s < 86400:    return f'{int(age_s/3600)}h ago'
    return f'{int(age_s/86400)}d ago'


def fmt_datetime(ts: float) -> str:
    """Format a Unix timestamp as 'dd Mon HH:MM'."""
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime('%d %b %H:%M')


# ── Background retrain thread ──────────────────────────────────────────────
def _retrain_thread(data_path: str, dashboard_dir: str, status_key: str):
    """Run model_training.train() in a background thread.
    Writes progress to st.session_state[status_key].
    """
    try:
        sys_path = os.path.dirname(os.path.abspath(__file__))
        import sys; sys.path.insert(0, sys_path)
        import model_training as mt

        st.session_state[status_key] = {'status':'running','pct':10,
                                         'msg':'Loading and cleaning data...'}
        pp, sq = mt.load_clean_data(data_path)

        st.session_state[status_key] = {'status':'running','pct':30,
                                         'msg':'Feature engineering...'}
        from feature_engineering import engineer
        sq_fe = engineer(sq, is_pp=False)
        pp_fe = engineer(pp, is_pp=True)

        st.session_state[status_key] = {'status':'running','pct':55,
                                         'msg':'Training models (RF/XGB/LGB/CAT)...'}
        artifacts_dir = os.path.join(dashboard_dir, 'artifacts')
        art = mt._run_training(pp, sq, artifacts_dir)

        st.session_state[status_key] = {'status':'done','pct':100,
                                         'msg':'Training complete — reloading dashboard...'}
    except Exception as e:
        st.session_state[status_key] = {'status':'error','pct':0,
                                         'msg':f'Training failed: {e}'}


def trigger_retrain(data_path: str, dashboard_dir: str) -> str:
    """Start a background retrain thread. Returns the session_state key
    to poll for progress."""
    key = 'retrain_status'
    st.session_state[key] = {'status':'starting','pct':0,'msg':'Starting...'}
    t = threading.Thread(
        target=_retrain_thread,
        args=(data_path, dashboard_dir, key),
        daemon=True,
    )
    t.start()
    return key


# ── Sidebar real-time controls ─────────────────────────────────────────────
def render_sidebar_controls(dashboard_dir: str) -> dict:
    """
    Renders the real-time control panel in the sidebar.
    Returns a dict with keys:
      refresh_interval_s: int   (0 = off)
      should_reload:      bool  (True if mtime changed since last render)
      data_age_str:       str
      last_trained_fmt:   str
    """
    current_mtime = get_artifact_mtime(dashboard_dir)
    last_mtime    = st.session_state.get('_last_art_mtime', 0.0)
    should_reload = (current_mtime > last_mtime) and (last_mtime > 0)

    if current_mtime != last_mtime:
        st.session_state['_last_art_mtime'] = current_mtime

    st.markdown(
        '<div style="font-size:0.72rem;color:#4a90d9;letter-spacing:1px;'
        'text-transform:uppercase;margin-bottom:8px;font-weight:600">'
        '🔄 REAL-TIME SETTINGS</div>', unsafe_allow_html=True)

    # Auto-refresh interval
    interval_label = st.selectbox(
        'Auto-refresh interval',
        list(REFRESH_OPTIONS.keys()),
        index=list(REFRESH_OPTIONS.keys()).index(DEFAULT_REFRESH),
        key='_refresh_interval',
        label_visibility='collapsed',
    )
    interval_s = REFRESH_OPTIONS[interval_label]

    # Data age display
    age_str      = get_data_age_str(current_mtime)
    trained_fmt  = fmt_datetime(current_mtime) if current_mtime else '—'
    age_color    = ('#22c55e' if current_mtime and time.time()-current_mtime < 3600
                    else '#f59e0b' if current_mtime else '#ef4444')

    st.markdown(
        f'<div style="background:#0a1628;border:1px solid #1e3a5f;'
        f'border-radius:8px;padding:10px 12px;margin:6px 0">'
        f'<div style="font-size:0.65rem;color:#64748b;margin-bottom:3px">DATA AGE</div>'
        f'<div style="font-size:0.9rem;font-weight:700;color:{age_color}">{age_str}</div>'
        f'<div style="font-size:0.63rem;color:#475569;margin-top:2px">'
        f'Last trained: {trained_fmt}</div>'
        f'</div>', unsafe_allow_html=True)

    # Manual refresh
    if st.button('🔄 Refresh Now', key='_manual_refresh', use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    # Retrain on current file
    data_path = os.path.join(dashboard_dir, 'Sinter_Data.xlsx')
    if st.button('🚀 Retrain on Sinter_Data.xlsx',
                 key='_retrain_btn', use_container_width=True):
        trigger_retrain(data_path, dashboard_dir)
        st.rerun()

    # Retrain progress
    rs = st.session_state.get('retrain_status')
    if rs:
        status = rs['status']
        pct    = rs['pct']
        msg    = rs['msg']
        if status == 'running':
            st.progress(pct / 100, text=msg)
        elif status == 'done':
            st.success(msg)
            # Clear and reload once
            if st.session_state.get('_retrain_reload_done') != pct:
                st.session_state['_retrain_reload_done'] = pct
                st.cache_resource.clear()
                st.session_state.pop('retrain_status', None)
                st.rerun()
        elif status == 'error':
            st.error(msg)
            if st.button('Clear error', key='_clear_err'):
                st.session_state.pop('retrain_status', None)
                st.rerun()

    return {
        'refresh_interval_s': interval_s,
        'should_reload':      should_reload,
        'data_age_str':       age_str,
        'last_trained_fmt':   trained_fmt,
        'current_mtime':      current_mtime,
    }


# ── Auto-refresh polling loop (inject into app.py top) ────────────────────
def inject_autorefresh(interval_s: int):
    """
    If streamlit-autorefresh is available, use it to trigger st.rerun()
    every interval_s seconds. Otherwise silently do nothing — the manual
    Refresh Now button remains the fallback.
    """
    if interval_s <= 0 or not HAS_AUTOREFRESH:
        return
    st_autorefresh(interval=interval_s * 1000, key='_autorefresh_counter')


# ── Smart cache loader ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts_cached(artifacts_dir: str, _mtime: float):
    """
    Loads artifacts.pkl. The `_mtime` parameter is intentionally NOT cached
    itself — it forces a cache miss (and a real reload) every time the file
    changes, without needing to call st.cache_resource.clear() globally.

    Callers should pass `_mtime=get_artifact_mtime(dashboard_dir)` to get
    automatic invalidation when the file is updated by a retrain.
    """
    art_path = os.path.join(artifacts_dir, 'artifacts.pkl')
    if not os.path.exists(art_path):
        return None
    with open(art_path, 'rb') as f:
        return pickle.load(f)


# ── Live data status banner ────────────────────────────────────────────────
def render_live_status_badge(rt_info: dict, data_updated: bool):
    """Inline badge shown in the top nav bar area."""
    age   = rt_info['data_age_str']
    color = ('#22c55e' if 'ago' in age and 'm ago' not in age and 'h ago' not in age
             else '#f59e0b' if rt_info['current_mtime'] > 0 else '#ef4444')
    live  = '🟢 LIVE' if data_updated else '📊 STATIC'
    return (f'<div class="nav-badge" style="border-color:{color}">'
            f'{live} · {age}</div>')
