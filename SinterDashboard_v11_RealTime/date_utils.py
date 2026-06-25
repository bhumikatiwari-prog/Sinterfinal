"""
date_utils.py
─────────────────────────────────────────────────────────────
Shared, defensive date parsing.

Root-cause bug found while migrating to Sinter_Data.xlsx:
the Process Parameters sheet stores dates as text in DD.MM.YYYY
format (e.g. '09.03.2024'). `pd.to_datetime(series, errors='coerce')`
defaults to a month-first reading, so any row where the day is <=12
gets silently mis-parsed (date is wrong, not NaT) and any row where
the day is >12 fails to parse at all and becomes NaT — over half the
rows in this dataset. That breaks date filtering, charting, and (most
seriously) silently scrambles the row order that the rolling/lag
features in feature_engineering.py depend on.

parse_dates_robust() tries both interpretations and keeps whichever
parses more rows successfully, so it self-adapts to the file's actual
convention instead of assuming one.
"""
import pandas as pd


def parse_dates_robust(series: pd.Series) -> pd.Series:
    """Parse a date-like column without assuming day-first or month-first.

    - If already a real datetime dtype, returned unchanged.
    - Otherwise both the default (month-first) and dayfirst=True
      interpretations are tried; whichever yields more successfully
      parsed (non-NaT) values is returned.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    s = series.astype(str).str.strip()
    # Try explicit formats first to avoid ambiguous-parse warnings
    for fmt in ('%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y',
                '%m/%d/%Y', '%d %b %Y', '%d %B %Y'):
        try:
            parsed = pd.to_datetime(s, format=fmt, errors='coerce')
            if parsed.notna().mean() > 0.80:   # >80% of rows parsed → good fit
                return parsed
        except Exception:
            continue
    # Fallback: let pandas infer, picking whichever interpretation gives more rows
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        default  = pd.to_datetime(s, errors='coerce')
        dayfirst = pd.to_datetime(s, errors='coerce', dayfirst=True)
    if dayfirst.notna().sum() >= default.notna().sum():
        return dayfirst
    return default


def sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse `date_col` robustly and return df sorted chronologically
    (ascending), with a clean reset index. Rows with an unparsable date
    are pushed to the end rather than dropped, so no data is silently lost.
    """
    out = df.copy()
    out[date_col] = parse_dates_robust(out[date_col])
    out = out.sort_values(date_col, na_position='last').reset_index(drop=True)
    return out
