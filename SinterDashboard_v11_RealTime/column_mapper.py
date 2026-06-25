"""
column_mapper.py
─────────────────────────────────────────────────────────────
Fuzzy column-name mapping for uploaded Excel sheets.

Manually-maintained plant export sheets drift over time — extra spaces,
inconsistent punctuation/units, case changes ('%CaO' vs '% CaO' vs 'CaO%').
Rather than hard-failing validation every time a header drifts slightly,
this maps an uploaded sheet's actual column names onto the pipeline's
expected schema wherever the match is unambiguous.
"""
import re
import difflib


def _normalize(name: str) -> str:
    s = str(name).strip().lower()
    s = (s.replace('₂', '2').replace('₃', '3').replace('₄', '4')
           .replace('(', '').replace(')', ''))
    s = re.sub(r'[\s\-_/.%]+', '', s)
    return s


def map_columns(actual_columns, expected_columns, cutoff: float = 0.82) -> dict:
    """Build a {actual_col: expected_col} rename map.

    1) Exact match after normalizing whitespace/punctuation/case/units.
    2) For anything left, fuzzy-match via difflib's SequenceMatcher ratio,
       only accepting matches at or above `cutoff` so a clearly-different
       column never gets silently misassigned.

    Each expected column is used at most once. Columns that already match
    exactly are included in the map too (identity), so callers can detect
    "nothing changed" via `{a: e for a, e in result.items() if a != e}`.
    """
    norm_actual   = {c: _normalize(c) for c in actual_columns}
    norm_expected = {e: _normalize(e) for e in expected_columns}

    rename, used_expected = {}, set()

    # Pass 1 — exact normalized match
    for a, na in norm_actual.items():
        for e, ne in norm_expected.items():
            if e in used_expected:
                continue
            if na == ne:
                rename[a] = e
                used_expected.add(e)
                break

    # Pass 2 — fuzzy match among what's left
    remaining_expected = {e: ne for e, ne in norm_expected.items() if e not in used_expected}
    for a, na in norm_actual.items():
        if a in rename or not remaining_expected:
            continue
        best_e, best_score = None, 0.0
        for e, ne in remaining_expected.items():
            score = difflib.SequenceMatcher(None, na, ne).ratio()
            if score > best_score:
                best_e, best_score = e, score
        if best_e is not None and best_score >= cutoff:
            rename[a] = best_e
            used_expected.add(best_e)
            del remaining_expected[best_e]

    return rename
