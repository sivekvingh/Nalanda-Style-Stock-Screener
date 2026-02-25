#!/usr/bin/env python3
"""
Cross-verify Nalanda screener data with SEC EDGAR (10-year financials).

Uses the SEC's public Company Facts API (no API key). Data is from actual 10-K
filings — the source of truth for US public companies.

Usage:
    python nalanda_verify.py AAPL
    python nalanda_verify.py AAPL MSFT GOOGL
    python nalanda_verify.py --list   # show ticker->CIK mapping only

10-year data:
    - SEC returns up to 10 years of annual 10-K data (ROCE from Operating Income,
      Total Assets, Current Liabilities). Cached under .sec_cache/.
    - yfinance often returns only ~4 years; the main screener now uses up to 10
      when available (nalanda_screener.py).

If you get 403 Forbidden from SEC, set SEC_UA in this file to include your
contact (see https://www.sec.gov/os/accessing-edgar-data).

Requirements: same as nalanda_screener (yfinance, pandas). Uses stdlib urllib.
"""

import json
import os
import sys
from datetime import date
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Install: pip install yfinance pandas")
    sys.exit(1)

# SEC requires a descriptive User-Agent; see https://www.sec.gov/os/accessing-edgar-data
SEC_UA = "Mozilla/5.0 (compatible; NalandaScreener/1.0; +https://www.sec.gov/developer; educational)"
SEC_BASE = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_DIR = os.path.join(os.path.dirname(__file__) or ".", ".sec_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
TICKER_CIK_CACHE = os.path.join(CACHE_DIR, "company_tickers.json")
FACTS_CACHE_DIR = os.path.join(CACHE_DIR, "companyfacts")


def _fetch(url, use_cache=True, cache_path=None):
    """Fetch URL with SEC User-Agent. Optionally cache to file."""
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    req = Request(url, headers={"User-Agent": SEC_UA})
    try:
        with urlopen(req, timeout=25) as r:
            data = json.loads(r.read().decode())
    except (HTTPError, URLError) as e:
        print(f"  HTTP error: {e}")
        return None
    if use_cache and cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=0)
        except Exception:
            pass
    return data


def get_ticker_cik_map():
    """Return dict ticker_upper -> CIK (zero-padded 10 digits)."""
    raw = _fetch(TICKERS_URL, use_cache=True, cache_path=TICKER_CIK_CACHE)
    if not raw:
        return {}
    out = {}
    for v in raw.values():
        if isinstance(v, dict):
            ticker = v.get("ticker")
            cik = v.get("cik_str")
            if ticker is not None and cik is not None:
                out[str(ticker).strip().upper()] = str(int(cik)).zfill(10)
    return out


def get_company_facts(cik):
    """Fetch Company Facts JSON for a CIK. Cached under .sec_cache/companyfacts/."""
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    path = os.path.join(FACTS_CACHE_DIR, f"CIK{cik}.json")
    return _fetch(url, use_cache=True, cache_path=path)


def _annual_10k_values(facts, concept_aliases, units_key="USD"):
    """
    From SEC company facts, extract annual 10-K values for a concept.
    concept_aliases: list of us-gaap concept names to try (e.g. OperatingIncomeLoss).
    Returns dict: year -> value (one value per fiscal year, from 10-K).
    """
    if not facts or "facts" not in facts:
        return {}
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in concept_aliases:
        if name not in gaap:
            continue
        obj = gaap[name]
        units = obj.get("units", {})
        if units_key not in units:
            continue
        by_year = {}
        for item in units[units_key]:
            form = (item.get("form") or "").upper()
            fp = (item.get("fp") or "").upper()
            if form != "10-K":
                continue
            # FY = fiscal year (income statement); instant facts (balance sheet) may have no fp
            if fp and fp != "FY":
                continue
            fy = item.get("fy")
            if fy is None:
                end = item.get("end")
                if end:
                    fy = int(end[:4])
            if fy is None:
                continue
            val = item.get("val")
            if val is None:
                continue
            # Keep latest reported value per year (in case of amendments)
            end = item.get("end") or ""
            if fy not in by_year or end > (by_year.get("_ends", {}).get(fy) or ""):
                by_year[fy] = val
                if "_ends" not in by_year:
                    by_year["_ends"] = {}
                by_year["_ends"][fy] = end
        if by_year:
            return {k: v for k, v in by_year.items() if k != "_ends"}
    return {}


def compute_sec_roce(facts):
    """
    Compute ROCE = EBIT / (Total Assets - Current Liabilities) from SEC facts.
    EBIT ~ Operating Income. Returns list of (year, roce_decimal) sorted newest first.
    """
    ebit_years = _annual_10k_values(facts, [
        "OperatingIncomeLoss",
        "OperatingIncome",
        "ProfitLossFromOperatingActivities",
    ])
    assets_years = _annual_10k_values(facts, [
        "Assets",
        "AssetsAbstract",
    ])
    # Assets is often under "Assets" with no units in some feeds; check "USD" or no key
    if not assets_years:
        gaap = facts.get("facts", {}).get("us-gaap", {})
        if "Assets" in gaap:
            units = gaap["Assets"].get("units", {})
            for uk in ["USD", "usd"]:
                if uk in units:
                    by_year = {}
                    for item in units[uk]:
                        if (item.get("form") or "").upper() == "10-K" and (item.get("fp") or "").upper() == "FY":
                            fy = item.get("fy") or (int(item["end"][:4]) if item.get("end") else None)
                            if fy and item.get("val") is not None:
                                by_year[fy] = item["val"]
                    if by_year:
                        assets_years = by_year
                    break

    liab_years = _annual_10k_values(facts, [
        "LiabilitiesCurrent",
        "CurrentLiabilities",
    ])

    if not ebit_years or not assets_years or not liab_years:
        return []

    years_common = sorted(set(ebit_years) & set(assets_years) & set(liab_years), reverse=True)
    out = []
    for yr in years_common[:10]:
        ebit = ebit_years[yr]
        assets = assets_years[yr]
        liab = liab_years[yr]
        cap_emp = assets - liab
        if cap_emp <= 0:
            continue
        roce = max(-1.0, min(5.0, ebit / cap_emp))
        out.append((yr, roce))
    return out


def get_yf_roce_history(ticker):
    """Get ROCE history from yfinance (same logic as screener)."""
    t = yf.Ticker(ticker)
    info = t.info
    inc = t.financials
    bs = t.balance_sheet
    if inc is None or bs is None or inc.empty or bs.empty:
        return []

    def _fs_val(df, col, names):
        for name in names:
            try:
                v = df.loc[name, col]
                if v is not None and not pd.isna(v) and v != 0:
                    return float(v)
            except Exception:
                continue
        return None

    shared = [c for c in inc.columns if c in bs.columns][:10]
    results = []
    for col in shared:
        ebit = _fs_val(inc, col, [
            "Operating Income", "EBIT", "Operating Income Or Loss",
            "Total Operating Income As Reported",
        ])
        ta = _fs_val(bs, col, ["Total Assets"])
        cl = _fs_val(bs, col, [
            "Current Liabilities", "Total Current Liabilities",
            "Current Liabilities Net Minority Interest",
        ])
        if ebit is None or ta is None or cl is None:
            continue
        cap_emp = ta - cl
        if cap_emp <= 0:
            continue
        roce = max(-1.0, min(5.0, ebit / cap_emp))
        results.append((col.year, roce))
    return results


def run_verify(tickers):
    """Run comparison for a list of tickers. Print report."""
    ticker_to_cik = get_ticker_cik_map()
    if not ticker_to_cik:
        print("Could not load SEC ticker->CIK mapping.")
        return

    for ticker in tickers:
        t = ticker.strip().upper()
        cik = ticker_to_cik.get(t)
        if not cik:
            print(f"\n{ticker}: no CIK found on SEC (not a US registrant or ticker typo?).")
            continue

        print(f"\n{'='*60}")
        print(f"  {t}  (CIK {cik})")
        print("="*60)

        facts = get_company_facts(cik)
        sec_roce = compute_sec_roce(facts) if facts else []
        yf_roce = get_yf_roce_history(t)

        if not sec_roce:
            print("  SEC: no annual 10-K ROCE computed (missing OperatingIncome/Assets/Liabilities).")
        else:
            print("  SEC (10-K annual) ROCE:")
            for yr, r in sec_roce[:10]:
                print(f"    {yr}  {r*100:6.1f}%")

        if not yf_roce:
            print("  yfinance: no ROCE history.")
        else:
            print("  yfinance ROCE:")
            for yr, r in yf_roce[:10]:
                print(f"    {yr}  {r*100:6.1f}%")

        # Cross-check: same years
        sec_years = {yr for yr, _ in sec_roce}
        yf_years = {yr for yr, _ in yf_roce}
        common_years = sorted(sec_years & yf_years, reverse=True)
        if common_years:
            print("  Comparison (same year):")
            for yr in common_years[:5]:
                sec_val = next((r for y, r in sec_roce if y == yr), None)
                yf_val = next((r for y, r in yf_roce if y == yr), None)
                if sec_val is not None and yf_val is not None:
                    diff = (yf_val - sec_val) * 100
                    ok = "✓" if abs(diff) < 2 else "⚠"
                    print(f"    {yr}  SEC {sec_val*100:.1f}%  yf {yf_val*100:.1f}%  diff {diff:+.1f}pp  {ok}")
        else:
            print("  (No overlapping fiscal years to compare.)")
        print()


def main():
    if "--list" in sys.argv:
        ticker_to_cik = get_ticker_cik_map()
        for t in sorted(ticker_to_cik.keys())[:50]:
            print(f"  {t}  CIK {ticker_to_cik[t]}")
        print(f"  ... ({len(ticker_to_cik)} total). Omit --list to verify tickers.")
        return

    tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tickers:
        print("Usage: python nalanda_verify.py TICKER [TICKER ...]")
        print("       python nalanda_verify.py --list   # show ticker->CIK map")
        print("Example: python nalanda_verify.py AAPL MSFT")
        sys.exit(1)

    run_verify(tickers)


if __name__ == "__main__":
    main()
