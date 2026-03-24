#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         NALANDA-STYLE INDIAN STOCK SCREENER  v1 (INR)           ║
║         yfinance — NSE (.NS) · no API key required              ║
║         Framework: Pulak Prasad / Nalanda Capital               ║
╚══════════════════════════════════════════════════════════════════╝

Same principles as the US screener; all currency and units in INR (₹).

TICKER LIST (default: official NSE Nifty 500 CSV, cached 7 days):
  - Default: https://archives.nseindia.com/content/indices/ind_nifty500list.csv
  - If NSE fetch fails: built-in Nifty 50 fallback list.
  - Optional: python nalanda_screener_inr.py --local-tickers
      reads indian_tickers.txt or nifty50_tickers.txt from current dir (override).
  - Optional: python nalanda_screener_inr.py --investing
      Investing.com S&P CNX 500 (needs requests + beautifulsoup4).

  To use Investing.com (Nifty 500 list):
    pip install requests beautifulsoup4
    python nalanda_screener_inr.py --investing
  If the site returns 403, save the page in your browser (Save Page As → HTML)
  and run: INVESTING_HTML=/path/to/saved.html python nalanda_screener_inr.py --investing

SETUP:
    uv venv .venv
    uv pip install --python .venv/bin/python yfinance pandas tabulate colorama
    source .venv/bin/activate

RUN:
    python nalanda_screener_inr.py

ENTRY CHECKLIST (one ticker):
    python nalanda_screener_inr.py --entry RELIANCE
    (use NSE symbol; .NS is added automatically)

DEBUG:
    python nalanda_screener_inr.py --debug RELIANCE
"""

import yfinance as yf
import pandas as pd
import time, os, sys, json, re, io
from contextlib import redirect_stderr
from datetime import date, datetime

try:
    from tabulate import tabulate; HAS_TAB = True
except ImportError: HAS_TAB = False

try:
    import requests as _req
    from bs4 import BeautifulSoup
    HAS_INVESTING_DEPS = True
except ImportError:
    HAS_INVESTING_DEPS = False

try:
    from colorama import init, Fore, Style; init(autoreset=True)
except ImportError:
    class Fore:
        RED=GREEN=YELLOW=CYAN=LIGHTGREEN_EX=MAGENTA=WHITE=""
    class Style: RESET_ALL=""

# ── CONFIG (INR) ─────────────────────────────────────────────────────────────
PAUSE      = 0.6
MAX_STOCKS = None
OUTPUT_DIR = "."
MASTER_CSV    = os.path.join(OUTPUT_DIR, "nalanda_master_inr.csv")
SCANNED_FILE  = os.path.join(OUTPUT_DIR, ".scanned_tickers_inr.json")
DEBUG         = "--debug" in sys.argv

# Currency and thresholds (INR)
CURRENCY_SYMBOL   = "₹"
MIN_MKT_CAP_INR   = 70_000_000_000   # ₹70,000 Cr (~$850M equivalent) — hard gate
HURDLE_RISK_PREMIUM         = 0.03
REVERSE_DCF_DISCOUNT_RATE    = 0.10
RSI_OVERSOLD                 = 35
QUALITY_ROCE_MIN             = 20
QUALITY_NET_DEBT_EBITDA_MAX  = 1.0
QUALITY_FCF_NI_MIN           = 0.80

# ── INDUSTRY CLASSIFICATION (India) ───────────────────────────────────────────
VOLATILE_INDUSTRIES = {
    "Semiconductors", "Semiconductor Equipment & Materials",
    "Airlines",
    "Oil & Gas Exploration & Production", "Oil & Gas Refining & Marketing",
    "Oil & Gas Equipment & Services", "Oil & Gas Integrated",
    "Oil, Gas & Consumable Fuels",
    "Steel", "Aluminum", "Copper", "Gold", "Silver", "Coal",
    "Metals & Mining",
    "Biotechnology",
    "Auto Manufacturers", "Auto Parts", "Automobile and Auto Components",
    "Marine Shipping",
    "Paper & Paper Products", "Commodity Chemicals",
}

CONGLOMERATE_KEYWORDS = [
    "conglomerate", "diversified", "multi-sector", "holding company"
]

# ── INDIAN TICKERS — NSE official Nifty 500, or local file, or fallback ───────
_NIFTY500_CACHE_FILE = os.path.join(OUTPUT_DIR, ".nifty500_cache.json")
_NIFTY_CACHE_DAYS = 7
INVESTING_CNX500_URL = "https://in.investing.com/indices/s-p-cnx-500-components"
INVESTING_CACHE_FILE = os.path.join(OUTPUT_DIR, ".investing_nse_cache.json")
INVESTING_CACHE_DAYS = 14
TICKER_FILES = [
    "indian_tickers.txt",
    "nifty50_tickers.txt",
]

def _normalize_ticker(s):
    """Ensure ticker has .NS for NSE (yfinance)."""
    s = (s or "").strip()
    if not s or s.startswith("#"):
        return None
    if ".NS" in s.upper() or ".BO" in s.upper():
        return s
    return s + ".NS"


def get_nifty_tickers():
    """
    Fetch the current NIFTY 500 constituent list directly from the official NSE India archives.
    Caches the result to avoid hitting the NSE servers excessively.
    Tickers are suffixed with .NS for Yahoo Finance compatibility.

    Returns:
        (list[str], str): tickers and a short label for reports ("Nifty 500" or "Nifty 50 (fallback)").
    """
    if os.path.exists(_NIFTY500_CACHE_FILE):
        try:
            with open(_NIFTY500_CACHE_FILE) as f:
                cached = json.load(f)
            cached_date = date.fromisoformat(cached["date"])
            if (date.today() - cached_date).days < _NIFTY_CACHE_DAYS:
                return cached["tickers"], "Nifty 500"
        except Exception:
            pass

    print(f"  {Fore.CYAN}Fetching official NIFTY 500 list from NSE India...", end=" ", flush=True)
    try:
        import requests as _req
        import io

        # Official NSE CSV endpoint for NIFTY 500
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

        # NSE aggressively blocks default Python user agents, so we mimic a standard browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        resp = _req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # Read the CSV directly into pandas
        df = pd.read_csv(io.StringIO(resp.text))

        # Extract the 'Symbol' column and append .NS for Yahoo Finance
        tickers = df["Symbol"].str.strip().tolist()
        tickers = [t + ".NS" for t in tickers]

        # Deduplicate while preserving order
        seen_t = set()
        tickers = [t for t in tickers if not (t in seen_t or seen_t.add(t))]

        with open(_NIFTY500_CACHE_FILE, "w") as f:
            json.dump({"date": date.today().isoformat(), "tickers": tickers}, f)

        print(f"{Fore.GREEN}OK ({len(tickers)} stocks)")
        return tickers, "Nifty 500"

    except Exception as e:
        print(f"{Fore.YELLOW}failed ({e}) — using built-in Nifty 50 fallback list")
        return _NIFTY_FALLBACK, "Nifty 50 (fallback)"


def _fetch_investing_components_html():
    """Fetch Investing.com S&P CNX 500 components page. Returns HTML str or None (e.g. 403)."""
    if not HAS_INVESTING_DEPS:
        return None
    html_file = os.environ.get("INVESTING_HTML") or os.environ.get("INVESTING_HTML_FILE")
    if html_file and os.path.isfile(html_file):
        try:
            with open(html_file, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None
    try:
        sess = _req.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://in.investing.com/",
        })
        r = sess.get(INVESTING_CNX500_URL, timeout=20)
        if r.status_code == 200 and len(r.text) > 10000:
            return r.text
    except Exception:
        pass
    return None


def _parse_investing_components_html(html):
    """
    Parse Investing.com components page HTML; return list of equity slugs (e.g. 'reliance-industries').
    """
    if not HAS_INVESTING_DEPS or not html:
        return []
    slugs = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/equities/" in href and "indices" not in href and "?" not in href.split("/equities/")[-1]:
                # e.g. /equities/reliance-industries or https://in.investing.com/equities/...
                parts = href.strip("/").split("/")
                for i, p in enumerate(parts):
                    if p == "equities" and i + 1 < len(parts):
                        slug = parts[i + 1].split("?")[0].strip()
                        if slug and slug not in slugs and len(slug) > 2:
                            slugs.append(slug)
                        break
        # Deduplicate preserving order
        seen = set()
        out = []
        for s in slugs:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    except Exception:
        return []


def _slug_to_nse(slug, cache, session):
    """
    Resolve Investing.com equity slug to NSE symbol. Fetches equity page and looks for NSEI:SYMBOL.
    Uses and updates cache dict; session is requests.Session.
    """
    if slug in cache:
        return cache[slug]
    url = f"https://in.investing.com/equities/{slug}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            cache[slug] = None
            return None
        # Investing.com embeds NSE symbol as NSEI:RELIANCE in links (e.g. /pro/NSEI:RELIANCE/explorer)
        m = re.search(r"NSEI:([A-Z0-9.-]+)", r.text)
        if m:
            sym = m.group(1).strip()
            cache[slug] = sym
            return sym
    except Exception:
        pass
    cache[slug] = None
    return None


def _load_investing_cache():
    """Load slug→NSE cache from disk if not stale."""
    if not os.path.isfile(INVESTING_CACHE_FILE):
        return {}
    try:
        with open(INVESTING_CACHE_FILE) as f:
            data = json.load(f)
        if data.get("date"):
            d = date.fromisoformat(data["date"])
            if (date.today() - d).days > INVESTING_CACHE_DAYS:
                return {}
        return data.get("symbols", {})
    except Exception:
        return {}


def _save_investing_cache(cache):
    try:
        with open(INVESTING_CACHE_FILE, "w") as f:
            json.dump({"date": date.today().isoformat(), "symbols": cache}, f)
    except Exception:
        pass


def get_indian_tickers():
    """
    Load NSE tickers. Priority:
    1. --investing → Investing.com S&P CNX 500 (if deps + fetch OK).
    2. --local-tickers → indian_tickers.txt or nifty50_tickers.txt if present in OUTPUT_DIR.
    3. Default → official NSE Nifty 500 via get_nifty_tickers() (cached), else Nifty 50 fallback.

    Returns:
        (list[str], str): tickers and a label for HTML/console (e.g. "Nifty 500").
    """
    use_investing = "--investing" in sys.argv
    if use_investing and HAS_INVESTING_DEPS:
        html = _fetch_investing_components_html()
        if html:
            slugs = _parse_investing_components_html(html)
            if slugs:
                print(f"  {Fore.CYAN}Investing.com: found {len(slugs)} equity links, resolving NSE symbols...")
                cache = _load_investing_cache()
                sess = _req.Session()
                sess.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://in.investing.com/",
                })
                tickers = []
                for i, slug in enumerate(slugs):
                    sym = _slug_to_nse(slug, cache, sess)
                    if sym:
                        t = _normalize_ticker(sym)
                        if t and t not in tickers:
                            tickers.append(t)
                    if (i + 1) % 50 == 0:
                        print(f"  {Fore.CYAN}  resolved {i+1}/{len(slugs)}...")
                    time.sleep(0.25)
                _save_investing_cache(cache)
                if tickers:
                    print(f"  {Fore.GREEN}Loaded {len(tickers)} NSE tickers from Investing.com S&P CNX 500")
                    return tickers, "S&P CNX 500"
        elif use_investing:
            print(f"  {Fore.YELLOW}Investing.com: fetch failed or 403. Save the page as HTML and set INVESTING_HTML=/path/to/file")
    if use_investing and not HAS_INVESTING_DEPS:
        print(f"  {Fore.YELLOW}Install requests and beautifulsoup4 to use --investing")
    # Local file only when explicitly requested (otherwise Nifty 500 is default)
    if "--local-tickers" in sys.argv:
        for fname in TICKER_FILES:
            path = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                tickers = []
                for line in lines:
                    t = _normalize_ticker(line)
                    if t and t not in tickers:
                        tickers.append(t)
                if tickers:
                    print(f"  {Fore.GREEN}Loaded {len(tickers)} tickers from {fname} (--local-tickers)")
                    return tickers, f"Custom ({fname})"
            except Exception as e:
                print(f"  {Fore.YELLOW}Could not read {fname}: {e}")
        print(f"  {Fore.YELLOW}--local-tickers: no {TICKER_FILES[0]} / {TICKER_FILES[1]} found — using Nifty 500")
    # Default: official NSE Nifty 500 (cached), or built-in fallback on failure
    return get_nifty_tickers()


# Built-in Nifty 50 (fallback when NSE fetch and local file fail).
_NIFTY50_FALLBACK = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]
_NIFTY_FALLBACK = _NIFTY50_FALLBACK


INDIAN_TICKERS, UNIVERSE_LABEL = get_indian_tickers()

# ── YFINANCE DATA FETCH (same as US; bond yield differs) ───────────────────────
def _fs_val(df, col, names):
    for name in names:
        try:
            v = df.loc[name, col]
            if v is not None and not pd.isna(v) and v != 0:
                return float(v)
        except Exception:
            continue
    return None


def _calc_historical_roce(t, info):
    try:
        inc = t.financials
        bs  = t.balance_sheet
        if inc is None or bs is None or inc.empty or bs.empty:
            return [], "unknown", 99, True
        shared_cols = [c for c in inc.columns if c in bs.columns][:10]
        results = []
        for col in shared_cols:
            try:
                ebit = _fs_val(inc, col, [
                    "Operating Income", "EBIT", "Operating Income Or Loss",
                    "Total Operating Income As Reported",
                ])
                if ebit is None:
                    continue
                total_assets = _fs_val(bs, col, ["Total Assets"])
                curr_liab    = _fs_val(bs, col, [
                    "Current Liabilities", "Total Current Liabilities",
                    "Current Liabilities Net Minority Interest",
                ])
                if total_assets is None or curr_liab is None:
                    continue
                cap_emp = total_assets - curr_liab
                if cap_emp <= 0:
                    continue
                roce = max(-1.0, min(5.0, ebit / cap_emp))
                results.append((str(col.year), roce))
            except Exception:
                continue
        op_margin = info.get("operatingMargins", 0) or 0
        revenue   = info.get("totalRevenue",     0) or 0
        ttm_ebit  = op_margin * revenue
        if ttm_ebit and not bs.empty:
            most_recent = bs.columns[0]
            ta = _fs_val(bs, most_recent, ["Total Assets"])
            cl = _fs_val(bs, most_recent, [
                "Current Liabilities", "Total Current Liabilities",
                "Current Liabilities Net Minority Interest",
            ])
            if ta and cl and (ta - cl) > 0:
                ttm_roce = max(-1.0, min(5.0, ttm_ebit / (ta - cl)))
                results.insert(0, ("TTM", ttm_roce))
        if shared_cols:
            filing_date = str(shared_cols[0].date())
            months_old   = (date.today() - shared_cols[0].date()).days // 30
            is_stale    = months_old > 15
        else:
            filing_date, months_old, is_stale = "unknown", 99, True
        return results, filing_date, months_old, is_stale
    except Exception:
        return [], "unknown", 99, True


def _calc_growth(t):
    try:
        inc = t.financials
        if inc is None or inc.empty:
            return None, None
        def get_row(names):
            for name in names:
                if name in inc.index:
                    row = inc.loc[name].dropna()
                    if len(row) >= 2:
                        return row
            return None
        rev_row  = get_row(["Total Revenue", "Revenue"])
        ni_row   = get_row(["Net Income", "Net Income Common Stockholders",
                            "Net Income Including Noncontrolling Interests"])
        def cagr(series):
            vals = [float(v) for v in series.sort_index().values]
            start, end = vals[0], vals[-1]
            n = len(vals) - 1
            if start <= 0 or end <= 0 or n <= 0:
                return None
            return (end / start) ** (1.0 / n) - 1
        return cagr(rev_row), cagr(ni_row)
    except Exception:
        return None, None


def _calc_cfo_quality(t):
    try:
        cf  = t.cashflow
        inc = t.financials
        if cf is None or cf.empty or inc is None or inc.empty:
            return None
        def get_row(df, names):
            for name in names:
                if name in df.index:
                    return df.loc[name].dropna()
            return None
        cfo_row = get_row(cf, [
            "Operating Cash Flow", "Cash From Operations",
            "Total Cash From Operating Activities",
            "Net Cash Provided By Operating Activities",
        ])
        ni_row = get_row(inc, ["Net Income", "Net Income Common Stockholders"])
        if cfo_row is None or ni_row is None:
            return None
        common = [c for c in cfo_row.index if c in ni_row.index]
        if not common:
            return None
        total_cfo = sum(float(cfo_row[c]) for c in common)
        total_ni  = sum(float(ni_row[c])  for c in common)
        return total_cfo / total_ni if total_ni > 0 else None
    except Exception:
        return None


_BOND_YIELD_CACHE: dict = {}

def _fetch_bond_yield():
    """
    India 10-year government bond yield (decimal). Cached per run.
    Tries yfinance; fallback 7% if unavailable. Suppresses yfinance stderr (404/delisted noise).
    """
    if "val" in _BOND_YIELD_CACHE:
        return _BOND_YIELD_CACHE["val"]
    for symbol in ("IN10YT=RR", "IN10Y.NS", "^IN10Y"):
        try:
            with redirect_stderr(io.StringIO()):
                t    = yf.Ticker(symbol)
                info = t.fast_info
                raw  = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
                if raw is None:
                    info2 = t.info
                    raw   = info2.get("regularMarketPrice") or info2.get("previousClose") or 0
                if raw and float(raw) > 0:
                    val = float(raw)
                    if val > 1:
                        val = val / 100
                    _BOND_YIELD_CACHE["val"] = val
                    return val
        except Exception:
            continue
    _BOND_YIELD_CACHE["val"] = 0.07
    return 0.07


def _calc_hist_pe_median(t, info):
    try:
        inc = t.financials
        if inc is None or inc.empty:
            return None
        ni_row = None
        for name in ["Net Income", "Net Income Common Stockholders",
                     "Net Income Including Noncontrolling Interests"]:
            if name in inc.index:
                ni_row = inc.loc[name].dropna()
                break
        if ni_row is None or len(ni_row) < 2:
            return None
        shares = (info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0)
        if not shares:
            return None
        hist = t.history(period="10y", interval="1mo")
        if hist.empty:
            return None
        pe_vals = []
        for col, ni_val in ni_row.items():
            year = col.year
            ni   = float(ni_val)
            if ni <= 0:
                continue
            eps = ni / float(shares)
            year_prices = hist[hist.index.year == year]["Close"]
            if year_prices.empty:
                continue
            price = float(year_prices.iloc[-1])
            pe    = price / eps
            if 1 < pe < 200:
                pe_vals.append(pe)
        if len(pe_vals) < 2:
            return None
        return float(sorted(pe_vals)[len(pe_vals) // 2])
    except Exception:
        return None


def _calc_hist_ev_fcf_median(t, info):
    try:
        cf  = t.cashflow
        bs  = t.balance_sheet
        qbs = t.quarterly_balance_sheet
        if cf is None or cf.empty or bs is None or bs.empty:
            return None, 0
        fcf_row = None
        if "Free Cash Flow" in cf.index:
            fcf_row = cf.loc["Free Cash Flow"].dropna()
        else:
            ocf_row = capex_row = None
            for n in ["Operating Cash Flow", "Cash Flow From Operations"]:
                if n in cf.index: ocf_row = cf.loc[n].dropna(); break
            for n in ["Capital Expenditure", "Capital Expenditures"]:
                if n in cf.index: capex_row = cf.loc[n].dropna(); break
            if ocf_row is not None and capex_row is not None:
                fcf_row = ocf_row + capex_row
        if fcf_row is None or len(fcf_row) < 2:
            return None, 0
        debt_row = cash_row = None
        for n in ["Total Debt", "Short Long Term Debt"]:
            if n in bs.index: debt_row = bs.loc[n]; break
        for n in ["Cash Cash Equivalents And Short Term Investments",
                  "Cash And Cash Equivalents", "Cash"]:
            if n in bs.index: cash_row = bs.loc[n]; break
        if debt_row is None or cash_row is None:
            return None, 0
        shares_series = None
        if qbs is not None and not qbs.empty:
            for n in ["Ordinary Shares Number", "Share Issued", "Common Stock"]:
                if n in qbs.index:
                    shares_series = qbs.loc[n].dropna()
                    break
        fallback_shares = (info.get("sharesOutstanding") or
                          info.get("impliedSharesOutstanding") or 0)
        fy_end_month = fcf_row.index[0].month
        hist = t.history(period="10y", interval="1mo")
        if hist is None or hist.empty:
            return None, 0
        fy_prices = hist[hist.index.month == fy_end_month]["Close"].groupby(
            hist[hist.index.month == fy_end_month].index.year).last()
        import math
        def _safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                f = float(val)
                return default if math.isnan(f) else f
            except (TypeError, ValueError):
                return default
        ratios = []
        for col in fcf_row.index:
            yr = col.year
            fcf_val = _safe_float(fcf_row.get(col))
            if fcf_val <= 0:
                continue
            debt_val = _safe_float(debt_row.get(col, 0))
            cash_val = _safe_float(cash_row.get(col, 0))
            if shares_series is not None:
                yr_shares = shares_series[shares_series.index.year == yr]
                shr = float(yr_shares.iloc[-1]) if not yr_shares.empty else fallback_shares
            else:
                shr = fallback_shares
            if not shr or shr <= 0:
                continue
            price_val = fy_prices.get(yr)
            if price_val is None or float(price_val) <= 0:
                continue
            price_val = float(price_val)
            mkt_cap = price_val * shr
            ev_val  = mkt_cap + debt_val - cash_val
            if ev_val <= 0:
                continue
            ratio = ev_val / fcf_val
            if 1 < ratio < 200:
                ratios.append(ratio)
        if len(ratios) < 2:
            return None, 0
        ratios.sort()
        return round(ratios[len(ratios) // 2], 1), len(ratios)
    except Exception:
        return None, 0


def _calc_technical(t, price):
    out = {"rsi14": None, "sma200": None, "priceVsSma200": "—"}
    try:
        hist = t.history(period="2y", interval="1d")
        if hist is None or hist.empty or len(hist) < 15:
            return out
        close = hist["Close"].astype(float)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        if not rsi.empty and not pd.isna(rsi.iloc[-1]):
            out["rsi14"] = round(float(rsi.iloc[-1]), 1)
        if len(close) >= 200:
            sma200 = float(close.rolling(200).mean().iloc[-1])
            out["sma200"] = round(sma200, 2)
            if price and sma200:
                out["priceVsSma200"] = "below" if price < sma200 else "above"
    except Exception:
        pass
    return out


def get_yf_data(ticker):
    """Fetch data for one ticker (use NSE symbol with .NS, e.g. RELIANCE.NS)."""
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        if not info:
            return {"__err": "empty info"}, {}, {}, []
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        if not price:
            return {"__err": "no price"}, {}, {}, []
        hist_roce, filing_date, data_age_months, is_stale = _calc_historical_roce(t, info)
        if hist_roce:
            roce_ttm = hist_roce[0][1]
        else:
            op_margin = info.get("operatingMargins", 0) or 0
            roce_ttm  = max(-1.0, min(5.0, op_margin * 1.5))
        de_raw = info.get("debtToEquity")
        if de_raw is None or de_raw <= 0:
            de = 9.99
        else:
            de = de_raw / 100
        fcf = info.get("freeCashflow", 0) or 0
        rev = info.get("totalRevenue", 0) or 0
        ev  = info.get("enterpriseValue", 0) or 0
        bs = t.balance_sheet
        cf = t.cashflow
        inc = t.financials
        most_recent = bs.columns[0] if bs is not None and not bs.empty else None
        total_debt = info.get("totalDebt")
        if total_debt is None and bs is not None and most_recent is not None:
            st_debt = _fs_val(bs, most_recent, ["Short Term Debt", "Short Long Term Debt", "Current Debt"])
            lt_debt = _fs_val(bs, most_recent, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
            total_debt = (st_debt or 0) + (lt_debt or 0) or None
        cash = info.get("cash") or info.get("totalCash")
        if cash is None and bs is not None and most_recent is not None:
            cash = _fs_val(bs, most_recent, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"])
        net_debt = (float(total_debt) - float(cash)) if (total_debt is not None and cash is not None) else None
        ebitda = info.get("ebitda")
        if ebitda is None and cf is not None and not cf.empty and inc is not None and not inc.empty:
            dep = _fs_val(cf, cf.columns[0], ["Depreciation", "Depreciation And Amortization", "Depreciation Depletion And Amortization"])
            op_inc = _fs_val(inc, inc.columns[0], ["Operating Income", "EBIT", "Operating Income Or Loss"])
            if dep is not None and op_inc is not None:
                ebitda = op_inc + dep
        ni_ttm = info.get("netIncomeToCommon") or info.get("netIncome")
        if ni_ttm is None and inc is not None and not inc.empty:
            ni_ttm = _fs_val(inc, inc.columns[0], ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"])
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0
        fcf_yield = (fcf / ev) if ev and ev > 0 else None
        fcf_to_ni = (fcf / ni_ttm) if ni_ttm and ni_ttm > 0 else None
        net_debt_to_ebitda = (net_debt / ebitda) if ebitda and ebitda > 0 else None
        tech = _calc_technical(t, float(price))
        rev_cagr, earn_cagr = _calc_growth(t)
        cfo_quality         = _calc_cfo_quality(t)
        hist_pe_median      = _calc_hist_pe_median(t, info)
        bond_yield          = _fetch_bond_yield()
        hist_ev_fcf_median, hist_ev_fcf_years = _calc_hist_ev_fcf_median(t, info)
        km = {
            "returnOnCapitalEmployedTTM": roce_ttm,
            "evToFreeCashFlowTTM":        (ev / fcf) if fcf > 0 else 0,
            "freeCashFlowTTM":            fcf,
            "revenueTTM":                 rev,
            "netDebt":                    net_debt,
            "ebitda":                     ebitda,
            "netIncomeTTM":               ni_ttm,
            "fcfYield":                   fcf_yield,
            "sharesOutstanding":          shares,
            "enterpriseValue":             ev,
            "netDebtToEbitda":            net_debt_to_ebitda,
            "fcfToNiRatio":               fcf_to_ni,
            "revenueCagr":                rev_cagr,
            "earningsCagr":               earn_cagr,
            "cfoQualityRatio":            cfo_quality,
            "histPeMedian":               hist_pe_median,
            "bondYield10yr":              bond_yield,
            "histEvFcfMedian":            hist_ev_fcf_median,
            "histEvFcfYears":             hist_ev_fcf_years,
            "filingDate":                 filing_date,
            "dataAgeMonths":              data_age_months,
            "isStale":                    is_stale,
        }
        rat = {
            "debtToEquityRatioTTM":     de,
            "fcfMarginTTM":             (fcf / rev) if rev > 0 else 0,
            "grossProfitMarginTTM":     info.get("grossMargins", 0) or 0,
            "operatingProfitMarginTTM": info.get("operatingMargins", 0) or 0,
            "netProfitMarginTTM":       info.get("profitMargins", 0) or 0,
            "priceToEarningsRatioTTM":  info.get("trailingPE", 0) or 0,
        }
        prof = {
            "symbol":      ticker,
            "companyName": info.get("longName") or info.get("shortName", ticker),
            "sector":      info.get("sector", "--"),
            "industry":    info.get("industry", "--"),
            "mktCap":      info.get("marketCap", 0),
            "price":       price,
            "rsi14":       tech["rsi14"],
            "sma200":      tech["sma200"],
            "priceVsSma200": tech["priceVsSma200"],
        }
        return km, rat, prof, hist_roce
    except Exception as e:
        return {"__err": str(e)}, {}, {}, []


def run_debug(ticker="RELIANCE.NS"):
    if not ticker.upper().endswith(".NS") and not ticker.upper().endswith(".BO"):
        ticker = _normalize_ticker(ticker) or "RELIANCE.NS"
    print(f"\n{Fore.CYAN}DEBUG MODE (INR) — {ticker}\n")
    t    = yf.Ticker(ticker)
    info = t.info
    print(f"{'='*60}\n  yfinance Ticker.info\n{'='*60}")
    for k, v in sorted(info.items()):
        if v not in (None, "", 0, 0.0, [], {}):
            print(f"  {k:<45} {v}")
    km, rat, prof, hist_roce = get_yf_data(ticker)
    if "__err" in km:
        print(f"\n  {Fore.RED}Error: {km['__err']}")
    else:
        print(f"\n{Fore.CYAN}  --- Nalanda metrics (INR) ---")
        for lbl, r in hist_roce:
            print(f"    {lbl:<6} {r*100:6.1f}%")
        if not hist_roce:
            print(f"    fallback  {km.get('returnOnCapitalEmployedTTM',0)*100:.1f}%")
        print(f"\n  Bond Yield (India 10Y) {km.get('bondYield10yr',0.07)*100:.1f}%")
        print(f"  Market Cap (INR)       {capmkt(prof.get('mktCap',0))}")
        score, verdict, tags, det = nalanda_score(km, rat, prof, hist_roce)
        print(f"\n  Score: {score}/100  →  {verdict}")
    print(f"\n{Fore.GREEN}Debug complete.")
    sys.exit(0)


# ── SCORING (INR: market cap in INR, capmkt in ₹ Cr) ───────────────────────────
def safe(d, *keys, default=0):
    for k in keys:
        v = d.get(k)
        if v is not None and v != 0 and v != "":
            try: return float(v)
            except: pass
    return default


def capmkt(v):
    """Format market cap in INR: ₹ X Cr / ₹ X Lakh Cr."""
    if not v:
        return "—"
    v = float(v)
    if v >= 1e12:
        return f"{CURRENCY_SYMBOL}{v/1e12:.1f}L Cr"
    if v >= 1e7:
        return f"{CURRENCY_SYMBOL}{v/1e7:.0f} Cr"
    return f"{CURRENCY_SYMBOL}{v/1e5:.0f}L"


def nalanda_score(km, rat, prof, hist_roce=None):
    """Same as US; hard gate uses MIN_MKT_CAP_INR and capmkt is INR."""
    score, tags, details = 0, [], {}
    industry = prof.get("industry", "") or ""
    sector   = prof.get("sector",   "") or ""
    mkt_cap = prof.get("mktCap", 0) or 0
    details["Market Cap"] = capmkt(mkt_cap)
    if mkt_cap < MIN_MKT_CAP_INR:
        details["Market Cap"] += f" — below {CURRENCY_SYMBOL}70B threshold"
        return 0, "FAIL", ["Below mkt cap"], details
    roce_ttm = safe(km, "returnOnCapitalEmployedTTM") * 100
    details["ROCE TTM"] = f"{roce_ttm:.1f}%"
    if   roce_ttm >= 40: score += 25; tags.append("Elite ROCE")
    elif roce_ttm >= 30: score += 22; tags.append("High ROCE")
    elif roce_ttm >= 20: score += 18; tags.append("Good ROCE")
    elif roce_ttm >= 12: score += 8
    else:                score += 0
    hist_pts  = 0
    hist_note = "no history"
    if hist_roce and len(hist_roce) >= 2:
        fy_only  = [(yr, r) for yr, r in hist_roce if yr != "TTM"]
        all_vals = [r for _, r in fy_only] if fy_only else [r for _, r in hist_roce]
        n = len(all_vals)
        avg_roce = (sum(all_vals) / n) * 100
        min_roce = min(all_vals) * 100
        yrs_above_20 = sum(1 for r in all_vals if r * 100 >= 20)
        details["ROCE Avg"] = f"{avg_roce:.1f}% ({n}yr)"
        details["ROCE Min"] = f"{min_roce:.1f}%"
        details["ROCE History"] = "  |  ".join(f"{yr}: {r*100:.0f}%" for yr, r in hist_roce)
        if yrs_above_20 == n and avg_roce >= 25:
            hist_pts = 30; hist_note = f"All {n}yr ≥20%, avg {avg_roce:.0f}%"
            tags.append(f"Consistent ROCE ({n}yr)")
        elif yrs_above_20 >= n - 1 and avg_roce >= 22:
            hist_pts = 24; hist_note = f"{yrs_above_20}/{n}yr ≥20%"
            tags.append(f"Steady ROCE ({n}yr)")
        elif yrs_above_20 >= n // 2 and avg_roce >= 18:
            hist_pts = 16; hist_note = f"{yrs_above_20}/{n}yr ≥20%, avg {avg_roce:.0f}%"
        elif avg_roce >= 15:
            hist_pts = 8; hist_note = f"avg ROCE {avg_roce:.0f}%"
        elif avg_roce >= 10:
            hist_pts = 3
            hist_note = f"avg ROCE {avg_roce:.0f}%"
        else:
            hist_pts = 0
            hist_note = f"avg ROCE {avg_roce:.0f}%"
        if min_roce < 0:
            hist_pts = max(0, hist_pts - 12); tags.append("ROCE Negative (some years)")
        elif min_roce < 8:
            hist_pts = max(0, hist_pts - 8); tags.append("ROCE Volatile")
    else:
        hist_note = "TTM only — history unavailable"
    score += hist_pts
    details["ROCE Consistency"] = f"{hist_pts}pts — {hist_note}"
    de_raw = rat.get("debtToEquityRatioTTM")
    de = float(de_raw) if de_raw is not None else 99.0
    details["D/E"] = f"{de:.2f}x" if de < 9.0 else "Neg. Equity / N/A"
    if   de < 0:     score += 20; tags.append("Net Cash")
    elif de <= 0.10: score += 20; tags.append("Near Zero Debt")
    elif de <= 0.30: score += 15; tags.append("Low Debt")
    elif de <= 0.50: score += 8;  tags.append("Modest Debt")
    elif de <= 1.00: score += 3
    elif de <= 2.00: score += 0
    else:            score -= 5;  tags.append("High Debt")
    rev_cagr  = km.get("revenueCagr")
    earn_cagr = km.get("earningsCagr")
    details["Revenue CAGR"]  = (f"{rev_cagr*100:.1f}% ({_cagr_years(km)}yr)" if rev_cagr  is not None else "— (unavailable)")
    details["Earnings CAGR"] = (f"{earn_cagr*100:.1f}% ({_cagr_years(km)}yr)" if earn_cagr is not None else "— (unavailable)")
    if rev_cagr is not None and earn_cagr is not None:
        if   rev_cagr >= 0.10 and earn_cagr >= 0.10: score += 10; tags.append("Strong Growth")
        elif rev_cagr >= 0.10 or  earn_cagr >= 0.10: score += 5;  tags.append("Partial Growth")
        elif rev_cagr >= 0.05 and earn_cagr >= 0.05: score += 3
        elif rev_cagr < 0 or earn_cagr < 0: score -= 3; tags.append("⚠ Declining Growth")
    cfo_ratio = km.get("cfoQualityRatio")
    if cfo_ratio is not None:
        details["CFO / Net Income"] = f"{cfo_ratio:.2f}x"
        if   cfo_ratio >= 1.0: score += 5; tags.append("High Cash Quality")
        elif cfo_ratio >= 0.8: score += 2
        elif cfo_ratio <  0.6: score -= 2; tags.append("⚠ Accruals Risk")
    else:
        details["CFO / Net Income"] = "— (unavailable)"
    industry_penalty = 0
    industry_flag    = ""
    for vi in VOLATILE_INDUSTRIES:
        if vi.lower() in industry.lower():
            industry_penalty = 20
            industry_flag    = f"Volatile industry ({industry})"
            tags.append("Volatile Industry")
            break
    if not industry_penalty:
        volatile_sectors = {"Energy", "Materials", "Utilities"}
        if sector in volatile_sectors:
            industry_penalty = 10
            industry_flag    = f"Commodity/utility sector ({sector})"
            tags.append("Commodity Sector")
    for kw in CONGLOMERATE_KEYWORDS:
        if kw in industry.lower() or kw in sector.lower():
            industry_penalty = max(industry_penalty, 20)
            industry_flag    = "Conglomerate (Prasad avoids)"
            if "Conglomerate" not in " ".join(tags):
                tags.append("Conglomerate — Avoid")
            break
    if industry_penalty:
        score -= industry_penalty
        details["Industry Flag"] = f"−{industry_penalty}pts: {industry_flag}"
    pe          = safe(rat, "priceToEarningsRatioTTM")
    hist_pe_med = km.get("histPeMedian")
    bond_yield  = km.get("bondYield10yr", 0.07)
    earn_yield  = (1.0 / pe) if pe > 1 else 0
    evcf        = safe(km, "evToFreeCashFlowTTM")
    details["P/E TTM"]           = f"{pe:.1f}x" if pe > 0 else "—"
    details["P/E Median (4yr)"]  = f"{hist_pe_med:.1f}x" if hist_pe_med else "—"
    details["Earnings Yield"]    = f"{earn_yield*100:.1f}%" if earn_yield > 0 else "—"
    details["Bond Yield (10yr)"]  = f"{bond_yield*100:.1f}%"
    details["EV/FCF"]            = f"{evcf:.1f}x" if evcf > 0 else "—"
    if roce_ttm >= 20 and pe > 1:
        pe_below_median  = bool(hist_pe_med) and pe < hist_pe_med
        yield_beats_bond = earn_yield > bond_yield
        if   pe_below_median and yield_beats_bond: score += 10; tags.append("Cheap vs History + Bonds")
        elif yield_beats_bond: score += 5;  tags.append("Cheap vs Bonds")
        elif pe_below_median: score += 3;  tags.append("Below Historical P/E")
    net_debt   = km.get("netDebt")
    ebitda     = km.get("ebitda")
    ni_ttm     = km.get("netIncomeTTM")
    fcf        = km.get("freeCashFlowTTM") or 0
    fcf_ni     = km.get("fcfToNiRatio")
    nd_ebitda  = km.get("netDebtToEbitda")
    quality_roce = roce_ttm >= QUALITY_ROCE_MIN
    quality_debt = (net_debt is not None and ebitda is not None and ebitda > 0 and (net_debt / ebitda) < QUALITY_NET_DEBT_EBITDA_MAX) or (net_debt is not None and net_debt <= 0) or (ebitda is None or ebitda <= 0)
    fcf_ni_note = ""
    if ni_ttm is None or ni_ttm <= 0 or fcf_ni is None:
        quality_fcf_ni = None
    elif fcf_ni < 0:
        quality_fcf_ni = False
        fcf_ni_note = " ⚠ neg FCF"
    elif fcf_ni >= QUALITY_FCF_NI_MIN:
        quality_fcf_ni = True
    else:
        roce_declining = False
        if hist_roce and len(hist_roce) >= 3:
            fy_vals = [r for yr, r in hist_roce if yr != "TTM"]
            if len(fy_vals) >= 2:
                recent_roce = fy_vals[0]
                oldest_roce = fy_vals[-1]
                roce_min_decimal = QUALITY_ROCE_MIN / 100
                roce_declining = (oldest_roce > recent_roce * 1.10 and recent_roce < roce_min_decimal * 1.25)
        if roce_declining:
            quality_fcf_ni = False
            fcf_ni_note = " ⚠ accruals?"
        else:
            quality_fcf_ni = True
            fcf_ni_note = " (reinvesting)"
    quality_pass = quality_roce and quality_debt and (quality_fcf_ni is not False)
    details["Quality Pass"] = "Y" if quality_pass else "N"
    details["Net Debt/EBITDA"] = f"{nd_ebitda:.2f}x" if nd_ebitda is not None else "—"
    details["FCF/NI"] = (f"{fcf_ni:.2f}x{fcf_ni_note}" if fcf_ni is not None else "—")
    if quality_pass:
        tags.append("Entry Quality Pass")
    ev    = km.get("enterpriseValue") or 0
    fcf_yield = km.get("fcfYield")
    hurdle = (km.get("bondYield10yr") or 0.07) + HURDLE_RISK_PREMIUM
    net_d  = km.get("netDebt") or 0
    shares = km.get("sharesOutstanding") or 0
    b_reinv = None
    if ni_ttm and ni_ttm > 0 and fcf is not None:
        b_reinv = max(0.0, min(1.0, 1 - (fcf / float(ni_ttm))))
    roce_decimal = (roce_ttm / 100.0) if roce_ttm is not None else None
    g_sust = (roce_decimal * b_reinv) if (roce_decimal is not None and b_reinv is not None) else None
    fcf_floor = None
    if fcf and hurdle > 0 and shares > 0:
        floor_equity = (fcf / hurdle) - net_d
        if floor_equity > 0:
            fcf_floor = floor_equity / float(shares)
    fcf_fair_value = fcf_fair_value_method = None
    fcf_fv_low = fcf_fv_high = None
    fcf_fv_mult_low = fcf_fv_mult_high = None
    FCF_NI_GORDON_THRESHOLD = 0.90
    fcf_ni_ratio = (fcf / float(ni_ttm)) if (ni_ttm and ni_ttm > 0 and fcf) else None
    use_ev_fcf_reversion = (fcf_ni_ratio is not None and fcf_ni_ratio > FCF_NI_GORDON_THRESHOLD)
    if fcf and shares > 0:
        if not use_ev_fcf_reversion and g_sust is not None and g_sust > 0 and hurdle > 0:
            g_low  = max(0.0, min(g_sust, hurdle - 0.03))
            g_high = min(g_sust, hurdle - 0.01)
            for g_val, attr in [(g_low, "low"), (g_high, "high")]:
                spread = hurdle - g_val
                eq = (fcf / spread) - net_d
                if eq > 0:
                    price = eq / float(shares)
                    mult  = min((price * float(shares) + net_d) / fcf, 80.0)
                    if attr == "low": fcf_fv_low, fcf_fv_mult_low = price, mult
                    else:             fcf_fv_high, fcf_fv_mult_high = price, mult
            if fcf_fv_high is not None:
                fcf_fair_value = fcf_fv_high
                fcf_fair_value_method = "Gordon"
        elif use_ev_fcf_reversion:
            hist_ev_fcf = km.get("histEvFcfMedian")
            hist_ev_fcf_yrs = km.get("histEvFcfYears") or 0
            if hist_ev_fcf and hist_ev_fcf > 0 and hist_ev_fcf_yrs >= 2:
                mult_high = hist_ev_fcf
                mult_low  = hist_ev_fcf * 0.80
                for mult, attr in [(mult_low, "low"), (mult_high, "high")]:
                    eq = (fcf * mult) - net_d
                    if eq > 0:
                        price = eq / float(shares)
                        if attr == "low": fcf_fv_low, fcf_fv_mult_low = price, mult
                        else:             fcf_fv_high, fcf_fv_mult_high = price, mult
                if fcf_fv_high is not None:
                    fcf_fair_value = fcf_fv_high
                    fcf_fair_value_method = f"EV/FCF {hist_ev_fcf:.0f}× median ({hist_ev_fcf_yrs}yr)"
    def _fv_range_str(lo, hi, mult_lo, mult_hi):
        if lo is not None and hi is not None:
            return f"{CURRENCY_SYMBOL}{lo:.0f} – {CURRENCY_SYMBOL}{hi:.0f}  ({mult_lo:.0f}× – {mult_hi:.0f}× FCF)"
        if hi is not None:
            return f"{CURRENCY_SYMBOL}{hi:.0f}  ({mult_hi:.0f}× FCF)"
        return "—"
    details["FCF Yield"]      = f"{fcf_yield*100:.1f}%" if fcf_yield is not None else "—"
    details["Hurdle"]         = f"{hurdle*100:.1f}%"
    details["FCF Floor"]      = (f"{CURRENCY_SYMBOL}{fcf_floor:.0f}  ({(fcf_floor * float(shares) + net_d) / fcf:.0f}× FCF)" if fcf_floor is not None and fcf else "—")
    details["FCF Fair Value"] = _fv_range_str(fcf_fv_low, fcf_fv_high, fcf_fv_mult_low, fcf_fv_mult_high)
    details["FCF FV Method"]  = fcf_fair_value_method or "—"
    g_implied = None
    if ev and fcf and (ev + fcf) > 0:
        g_implied = (float(ev) * REVERSE_DCF_DISCOUNT_RATE - float(fcf)) / (float(ev) + float(fcf))
        g_implied = max(-0.5, min(0.5, g_implied))
    rev_dcf_signal = "—"
    if g_implied is not None and g_sust is not None:
        if g_sust <= 0:
            rev_dcf_signal = "—"
        else:
            rev_dcf_signal = "Buy" if g_implied < g_sust else "Not Buy"
    details["Implied Growth"]    = f"{g_implied*100:.1f}%" if g_implied is not None else "—"
    details["Sustainable Growth"] = f"{g_sust*100:.1f}%"  if g_sust  is not None else "—"
    details["Reverse DCF"]       = rev_dcf_signal
    rsi14 = prof.get("rsi14")
    sma200 = prof.get("sma200")
    price_vs_sma = prof.get("priceVsSma200") or "—"
    technical_ready = (rsi14 is not None and sma200 is not None and rsi14 < RSI_OVERSOLD and price_vs_sma == "below")
    details["RSI(14)"] = f"{rsi14:.1f}" if rsi14 is not None else "—"
    details["200d SMA"] = f"{CURRENCY_SYMBOL}{sma200:.2f}" if sma200 is not None else "—"
    details["Price vs 200d SMA"] = price_vs_sma
    details["Technical"] = "Ready" if technical_ready else ("Not yet" if (rsi14 is not None or sma200 is not None) else "—")
    if km.get("isStale"):
        tags.append("⚠ Stale Data")
        details["Data Quality"] = f"⚠ Filing {km.get('filingDate','?')} ({km.get('dataAgeMonths','?')}mo old) — verify manually"
    else:
        details["Data Quality"] = f"✓ Filing {km.get('filingDate','?')} ({km.get('dataAgeMonths','?')}mo old)"
    score = max(0, min(100, score))
    volatile_industry = industry_penalty >= 15
    roce_ok = (roce_ttm >= 20 and (not hist_roce or hist_pts >= 16) and not volatile_industry)
    verdict = (
        "STRONG PASS" if score >= 78 and roce_ok else
        "PASS"        if score >= 62 and roce_ok else
        "WATCH"       if score >= 45             else
        "FAIL"
    )
    return score, verdict, tags, details


def _cagr_years(km):
    return 3


# ── HTML REPORT (INR: ₹ everywhere) ─────────────────────────────────────────────
def write_html(df, path, run_time, n_scanned, universe_label="Nifty 500"):
    strong = len(df[df.Verdict == "STRONG PASS"])
    passed = len(df[df.Verdict == "PASS"])
    watch  = len(df[df.Verdict == "WATCH"])
    jsd = {}
    for r in df.itertuples():
        jsd[r.Ticker] = {f: str(getattr(r, f, "")) for f in df.columns}
    rows = ""
    for i, (_, r) in enumerate(df.iterrows()):
        sc  = "#34d399" if r.Score >= 78 else "#fbbf24" if r.Score >= 62 else "#fb923c"
        vc  = {"STRONG PASS":"#34d399","PASS":"#a3e635","WATCH":"#fbbf24"}.get(r.Verdict,"#94a3b8")
        bg  = "#0f172a" if i % 2 == 0 else "#0a1220"
        rd  = r.to_dict()
        tgs = " ".join(
            f"<span style='background:#1e3a2f;color:#6ee7b7;border:1px solid #065f46;"
            f"border-radius:999px;padding:1px 8px;font-size:10px'>{t}</span>"
            for t in str(r.Tags).split(", ") if t
        )
        rows += (
            f"<tr style='background:{bg}' onmouseover=\"this.style.background='#1e293b'\" "
            f"onmouseout=\"this.style.background='{bg}'\" onclick=\"sd('{r.Ticker}')\">"
            f"<td style='font-weight:bold;color:#f1f5f9'>{r.Ticker}</td>"
            f"<td style='color:#94a3b8;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{r.Name}</td>"
            f"<td style='color:#64748b;font-size:11px'>{r.Sector}</td>"
            f"<td style='color:{sc};font-weight:bold'>{rd.get('ROCE','')}</td>"
            f"<td>{rd.get('ROCEAvg','')}</td><td>{rd.get('DE','')}</td>"
            f"<td>{rd.get('RevGrowth','—')}</td><td>{rd.get('EarnGrowth','—')}</td>"
            f"<td>{rd.get('CFOQuality','—')}</td><td>{rd.get('PE','—')}</td>"
            f"<td>{rd.get('EarnYield','—')}</td><td>{rd.get('Price','')}</td>"
            f"<td style='color:#6ee7b7'>{rd.get('EntryPrice','—')}</td>"
            f"<td>{rd.get('QualityPass','—')}</td><td>{rd.get('FCFFairValue','—')}</td>"
            f"<td>{r.MktCap}</td>"
            f"<td><div style='display:flex;align-items:center;gap:5px'>"
            f"<div style='width:55px;height:4px;background:#1e293b;border-radius:2px;overflow:hidden'>"
            f"<div style='width:{r.Score}%;height:100%;background:{sc}'></div></div>"
            f"<span style='color:{sc};font-weight:bold'>{r.Score}</span></div></td>"
            f"<td style='color:{vc};font-weight:bold'>{r.Verdict}</td><td>{tgs}</td></tr>"
        )
    curr = CURRENCY_SYMBOL
    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>
<title>Nalanda Screener INR - {date.today()}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#020817;color:#e2e8f0;font-family:'Courier New',Consolas,monospace;font-size:13px;height:100vh;display:flex;flex-direction:column}}
h1{{font-family:Georgia,serif;font-size:1.6rem;color:#fff;line-height:1.15}}
.hdr{{border-bottom:1px solid #1e293b;padding:14px 20px;flex-shrink:0}}
.meta{{color:#475569;font-size:11px;margin-top:3px}}
.note{{color:#334155;font-size:10px;margin-top:3px;font-style:italic}}
.stats{{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}}
.stat{{background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:7px 13px}}
.sn{{font-size:1.25rem;font-weight:bold}}.sl{{color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:.1em}}
.ctrl{{display:flex;gap:8px;align-items:center;padding:9px 20px;border-bottom:1px solid #1e293b;background:#030e1a;flex-shrink:0;flex-wrap:wrap}}
.ctrl input{{background:#0f172a;border:1px solid #1e293b;color:#e2e8f0;padding:5px 9px;border-radius:6px;font-family:inherit;font-size:12px;outline:none;width:170px}}
.btn{{background:#0f172a;border:1px solid #1e293b;color:#64748b;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:11px;font-family:inherit}}
.btn:hover,.btn.on{{background:#0d2d1f;border-color:#065f46;color:#34d399}}
.main{{display:flex;flex:1;overflow:hidden;min-height:0}}
.tw{{flex:1;overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#080f1e;color:#475569;text-transform:uppercase;letter-spacing:.07em;padding:8px 10px;text-align:left;border-bottom:1px solid #1e293b;position:sticky;top:0;z-index:10;cursor:pointer;white-space:nowrap;font-size:10px;user-select:none}}
th:hover{{color:#94a3b8}}
td{{padding:7px 10px;border-bottom:1px solid #0a1220;cursor:pointer;vertical-align:middle;white-space:nowrap}}
.dp{{width:0;overflow:hidden;border-left:1px solid #1e293b;flex-shrink:0;background:#030e1a;transition:width .18s}}
.dp.open{{width:300px;overflow-y:auto;padding:14px}}
.mg{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0}}
.mc{{background:#0f172a;border-radius:6px;padding:7px 9px}}
.ml{{color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:.07em}}
.mv{{font-size:13px;font-weight:bold;margin-top:2px}}
.mv-sm{{font-size:11px;font-weight:bold;margin-top:2px;color:#64748b}}
.tag{{display:inline-block;background:#1e3a2f;color:#6ee7b7;border:1px solid #065f46;border-radius:999px;padding:2px 7px;font-size:10px;margin:2px}}
.tag-warn{{background:#3a1e1e;color:#fca5a5;border-color:#7f1d1d}}
.divider{{border-top:1px solid #1e293b;margin:10px 0;padding-top:8px}}
.info-label{{color:#334155;font-size:10px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}}
::-webkit-scrollbar{{width:4px;height:4px}}::-webkit-scrollbar-track{{background:#0a1628}}::-webkit-scrollbar-thumb{{background:#1e293b;border-radius:2px}}
</style></head><body>
<div class='hdr'>
  <div style='font-size:9px;color:#334155;letter-spacing:.35em;text-transform:uppercase;margin-bottom:3px'>India · NSE · INR</div>
  <h1>Nalanda Screener <span style='color:#34d399'>INR · {universe_label}</span></h1>
  <div class='meta'>Run: {run_time} · Scanned {n_scanned} stocks · All figures in {curr} (INR) · yfinance · Not investment advice</div>
  <div class='note'>ROCE = EBIT / (Total Assets − Current Liabilities) · Valuation: P/E vs 4yr median + earnings yield vs 10yr bond · Growth: ~3-4yr CAGR · NSE</div>
  <div class='stats'>
    <div class='stat'><div class='sn' style='color:#34d399'>{strong}</div><div class='sl'>Strong Pass</div></div>
    <div class='stat'><div class='sn' style='color:#a3e635'>{passed}</div><div class='sl'>Pass</div></div>
    <div class='stat'><div class='sn' style='color:#fbbf24'>{watch}</div><div class='sl'>Watch</div></div>
    <div class='stat'><div class='sn' style='color:#94a3b8'>{len(df)}</div><div class='sl'>Total</div></div>
  </div>
</div>
<div class='ctrl'>
  <input id='srch' placeholder='Search ticker or name...' oninput='ft()'>
  <button class='btn on' onclick='sv("ALL",this)'>All</button>
  <button class='btn' onclick='sv("STRONG PASS",this)'>Strong Pass</button>
  <button class='btn' onclick='sv("PASS",this)'>Pass</button>
  <button class='btn' onclick='sv("WATCH",this)'>Watch</button>
  <span id='cnt' style='color:#475569;font-size:11px;margin-left:auto'></span>
</div>
<div class='main'>
  <div class='tw'><table id='tbl'>
    <thead><tr>
      <th onclick='st(0)'>Ticker</th><th onclick='st(1)'>Name</th><th onclick='st(2)'>Sector</th>
      <th onclick='st(3)'>ROCE TTM</th>
      <th onclick='st(4)'>ROCE Avg</th>
      <th onclick='st(5)'>D/E</th>
      <th onclick='st(6)' title='Revenue CAGR (~3-4yr)'>Rev Gr</th>
      <th onclick='st(7)' title='Earnings CAGR (~3-4yr)'>Earn Gr</th>
      <th onclick='st(8)' title='CFO / Net Income — cash earnings quality'>CFO/NI</th>
      <th onclick='st(9)' title='P/E vs 4yr median — Prasad cheapness test'>P/E</th>
      <th onclick='st(10)' title='Earnings Yield vs 10yr Bond'>Earn Yld</th>
      <th onclick='st(11)'>Price</th>
      <th onclick='st(12)' title='Buy at or below: satisfies both cheapness metrics'>Entry</th>
      <th onclick='st(13)' title='Phase 1 quality filter'>Quality</th>
      <th onclick='st(14)' title='FCF Fair Value: Gordon Growth if FCF&lt;NI, else EV/FCF median reversion'>FCF FV</th>
      <th onclick='st(15)'>Mkt Cap</th>
      <th onclick='st(16)'>Score</th><th onclick='st(17)'>Verdict</th><th>Tags</th>
    </tr></thead>
    <tbody id='tb'>{rows}</tbody>
  </table></div>
  <div class='dp' id='dp'></div>
</div>
<script>
const D={json.dumps(jsd)};
let av='ALL',sc=16,sa=false;
function nv(s){{if(!s||s==='--'||s==='—')return -9999;return parseFloat(String(s).replace(/[^0-9.\\-]/g,''))||0;}}
function ft(){{
  const q=document.getElementById('srch').value.toLowerCase();
  const rows=document.querySelectorAll('#tb tr');let n=0;
  rows.forEach(r=>{{
    const ok=(av==='ALL'||r.cells[17].textContent.trim()===av)&&
      (!q||r.cells[0].textContent.toLowerCase().includes(q)||r.cells[1].textContent.toLowerCase().includes(q));
    r.style.display=ok?'':'none';if(ok)n++;
  }});document.getElementById('cnt').textContent=n+' stocks';
}}
function sv(v,b){{av=v;document.querySelectorAll('.btn').forEach(x=>x.classList.remove('on'));b.classList.add('on');ft();}}
function st(c){{
  if(sc===c)sa=!sa;else{{sc=c;sa=false;}}
  const tb=document.getElementById('tb');
  const rows=Array.from(tb.querySelectorAll('tr'));
  rows.sort((a,b)=>{{
    const av2=a.cells[c]?.textContent.trim()||'',bv2=b.cells[c]?.textContent.trim()||'';
    const an=nv(av2),bn=nv(bv2);
    const cmp=(an!==-9999&&bn!==-9999)?an-bn:av2.localeCompare(bv2);
    return sa?cmp:-cmp;
  }});rows.forEach(r=>tb.appendChild(r));ft();
}}
function sd(t){{
  const d=D[t];if(!d)return;
  const p=document.getElementById('dp');p.classList.add('open');
  const sc2=parseInt(d.Score)>=78?'#34d399':parseInt(d.Score)>=62?'#fbbf24':'#fb923c';
  const vc={{'STRONG PASS':'#34d399','PASS':'#a3e635','WATCH':'#fbbf24'}}[d.Verdict]||'#94a3b8';
  const tags=(d.Tags||'').split(', ').filter(Boolean).map(t=>{{
    const warn=t.includes('Volatile')||t.includes('Avoid')||t.includes('Conglomerate')||t.includes('Negative');
    return `<span class='tag ${{warn?"tag-warn":""}}' >${{t}}</span>`;
  }}).join('');

  const h=d.ROCEHistory||'';
  let roceBars='';
  if(h){{
    const pairs=h.split('  |  ').map(s=>s.trim()).filter(Boolean);
    const vals=pairs.map(p=>{{const m=p.match(/([\\d.]+)%/);return m?parseFloat(m[1]):0;}});
    const mx=Math.max(...vals,1);
    roceBars=`<div class='divider'><div class='info-label'>ROCE History (EBIT / Cap. Employed)</div>`+
      pairs.map((pr,i)=>{{
        const pct=Math.round(vals[i]/mx*100);
        const col=vals[i]>=25?'#34d399':vals[i]>=15?'#fbbf24':'#fb923c';
        const yr=pr.split(':')[0].trim();
        return `<div style='display:flex;align-items:center;gap:5px;margin:3px 0'>
          <span style='color:#475569;font-size:10px;width:36px'>${{yr}}</span>
          <div style='flex:1;height:10px;background:#1e293b;border-radius:2px;overflow:hidden'>
            <div style='width:${{pct}}%;height:100%;background:${{col}}'></div></div>
          <span style='color:${{col}};font-size:10px;width:36px;text-align:right'>${{vals[i].toFixed(0)}}%</span>
        </div>`;
      }}).join('')+`</div>`;
  }}

  p.innerHTML=`
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px'>
    <div>
      <div style='font-size:19px;font-weight:bold;color:#fff'>${{d.Ticker}}</div>
      <div style='color:#64748b;font-size:11px'>${{d.Name}}</div>
      <div style='color:#334155;font-size:10px'>${{d.Industry}}</div>
    </div>
    <div style='text-align:right'>
      <div style='color:${{vc}};font-weight:bold;font-size:11px'>${{d.Verdict}}</div>
      <div style='font-size:26px;font-weight:bold;color:${{sc2}};line-height:1'>${{d.Score}}</div>
      <div style='color:#475569;font-size:10px'>/100</div>
    </div>
  </div>
  <div style='margin-bottom:10px'>${{tags}}</div>

  <div class='info-label'>Primary Filter: Historical ROCE</div>
  <div class='mg'>
    <div class='mc'><div class='ml'>ROCE TTM</div><div class='mv' style='color:${{sc2}}'>${{d.ROCE}}</div></div>
    <div class='mc'><div class='ml'>ROCE Avg</div><div class='mv'>${{d.ROCEAvg||'—'}}</div></div>
  </div>
  ${{roceBars}}

  <div class='divider'><div class='info-label'>Growth (~3-4yr CAGR)</div></div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Revenue CAGR</div><div class='mv'>${{d.RevGrowth||'—'}}</div></div>
    <div class='mc'><div class='ml'>Earnings CAGR</div><div class='mv'>${{d.EarnGrowth||'—'}}</div></div>
  </div>
  <div class='mg'>
    <div class='mc'><div class='ml'>CFO / Net Income</div><div class='mv'>${{d.CFOQuality||'—'}}</div></div>
  </div>

  <div class='divider'><div class='info-label'>Valuation (Prasad's Cheapness Test)</div></div>
  <div class='mg'>
    <div class='mc'><div class='ml'>P/E TTM</div><div class='mv'>${{d.PE||'—'}}</div></div>
    <div class='mc'><div class='ml'>P/E Median (4yr)</div><div class='mv-sm'>${{d.PEMedian||'—'}}</div></div>
  </div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Earnings Yield</div><div class='mv'>${{d.EarnYield||'—'}}</div></div>
    <div class='mc'><div class='ml'>Bond Yield (10yr)</div><div class='mv-sm'>${{d.BondYield||'—'}}</div></div>
  </div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Entry Price</div><div class='mv' style='color:#6ee7b7'>${{(d.EntryPrice&&d.EntryPrice!=='nan'?d.EntryPrice:'—')}}</div></div>
    <div class='mc'><div class='ml'>Current Price</div><div class='mv-sm'>${{d.Price||'—'}}</div></div>
  </div>
  <div style='color:#475569;font-size:10px;margin-bottom:6px'>Entry: buy at or below so P/E &lt; median AND earnings yield &gt; bond yield</div>

  <div class='divider'><div class='info-label'>Nalanda Entry (3-Phase)</div></div>
  <div style='color:#64748b;font-size:10px;margin-bottom:6px'>Buy when Quality Pass, Price ≤ FCF Fair Value, Implied Growth &lt; Sustainable, and RSI &lt; 35 near 200d SMA.</div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Quality Pass</div><div class='mv' style='color:${{d.QualityPass==="Y"?"#34d399":"#94a3b8"}}'>${{d.QualityPass||'—'}}</div></div>
    <div class='mc'><div class='ml'>Net Debt/EBITDA</div><div class='mv-sm'>${{d.NetDebtEBITDA||'—'}}</div></div>
    <div class='mc'><div class='ml'>FCF/NI</div><div class='mv-sm'>${{d.FCFtoNI||'—'}}</div></div>
  </div>
  <div class='mg'>
    <div class='mc'><div class='ml'>FCF Yield</div><div class='mv-sm'>${{d.FCFYield||'—'}}</div></div>
    <div class='mc'><div class='ml'>Hurdle</div><div class='mv-sm'>${{d.Hurdle||'—'}}</div></div>
  </div>
  <div class='mg'>
    <div class='mc'><div class='ml'>FCF Fair Value</div><div class='mv' style='color:#6ee7b7' title='Gordon Growth: FCF÷(hurdle−g). Use as realistic entry target.'>${{(d.FCFFairValue&&d.FCFFairValue!=='nan'?d.FCFFairValue:'—')}}</div></div>
    <div class='mc'><div class='ml'>FCF Floor</div><div class='mv-sm' title='Zero-growth anchor: FCF÷hurdle. Backstop — never pay more than this if growth stops.'>${{(d.FCFFloor&&d.FCFFloor!=='nan'?d.FCFFloor:'—')}}</div></div>
    <div class='mc'><div class='ml'>Current Price</div><div class='mv-sm'>${{d.Price||'—'}}</div></div>
  </div>
  <div style='color:#475569;font-size:9px;margin-bottom:4px'>FCF Fair Value: if FCF &lt; NI → Gordon Growth FCF÷(hurdle−g); if FCF ≥ NI → EV/FCF multiple reversion at historical median. Method shown in parentheses. FCF Floor = zero-growth backstop.</div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Implied Growth</div><div class='mv-sm'>${{d.ImpliedGrowth||'—'}}</div></div>
    <div class='mc'><div class='ml'>Sustainable Growth</div><div class='mv-sm'>${{d.SustGrowth||'—'}}</div></div>
    <div class='mc'><div class='ml'>Reverse DCF</div><div class='mv-sm' style='color:${{d.ReverseDCF==="Buy"?"#34d399":"#94a3b8"}}'>${{d.ReverseDCF||'—'}}</div></div>
  </div>
  <div style='color:#334155;font-size:9px;margin-top:2px'>Reverse DCF "—": FCF ≥ NI (sustainable growth N/A). For these names, use FCF yield vs hurdle and P/E vs history.</div>
  <div class='mg'>
    <div class='mc'><div class='ml'>RSI(14)</div><div class='mv-sm'>${{d.RSI14||'—'}}</div></div>
    <div class='mc'><div class='ml'>200d SMA</div><div class='mv-sm'>${{d.SMA200||'—'}}</div></div>
    <div class='mc'><div class='ml'>Price vs 200d SMA</div><div class='mv-sm'>${{d.PriceVsSma200||'—'}}</div></div>
    <div class='mc'><div class='ml'>Technical</div><div class='mv-sm' style='color:${{d.TechnicalReady==="Ready"?"#34d399":"#94a3b8"}}'>${{d.TechnicalReady||'—'}}</div></div>
  </div>

  <div class='divider'><div class='info-label'>Balance Sheet</div></div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Debt / Equity</div><div class='mv'>${{d.DE||'—'}}</div></div>
  </div>

  <div class='divider'><div class='info-label'>Margin Info (info only — not scored)</div></div>
  <div style='color:#334155;font-size:10px;margin-bottom:6px'>FCF &amp; Gross Margin are downstream effects of ROCE.</div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Op Margin</div><div class='mv-sm'>${{d.OpMargin||'—'}}</div></div>
    <div class='mc'><div class='ml'>FCF Margin</div><div class='mv-sm'>${{d.FCFMargin||'—'}}</div></div>
    <div class='mc'><div class='ml'>Gross Margin</div><div class='mv-sm'>${{d.GrossMargin||'—'}}</div></div>
    <div class='mc'><div class='ml'>Net Margin</div><div class='mv-sm'>${{d.NetMargin||'—'}}</div></div>
  </div>

  <div class='divider'>
    <div style='color:#334155;font-size:10px'>Sector: ${{d.Sector}} · ${{d.MktCap}}</div>
    <div style='color:#334155;font-size:10px'>Price: ${{d.Price}}</div>
    <div style='font-size:10px;margin-top:4px;color:${{(d.DataQuality||"").includes("⚠")?"#fbbf24":"#334155"}}'>${{d.DataQuality||""}}</div>
  </div>
  <button onclick="document.getElementById('dp').classList.remove('open')"
    style='margin-top:12px;width:100%;background:#0f172a;border:1px solid #1e293b;
           color:#64748b;padding:6px;border-radius:6px;cursor:pointer;font-family:inherit'>
    Close ✕</button>`;
}}
ft();
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def load_scanned():
    if os.path.exists(SCANNED_FILE):
        try:
            with open(SCANNED_FILE) as f:
                data = json.load(f)
            return set(data.get("tickers", []))
        except Exception:
            pass
    return set()

def save_scanned(scanned_set):
    with open(SCANNED_FILE, "w") as f:
        json.dump({"tickers": list(scanned_set)}, f)

def load_master():
    if os.path.exists(MASTER_CSV):
        return pd.read_csv(MASTER_CSV)
    return pd.DataFrame()

def save_master(new_df, master_df):
    if master_df.empty:
        combined = new_df
    else:
        old = master_df[~master_df.Ticker.isin(new_df.Ticker)]
        combined = pd.concat([old, new_df], ignore_index=True)
    combined = combined.sort_values("Score", ascending=False).reset_index(drop=True)
    combined.to_csv(MASTER_CSV, index=False)
    return combined


def run_entry(ticker):
    ticker = _normalize_ticker(ticker) or "RELIANCE.NS"
    km, rat, prof, hist_roce = get_yf_data(ticker)
    if "__err" in km:
        print(f"{Fore.RED}Error: {km['__err']}")
        return
    score, verdict, tags, det = nalanda_score(km, rat, prof, hist_roce)
    print(f"\n{Fore.CYAN}{'='*60}\n  Nalanda Entry (INR) — {ticker}\n{'='*60}{Style.RESET_ALL}\n")
    print(f"  Score: {score}/100  →  {verdict}\n  Tags: {', '.join(tags)}\n")
    print(f"  Phase 1 Quality: {det.get('Quality Pass')}  Net Debt/EBITDA: {det.get('Net Debt/EBITDA')}  FCF/NI: {det.get('FCF/NI')}")
    print(f"  Phase 2 FCF Fair Value: {det.get('FCF Fair Value')}  Floor: {det.get('FCF Floor')}")
    print(f"  Phase 3 Reverse DCF: {det.get('Reverse DCF')}  Implied: {det.get('Implied Growth')}  Sustainable: {det.get('Sustainable Growth')}")
    print(f"  Phase 4 Technical: {det.get('Technical')}  RSI: {det.get('RSI(14)')}  Price vs 200d: {det.get('Price vs 200d SMA')}")
    print(f"\n  Current price: {CURRENCY_SYMBOL}{prof.get('price', 0):.2f}\n")


def main():
    if DEBUG:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        run_debug(args[0] if args else "RELIANCE.NS")
        return
    entry_idx = None
    for i, a in enumerate(sys.argv):
        if a == "--entry":
            entry_idx = i
            break
    if entry_idx is not None:
        ticker = (sys.argv[entry_idx + 1] if entry_idx + 1 < len(sys.argv) else "").strip() or "RELIANCE"
        run_entry(ticker)
        sys.exit(0)
    today    = date.today().isoformat()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    master      = load_master()
    scanned_all = load_scanned()
    remaining = [t for t in INDIAN_TICKERS if t not in scanned_all]
    if not remaining:
        print(f"{Fore.YELLOW}  Full cycle complete — resetting.")
        scanned_all = set()
        save_scanned(scanned_all)
        remaining = INDIAN_TICKERS
    to_scan = remaining[:MAX_STOCKS] if MAX_STOCKS else remaining
    print(f"\n{'='*60}")
    print(f"  NALANDA SCREENER INR (NSE) — {today}")
    print(f"  All figures in {CURRENCY_SYMBOL} (INR). Universe: {UNIVERSE_LABEL} (use --local-tickers for a text file).")
    print(f"  Scanned this cycle: {len(scanned_all)}/{len(INDIAN_TICKERS)}  |  This run: {len(to_scan)}")
    print(f"{'='*60}\n")
    results = []
    for i, ticker in enumerate(to_scan, 1):
        print(f"  [{i:3d}/{len(to_scan)}] {ticker:<18}", end=" ", flush=True)
        km, rat, prof, hist_roce = get_yf_data(ticker)
        scanned_all.add(ticker)
        save_scanned(scanned_all)
        if "__err" in km:
            print(f"{Fore.RED}skip: {km['__err'][:50]}")
            continue
        score, verdict, tags, det = nalanda_score(km, rat, prof, hist_roce)
        if verdict == "FAIL":
            print(f"{Fore.RED}FAIL ({score})")
            time.sleep(PAUSE)
            continue
        cap  = prof.get("mktCap") or 0
        opm  = safe(rat, "operatingProfitMarginTTM") * 100
        npm  = safe(rat, "netProfitMarginTTM") * 100
        fcfm = safe(rat, "fcfMarginTTM") * 100
        gm   = safe(rat, "grossProfitMarginTTM") * 100
        fy_only  = [(yr, r) for yr, r in (hist_roce or []) if yr != "TTM"]
        roce_avg = ""
        if fy_only:
            avg = sum(r for _, r in fy_only) / len(fy_only) * 100
            roce_avg = f"{avg:.1f}% ({len(fy_only)}yr)"
        price_raw = prof.get("price") or 0
        pe_val    = safe(rat, "priceToEarningsRatioTTM")
        bond_y    = km.get("bondYield10yr") or 0.07
        hist_pe   = km.get("histPeMedian")
        entry_price_str = "—"
        if price_raw and pe_val > 1 and bond_y > 0:
            eps_ttm = price_raw / pe_val
            entry_bond  = eps_ttm / bond_y
            entry_median = (eps_ttm * hist_pe) if hist_pe else None
            entry_val = min(entry_bond, entry_median) if entry_median is not None else entry_bond
            entry_price_str = f"{CURRENCY_SYMBOL}{entry_val:.2f}"
        results.append({
            "Ticker": ticker, "Name": prof.get("companyName", ticker), "Sector": prof.get("sector", "--"),
            "Industry": prof.get("industry", "--"), "ROCE": det["ROCE TTM"], "ROCEAvg": roce_avg,
            "DE": det["D/E"], "RevGrowth": det.get("Revenue CAGR", "—"), "EarnGrowth": det.get("Earnings CAGR", "—"),
            "CFOQuality": det.get("CFO / Net Income", "—"), "FCFMargin": f"{fcfm:.1f}%", "GrossMargin": f"{gm:.1f}%",
            "OpMargin": f"{opm:.1f}%", "NetMargin": f"{npm:.1f}%", "PE": det.get("P/E TTM", "—"),
            "PEMedian": det.get("P/E Median (4yr)", "—"), "EarnYield": det.get("Earnings Yield", "—"),
            "BondYield": det.get("Bond Yield (10yr)", "—"), "Price": f"{CURRENCY_SYMBOL}{price_raw:.2f}",
            "EntryPrice": entry_price_str, "QualityPass": det.get("Quality Pass", "—"),
            "NetDebtEBITDA": det.get("Net Debt/EBITDA", "—"), "FCFtoNI": det.get("FCF/NI", "—"),
            "FCFYield": det.get("FCF Yield", "—"), "Hurdle": det.get("Hurdle", "—"), "FCFFloor": det.get("FCF Floor", "—"),
            "FCFFairValue": det.get("FCF Fair Value", "—"), "ImpliedGrowth": det.get("Implied Growth", "—"),
            "SustGrowth": det.get("Sustainable Growth", "—"), "ReverseDCF": det.get("Reverse DCF", "—"),
            "RSI14": det.get("RSI(14)", "—"), "SMA200": det.get("200d SMA", "—"), "PriceVsSma200": det.get("Price vs 200d SMA", "—"),
            "TechnicalReady": det.get("Technical", "—"), "MktCap": capmkt(cap), "Score": score, "Verdict": verdict,
            "Tags": ", ".join(tags), "ROCEHistory": det.get("ROCE History", ""), "DataQuality": det.get("Data Quality", ""),
        })
        col = Fore.GREEN if score >= 78 else Fore.YELLOW if score >= 62 else Fore.MAGENTA
        print(f"{col}{score:3d}  ->  {verdict}")
        time.sleep(PAUSE)
    if not results and master.empty:
        print(f"\n{Fore.RED}No results.")
        return
    new_df = pd.DataFrame(results) if results else pd.DataFrame()
    df_out = save_master(new_df, master) if not new_df.empty else master.sort_values("Score", ascending=False)
    print(f"\n{'='*60}\n{Fore.GREEN}  This run: {len(results)} passed | Master: {len(df_out)}")
    for v, c in [("STRONG PASS", Fore.GREEN), ("PASS", Fore.LIGHTGREEN_EX), ("WATCH", Fore.YELLOW)]:
        n = len(df_out[df_out.Verdict == v])
        if n: print(f"  {c}{v:<14}{Style.RESET_ALL} {n}")
    print(f"{'='*60}\n")
    show_cols = [c for c in ["Ticker","Name","Sector","ROCE","ROCEAvg","DE","Price","EntryPrice","Score","Verdict"] if c in df_out.columns]
    if HAS_TAB:
        print(tabulate(df_out[show_cols].head(40), headers="keys", tablefmt="rounded_outline", showindex=True))
    else:
        print(df_out[show_cols].head(40).to_string())
    html_path = os.path.join(OUTPUT_DIR, f"nalanda_results_inr_{today}.html")
    write_html(df_out, html_path, run_time, len(to_scan), UNIVERSE_LABEL)
    print(f"\n{Fore.GREEN}  Master CSV  -> {MASTER_CSV}\n  HTML Report -> {html_path}")
    left = len(INDIAN_TICKERS) - len(scanned_all)
    print(f"  {left} stocks remaining in this cycle.\n")

if __name__ == "__main__":
    main()
