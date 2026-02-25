#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         NALANDA-STYLE US STOCK SCREENER  v8                     ║
║         yfinance — no API key required                          ║
║         Framework: Pulak Prasad / Nalanda Capital               ║
╚══════════════════════════════════════════════════════════════════╝

FAITHFUL TO PULAK PRASAD'S ACTUAL PRINCIPLES:
  1. Historical ROCE ≥ 20% (5yr avg) — the single primary filter
     ROCE = EBIT / (Total Assets − Current Liabilities)  [pre-tax, Nalanda def.]
     NOT ROIC (post-tax), NOT ROE (mixes financing with operations)
  2. Debt: near-zero preference — D/E > 0.5 is a penalty, > 1.0 is near-disqualifying
  3. Industry quality: hard penalty for volatile/low-profit/disrupted sectors
  4. Conglomerates: penalised (Prasad explicitly avoids them)
  5. Gross margin and FCF margin shown as INFO ONLY — not independently scored
     (they are downstream effects of ROCE, not separate filters)
  6. Valuation bonus only activates when quality threshold is already met

THREE-PHASE ENTRY (Nalanda-style):
  Phase 1 — Quality: ROCE > 20%, Net Debt/EBITDA < 1, FCF/NI > 80%
  Phase 2 — Buy Limit: price where FCF Yield = 10Y Treasury + 3%
  Phase 3 — Reverse DCF: Implied growth vs Sustainable growth (ROCE × reinvestment rate)
  Phase 4 — Technical: RSI(14) < 35, Price vs 200d SMA
  Buy when Quality Pass, Price below Buy Limit (FCF), Implied < Sustainable, RSI < 35.

SETUP (using uv):
    uv venv .venv
    uv pip install --python .venv/bin/python yfinance pandas tabulate colorama
    source .venv/bin/activate

RUN:
    python nalanda_screener.py

ENTRY CHECKLIST (one ticker, full Phase 1–4):
    python nalanda_screener.py --entry AAPL

DEBUG (shows raw yfinance fields for one ticker):
    python nalanda_screener.py --debug
    python nalanda_screener.py --debug AAPL
"""

import yfinance as yf
import pandas as pd
import time, os, sys, json
from datetime import date, datetime

try:
    from tabulate import tabulate; HAS_TAB = True
except ImportError: HAS_TAB = False

try:
    from colorama import init, Fore, Style; init(autoreset=True)
except ImportError:
    class Fore:
        RED=GREEN=YELLOW=CYAN=LIGHTGREEN_EX=MAGENTA=WHITE=""
    class Style: RESET_ALL=""

# ── CONFIG ────────────────────────────────────────────────────────────────────
PAUSE      = 0.6
MAX_STOCKS = None       # None = scan all S&P 500 in one run
OUTPUT_DIR = "."
MASTER_CSV    = os.path.join(OUTPUT_DIR, "nalanda_master.csv")
SCANNED_FILE  = os.path.join(OUTPUT_DIR, ".scanned_tickers.json")  # tracks ALL scanned (pass+fail)
DEBUG         = "--debug" in sys.argv

# Three-Phase Nalanda Entry (FCF-yield Buy Limit, Reverse DCF, technical)
HURDLE_RISK_PREMIUM         = 0.03   # 10Y Treasury + this = hurdle for FCF yield
REVERSE_DCF_DISCOUNT_RATE    = 0.10
RSI_OVERSOLD                 = 35
QUALITY_ROCE_MIN             = 20    # %
QUALITY_NET_DEBT_EBITDA_MAX  = 1.0
QUALITY_FCF_NI_MIN           = 0.80  # FCF/NI >= 80%

# ── INDUSTRY CLASSIFICATION ───────────────────────────────────────────────────
# Prasad: "steer clear of volatile or low-profit industries"
# "Nalanda loves stable, predictable, boring industries"
# "Avoid conglomerates"
# Penalty is applied to score, not a hard block — allows analyst override

VOLATILE_INDUSTRIES = {
    # Semiconductors — cyclical, capex-heavy, disruption-prone
    "Semiconductors", "Semiconductor Equipment & Materials",
    # Airlines — commodity pricing, high debt, cyclical
    "Airlines",
    # Oil & Gas — commodity prices, macro-driven
    "Oil & Gas Exploration & Production", "Oil & Gas Refining & Marketing",
    "Oil & Gas Equipment & Services", "Oil & Gas Integrated",
    # Metals & Mining — pure commodity
    "Steel", "Aluminum", "Copper", "Gold", "Silver", "Coal",
    "Metals & Mining",
    # Biotech — binary outcomes, unpredictable
    "Biotechnology",
    # Autos — cyclical, capital-intensive, disruption risk
    "Auto Manufacturers", "Auto Parts",
    # Shipping — commodity rates
    "Marine Shipping",
    # Paper/Chemicals — commodity margins
    "Paper & Paper Products", "Commodity Chemicals",
}

CONGLOMERATE_KEYWORDS = [
    "conglomerate", "diversified", "multi-sector", "holding company"
]

# ── S&P 500 TICKERS — fetched live from Wikipedia ────────────────────────────
_SP500_CACHE_FILE = os.path.join(OUTPUT_DIR, ".sp500_cache.json")
_SP500_CACHE_DAYS = 7   # refresh the list at most once a week

def get_sp500_tickers():
    """
    Fetch the current S&P 500 constituent list from Wikipedia.
    Caches the result for up to _SP500_CACHE_DAYS days so we don't
    hit Wikipedia on every run.

    Wikipedia's table is maintained by editors and reflects index changes
    (additions/deletions) within days. Dots in tickers are replaced with
    hyphens to match yfinance format (e.g. BRK.B → BRK-B).

    Falls back to a compact hardcoded list if the fetch fails.
    """
    # Check cache first
    if os.path.exists(_SP500_CACHE_FILE):
        try:
            with open(_SP500_CACHE_FILE) as f:
                cached = json.load(f)
            cached_date = date.fromisoformat(cached["date"])
            if (date.today() - cached_date).days < _SP500_CACHE_DAYS:
                return cached["tickers"]
        except Exception:
            pass

    print(f"  {Fore.CYAN}Fetching live S&P 500 list from Wikipedia...", end=" ", flush=True)
    try:
        import requests as _req
        url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; nalanda-screener/1.0)"}
        resp    = _req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        import io
        tables  = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        tickers = (
            tables[0]["Symbol"]
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        # Deduplicate while preserving order
        seen_t = set()
        tickers = [t for t in tickers if not (t in seen_t or seen_t.add(t))]

        # Save cache
        with open(_SP500_CACHE_FILE, "w") as f:
            json.dump({"date": date.today().isoformat(), "tickers": tickers}, f)

        print(f"{Fore.GREEN}OK ({len(tickers)} stocks)")
        return tickers

    except Exception as e:
        print(f"{Fore.YELLOW}failed ({e}) — using built-in fallback list")
        return _SP500_FALLBACK


# Compact fallback used only when Wikipedia is unreachable
_SP500_FALLBACK = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH",
    "ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG",
    "AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BAX","BDX","BRK-B","BBY","BIO","BIIB","BLK","BX","BA","BMY","AVGO","BR",
    "BRO","BLDR","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR",
    "CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CRL","SCHW","CHTR",
    "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME",
    "CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO",
    "CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHI",
    "DHR","DRI","DVA","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG",
    "DLTR","D","DPZ","DOV","DOW","DTE","DUK","DD","EMN","ETN","EBAY","ECL",
    "EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX",
    "EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR",
    "XOM","FFIV","FDS","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC",
    "F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
    "GEN","GNRC","GD","GIS","GPC","GILD","GL","GPN","GS","HAL","HIG","HAS",
    "HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST",
    "HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","ILMN","INCY",
    "IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV",
    "IRM","JBHT","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY",
    "KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS",
    "LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO",
    "MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH",
    "TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP",
    "NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH",
    "NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL",
    "OTIS","OGN","PCAR","PKG","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR",
    "PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG",
    "PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL",
    "RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP",
    "ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS",
    "SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SYF",
    "SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX",
    "TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC",
    "TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS",
    "VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VMC","WRB",
    "GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST",
    "WDC","WY","WHR","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZION",
    "ZTS","FICO","VEEV",
]

SP500_DEDUP = get_sp500_tickers()

# ── YFINANCE DATA FETCH ───────────────────────────────────────────────────────
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
    """
    Calculate ROCE = EBIT / Capital Employed for available fiscal years (up to 10) + TTM.

    Nalanda definition (Prasad's own words):
        ROCE             = EBIT / Capital Employed   [pre-tax, NOT NOPAT]
        Capital Employed = Total Assets − Current Liabilities
        EBIT             = Operating Income

    Why this definition:
        - Pre-tax: removes tax planning from the signal
        - Capital Employed vs Invested Capital: includes all operating liabilities
          (trade payables, accrued expenses), giving a truer picture of the
          capital the business actually needs to run its operations
        - Specifically rejects ROE because it mixes financing strategy with
          operating performance (Prasad, Chapter 3)

    Returns list of (label, roce_decimal) sorted TTM first, then FY newest→oldest.
    """
    try:
        inc = t.financials
        bs  = t.balance_sheet

        if inc is None or bs is None or inc.empty or bs.empty:
            return []

        # Use up to 10 years when yfinance provides them (often only ~4)
        shared_cols = [c for c in inc.columns if c in bs.columns][:10]
        results = []

        for col in shared_cols:
            try:
                ebit = _fs_val(inc, col, [
                    "Operating Income",
                    "EBIT",
                    "Operating Income Or Loss",
                    "Total Operating Income As Reported",
                ])
                if ebit is None:
                    continue

                total_assets = _fs_val(bs, col, ["Total Assets"])
                curr_liab    = _fs_val(bs, col, [
                    "Current Liabilities",
                    "Total Current Liabilities",
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

        # TTM ROCE: use TTM operating margin × TTM revenue + most recent balance sheet
        op_margin = info.get("operatingMargins", 0) or 0
        revenue   = info.get("totalRevenue",     0) or 0
        ttm_ebit  = op_margin * revenue

        if ttm_ebit and not bs.empty:
            most_recent = bs.columns[0]
            ta = _fs_val(bs, most_recent, ["Total Assets"])
            cl = _fs_val(bs, most_recent, [
                "Current Liabilities",
                "Total Current Liabilities",
                "Current Liabilities Net Minority Interest",
            ])
            if ta and cl and (ta - cl) > 0:
                ttm_roce = max(-1.0, min(5.0, ttm_ebit / (ta - cl)))
                results.insert(0, ("TTM", ttm_roce))

        # Build data-quality metadata
        if shared_cols:
            filing_date    = str(shared_cols[0].date())
            months_old     = (date.today() - shared_cols[0].date()).days // 30
            is_stale       = months_old > 15
        else:
            filing_date    = "unknown"
            months_old     = 99
            is_stale       = True

        return results, filing_date, months_old, is_stale

    except Exception:
        return [], "unknown", 99, True


def _calc_growth(t):
    """
    Compute revenue CAGR and earnings (net income) CAGR from annual financials.
    Returns (rev_cagr, earn_cagr) as decimals, or None if data unavailable.
    yfinance provides ~4 fiscal years; oldest→newest gives the growth direction.
    """
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
        ni_row   = get_row([
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests",
        ])

        def cagr(series):
            # series is indexed by Timestamp, newest first
            vals = [float(v) for v in series.sort_index().values]  # oldest → newest
            start, end = vals[0], vals[-1]
            n = len(vals) - 1
            if start <= 0 or end <= 0 or n <= 0:
                return None
            return (end / start) ** (1.0 / n) - 1

        return cagr(rev_row), cagr(ni_row)

    except Exception:
        return None, None


def _calc_cfo_quality(t):
    """
    CFO / Net Income ratio over available years (3-4yr).
    Ratio > 1.0 means the company is converting more than 100% of reported
    profit into cash — a strong quality signal. < 0.8 suggests accruals risk.
    Returns decimal ratio or None.
    """
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
            "Operating Cash Flow",
            "Cash From Operations",
            "Total Cash From Operating Activities",
            "Net Cash Provided By Operating Activities",
        ])
        ni_row = get_row(inc, [
            "Net Income",
            "Net Income Common Stockholders",
        ])

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


# Cache bond yield for the full run — one ^TNX fetch per session
_BOND_YIELD_CACHE: dict = {}


def _fetch_bond_yield():
    """
    Fetch US 10-year Treasury yield from ^TNX (quoted in %, e.g. 4.35).
    Cached after first call so it is only fetched once per run.
    Falls back to 4.5% if unavailable.
    """
    if "val" in _BOND_YIELD_CACHE:
        return _BOND_YIELD_CACHE["val"]
    try:
        tnx  = yf.Ticker("^TNX")
        info = tnx.fast_info
        raw  = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if raw is None:
            info2 = tnx.info
            raw   = info2.get("regularMarketPrice") or info2.get("previousClose") or 0
        val = float(raw) / 100 if raw else 0.045
    except Exception:
        val = 0.045
    _BOND_YIELD_CACHE["val"] = val
    return val


def _calc_hist_pe_median(t, info):
    """
    Compute median historical P/E using annual EPS + year-end prices.
    Steps:
      1. Get historical net income from t.financials (up to 4yr)
      2. Derive annual EPS = NI / shares outstanding
      3. Get Dec year-end closing price from t.history(period="10y")
      4. Compute P/E per year; return median
    Returns float or None if insufficient data.
    """
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

        shares = (info.get("sharesOutstanding") or
                  info.get("impliedSharesOutstanding") or 0)
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


def _calc_technical(t, price):
    """
    RSI(14) and 200-day SMA from daily history. Used for Phase 4 entry checklist.
    Returns dict: rsi14, sma200, priceVsSma200 ("below" | "above" | "—").
    """
    out = {"rsi14": None, "sma200": None, "priceVsSma200": "—"}
    try:
        hist = t.history(period="2y", interval="1d")
        if hist is None or hist.empty or len(hist) < 15:
            return out
        close = hist["Close"].astype(float)
        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        if not rsi.empty and not pd.isna(rsi.iloc[-1]):
            out["rsi14"] = round(float(rsi.iloc[-1]), 1)
        # 200-day SMA
        if len(close) >= 200:
            sma200 = float(close.rolling(200).mean().iloc[-1])
            out["sma200"] = round(sma200, 2)
            if price and sma200:
                out["priceVsSma200"] = "below" if price < sma200 else "above"
    except Exception:
        pass
    return out


def get_yf_data(ticker):
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        if not info:
            return {"__err": "empty info"}, {}, {}, []

        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        if not price:
            return {"__err": "no price"}, {}, {}, []

        hist_roce, filing_date, data_age_months, is_stale = _calc_historical_roce(t, info)

        # Primary ROCE for scoring: prefer TTM from calculated history
        if hist_roce:
            roce_ttm = hist_roce[0][1]
        else:
            op_margin = info.get("operatingMargins", 0) or 0
            roce_ttm  = max(-1.0, min(5.0, op_margin * 1.5))

        de_raw = info.get("debtToEquity")
        if de_raw is None or de_raw <= 0:
            # None  = equity data missing from yfinance
            # ≤ 0   = negative equity (e.g. buyback-heavy companies like MCD, SBUX)
            #         D/E is mathematically meaningless but signals high leverage
            de = 9.99
        else:
            de = de_raw / 100   # yfinance gives D/E as percentage

        fcf = info.get("freeCashflow",    0) or 0
        rev = info.get("totalRevenue",    0) or 0
        ev  = info.get("enterpriseValue", 0) or 0

        # ── Three-Phase Entry: net debt, EBITDA, NI, shares ────────────────────
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

        # ── New v8 metrics ────────────────────────────────────────────────────
        rev_cagr, earn_cagr = _calc_growth(t)
        cfo_quality         = _calc_cfo_quality(t)
        hist_pe_median      = _calc_hist_pe_median(t, info)
        bond_yield          = _fetch_bond_yield()

        km = {
            "returnOnCapitalEmployedTTM": roce_ttm,
            "evToFreeCashFlowTTM":        (ev / fcf) if fcf > 0 else 0,
            "freeCashFlowTTM":            fcf,
            "revenueTTM":                 rev,
            # Three-Phase Entry
            "netDebt":                    net_debt,
            "ebitda":                     ebitda,
            "netIncomeTTM":               ni_ttm,
            "fcfYield":                   fcf_yield,
            "sharesOutstanding":          shares,
            "enterpriseValue":            ev,
            "netDebtToEbitda":            net_debt_to_ebitda,
            "fcfToNiRatio":               fcf_to_ni,
            # Growth (new v8)
            "revenueCagr":                rev_cagr,    # decimal or None
            "earningsCagr":               earn_cagr,   # decimal or None
            "cfoQualityRatio":            cfo_quality, # decimal or None
            # Valuation (new v8)
            "histPeMedian":               hist_pe_median,  # float or None
            "bondYield10yr":              bond_yield,  # decimal (e.g. 0.043)
            # Data quality
            "filingDate":                 filing_date,
            "dataAgeMonths":              data_age_months,
            "isStale":                    is_stale,
        }

        rat = {
            "debtToEquityRatioTTM":     de,
            # INFO-ONLY — displayed but not independently scored
            "fcfMarginTTM":             (fcf / rev) if rev > 0 else 0,
            "grossProfitMarginTTM":     info.get("grossMargins",      0) or 0,
            "operatingProfitMarginTTM": info.get("operatingMargins",  0) or 0,
            "netProfitMarginTTM":       info.get("profitMargins",     0) or 0,
            "priceToEarningsRatioTTM":  info.get("trailingPE",        0) or 0,
        }

        prof = {
            "symbol":      ticker,
            "companyName": info.get("longName") or info.get("shortName", ticker),
            "sector":      info.get("sector",   "--"),
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


# ── DEBUG MODE ────────────────────────────────────────────────────────────────
def run_debug(ticker="MSFT"):
    print(f"\n{Fore.CYAN}DEBUG MODE — {ticker}\n")
    t    = yf.Ticker(ticker)
    info = t.info

    print(f"{'='*60}")
    print(f"  yfinance Ticker.info  ({ticker})")
    print(f"{'='*60}")
    for k, v in sorted(info.items()):
        if v not in (None, "", 0, 0.0, [], {}):
            print(f"  {k:<45} {v}")

    km, rat, prof, hist_roce = get_yf_data(ticker)
    if "__err" in km:
        print(f"\n  {Fore.RED}Error: {km['__err']}")
    else:
        print(f"\n{Fore.CYAN}  --- Nalanda metrics (v8) ---")
        filing     = km.get("filingDate", "unknown")
        age        = km.get("dataAgeMonths", 99)
        stale      = km.get("isStale", False)
        stale_warn = f"  {Fore.YELLOW}⚠ STALE DATA" if stale else f"  {Fore.GREEN}✓ Fresh"

        print(f"  ROCE = EBIT / (Total Assets - Current Liabilities)  [pre-tax]")
        print(f"  Latest filing: {filing}  ({age}mo old){stale_warn}")
        for lbl, r in hist_roce:
            src = "(TTM est.)" if lbl == "TTM" else "(annual)"
            print(f"    {lbl:<6} {r*100:6.1f}%  {src}")
        if not hist_roce:
            print(f"    fallback  {km.get('returnOnCapitalEmployedTTM',0)*100:.1f}%")

        # Growth
        rev_cagr  = km.get("revenueCagr")
        earn_cagr = km.get("earningsCagr")
        cfo_ratio = km.get("cfoQualityRatio")
        print(f"\n  Revenue CAGR     {f'{rev_cagr*100:.1f}%' if rev_cagr  is not None else '—'}")
        print(f"  Earnings CAGR    {f'{earn_cagr*100:.1f}%' if earn_cagr is not None else '—'}")
        print(f"  CFO / Net Income {f'{cfo_ratio:.2f}x'     if cfo_ratio is not None else '—'}")

        # Valuation
        pe         = rat.get("priceToEarningsRatioTTM", 0)
        hist_pe    = km.get("histPeMedian")
        bond_yield = km.get("bondYield10yr", 0.045)
        earn_yield = (1 / pe * 100) if pe > 1 else 0
        print(f"\n  P/E TTM          {pe:.1f}x")
        print(f"  P/E Median (4yr) {f'{hist_pe:.1f}x' if hist_pe else '—'}")
        print(f"  Earnings Yield   {earn_yield:.1f}%")
        print(f"  Bond Yield (10y) {bond_yield*100:.1f}%")
        print(f"  EV/FCF           {km.get('evToFreeCashFlowTTM', 0):.1f}x")

        de   = rat.get("debtToEquityRatioTTM",   0)
        fcfm = rat.get("fcfMarginTTM",            0) * 100
        gm   = rat.get("grossProfitMarginTTM",    0) * 100
        opm  = rat.get("operatingProfitMarginTTM",0) * 100
        cap  = prof.get("mktCap", 0)
        print(f"\n  D/E ratio        {de:.2f}x")
        print(f"  Market Cap       {capmkt(cap)}")
        print(f"  FCF Margin       {fcfm:.1f}%  [info only]")
        print(f"  Gross Margin     {gm:.1f}%  [info only]")
        print(f"  Op Margin        {opm:.1f}%")
        print(f"  Industry         {prof.get('industry','--')}")

        score, verdict, tags, det = nalanda_score(km, rat, prof, hist_roce)
        print(f"\n  Score: {score}/100  →  {verdict}")
        if tags: print(f"  Tags : {', '.join(tags)}")
        print(f"\n  Score breakdown:")
        for k, v in det.items():
            print(f"    {k:<25} {v}")

    print(f"\n{Fore.GREEN}Debug complete.")
    sys.exit(0)


# ── NALANDA SCORING ENGINE ────────────────────────────────────────────────────
def safe(d, *keys, default=0):
    for k in keys:
        v = d.get(k)
        if v is not None and v != 0 and v != "":
            try: return float(v)
            except: pass
    return default


def nalanda_score(km, rat, prof, hist_roce=None):
    """
    Nalanda scoring engine v8 — faithful to Prasad's published principles.

    SCORE STRUCTURE (100 pts max):
    ┌─────────────────────────────────────────────────────────┐
    │ ROCE TTM level               25 pts                     │
    │ ROCE historical consistency  30 pts                     │
    │ Balance sheet / debt         20 pts                     │
    │ Revenue + earnings growth    10 pts  ← new v8           │
    │ CFO quality (cash earnings)   5 pts  ← new v8           │
    │ Valuation (cheapness test)   10 pts  ← new v8           │
    │ Industry quality penalty   −0/−20 pts                   │
    │ Hard gate: Market Cap ≥ $1B                             │
    └─────────────────────────────────────────────────────────┘
    """
    score, tags, details = 0, [], {}
    industry = prof.get("industry", "") or ""
    sector   = prof.get("sector",   "") or ""

    # ══ HARD GATE: Market Cap ≥ $1B ════════════════════════════════════════════
    mkt_cap = prof.get("mktCap", 0) or 0
    details["Market Cap"] = capmkt(mkt_cap)
    if mkt_cap < 1_000_000_000:
        details["Market Cap"] += " — below $1B threshold"
        return 0, "FAIL", ["Below $1B"], details

    # ══ FILTER 1: HISTORICAL ROCE — 55 pts total ═══════════════════════════════
    # Part A: TTM ROCE level — 25 pts
    roce_ttm = safe(km, "returnOnCapitalEmployedTTM") * 100
    details["ROCE TTM"] = f"{roce_ttm:.1f}%"

    if   roce_ttm >= 40: score += 25; tags.append("Elite ROCE")
    elif roce_ttm >= 30: score += 22; tags.append("High ROCE")
    elif roce_ttm >= 20: score += 18; tags.append("Good ROCE")
    elif roce_ttm >= 12: score += 8
    else:                score += 0

    # Part B: Historical consistency — 30 pts
    # yfinance provides ~4 fiscal years; 10yr is ideal but unavailable for free.
    hist_pts  = 0
    hist_note = "no history"

    if hist_roce and len(hist_roce) >= 2:
        fy_only      = [(yr, r) for yr, r in hist_roce if yr != "TTM"]
        all_vals     = [r for _, r in fy_only] if fy_only else [r for _, r in hist_roce]
        n            = len(all_vals)
        avg_roce     = (sum(all_vals) / n) * 100
        min_roce     = min(all_vals) * 100
        yrs_above_20 = sum(1 for r in all_vals if r * 100 >= 20)

        details["ROCE Avg"]     = f"{avg_roce:.1f}% ({n}yr)"
        details["ROCE Min"]     = f"{min_roce:.1f}%"
        details["ROCE History"] = "  |  ".join(
            f"{yr}: {r*100:.0f}%" for yr, r in hist_roce
        )

        if yrs_above_20 == n and avg_roce >= 25:
            hist_pts = 30; hist_note = f"All {n}yr ≥20%, avg {avg_roce:.0f}%"
            tags.append(f"Consistent ROCE ({n}yr)")
        elif yrs_above_20 >= n - 1 and avg_roce >= 22:
            hist_pts = 24; hist_note = f"{yrs_above_20}/{n}yr ≥20%"
            tags.append(f"Steady ROCE ({n}yr)")
        elif yrs_above_20 >= n // 2 and avg_roce >= 18:
            hist_pts = 16; hist_note = f"{yrs_above_20}/{n}yr ≥20%, avg {avg_roce:.0f}%"
        elif avg_roce >= 15:
            hist_pts = 8;  hist_note = f"avg ROCE {avg_roce:.0f}%"
        elif avg_roce >= 10:
            hist_pts = 3
        else:
            hist_pts = 0

        if min_roce < 0:
            hist_pts = max(0, hist_pts - 12); tags.append("ROCE Negative (some years)")
        elif min_roce < 8:
            hist_pts = max(0, hist_pts - 8);  tags.append("ROCE Volatile")
    else:
        hist_pts  = 0
        hist_note = "TTM only — history unavailable"

    score += hist_pts
    details["ROCE Consistency"] = f"{hist_pts}pts — {hist_note}"

    # ══ FILTER 2: BALANCE SHEET / DEBT — 20 pts ════════════════════════════════
    # Prasad: "Detesting Debt" — near-binary. D/E > 1x is near-disqualifying.
    # Note: negative equity companies (buyback-heavy) use sentinel 9.99 from get_yf_data.
    # We avoid safe() here because safe() treats 0 as missing — 0 D/E is valid (no debt).
    de_raw = rat.get("debtToEquityRatioTTM")
    de = float(de_raw) if de_raw is not None else 99.0
    details["D/E"] = f"{de:.2f}x" if de < 9.0 else "Neg. Equity / N/A"

    if   de < 0:     score += 20; tags.append("Net Cash")
    elif de <= 0.10: score += 20; tags.append("Near Zero Debt")
    elif de <= 0.30: score += 15; tags.append("Low Debt")
    elif de <= 0.50: score += 8;  tags.append("Modest Debt")
    elif de <= 1.00: score += 3
    elif de <= 2.00: score += 0
    else:            score -= 5;  tags.append("High Debt")   # active penalty >2x

    # ══ FILTER 3: REVENUE & EARNINGS GROWTH — 10 pts ═══════════════════════════
    # Prasad expects compounders: sustainable double-digit growth over time.
    # yfinance provides ~4yr of history; we compute CAGR from that.
    rev_cagr  = km.get("revenueCagr")
    earn_cagr = km.get("earningsCagr")

    details["Revenue CAGR"]  = (f"{rev_cagr*100:.1f}% ({_cagr_years(km)}yr)"
                                 if rev_cagr  is not None else "— (unavailable)")
    details["Earnings CAGR"] = (f"{earn_cagr*100:.1f}% ({_cagr_years(km)}yr)"
                                 if earn_cagr is not None else "— (unavailable)")

    if rev_cagr is not None and earn_cagr is not None:
        if   rev_cagr >= 0.10 and earn_cagr >= 0.10:
            score += 10; tags.append("Strong Growth")
        elif rev_cagr >= 0.10 or  earn_cagr >= 0.10:
            score += 5;  tags.append("Partial Growth")
        elif rev_cagr >= 0.05 and earn_cagr >= 0.05:
            score += 3
        elif rev_cagr < 0 or earn_cagr < 0:
            score -= 3;  tags.append("⚠ Declining Growth")
    # Missing data → 0 pts, no penalty

    # ══ FILTER 4: CFO QUALITY — 5 pts ══════════════════════════════════════════
    # CFO / Net Income > 1.0 means cash earnings exceed reported earnings.
    # Low ratio (<0.8) suggests significant accruals — a red flag Prasad cites.
    cfo_ratio = km.get("cfoQualityRatio")

    if cfo_ratio is not None:
        details["CFO / Net Income"] = f"{cfo_ratio:.2f}x"
        if   cfo_ratio >= 1.0: score += 5; tags.append("High Cash Quality")
        elif cfo_ratio >= 0.8: score += 2
        elif cfo_ratio <  0.6: score -= 2; tags.append("⚠ Accruals Risk")
    else:
        details["CFO / Net Income"] = "— (unavailable)"

    # ══ FILTER 5: INDUSTRY QUALITY — penalty (0 to −20 pts) ═══════════════════
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

    # ══ FILTER 6: VALUATION — up to +10 pts ════════════════════════════════════
    # Prasad's two-metric cheapness test (only relevant when ROCE quality is met):
    #
    # Metric A: Current P/E < Historical median P/E
    #   — the stock is cheaper than its own history
    #
    # Metric B: Earnings Yield (1/PE) > 10yr Bond Yield
    #   — the stock earns more than the risk-free rate
    #
    # Scoring:
    #   Both pass  → +10 pts  "Cheap vs History + Bonds"
    #   Yield only → +5  pts  "Cheap vs Bonds"
    #   P/E only   → +3  pts  "Below Historical P/E"
    #   Neither    → +0  pts  (no penalty — Prasad holds quality regardless)
    pe          = safe(rat, "priceToEarningsRatioTTM")
    hist_pe_med = km.get("histPeMedian")
    bond_yield  = km.get("bondYield10yr", 0.045)
    earn_yield  = (1.0 / pe) if pe > 1 else 0
    evcf        = safe(km, "evToFreeCashFlowTTM")

    details["P/E TTM"]           = f"{pe:.1f}x"          if pe > 0          else "—"
    details["P/E Median (4yr)"]  = f"{hist_pe_med:.1f}x" if hist_pe_med     else "—"
    details["Earnings Yield"]    = f"{earn_yield*100:.1f}%" if earn_yield > 0 else "—"
    details["Bond Yield (10yr)"] = f"{bond_yield*100:.1f}%"
    details["EV/FCF"]            = f"{evcf:.1f}x"         if evcf > 0        else "—"

    if roce_ttm >= 20 and pe > 1:
        pe_below_median  = bool(hist_pe_med) and pe < hist_pe_med
        yield_beats_bond = earn_yield > bond_yield
        if   pe_below_median and yield_beats_bond:
            score += 10; tags.append("Cheap vs History + Bonds")
        elif yield_beats_bond:
            score += 5;  tags.append("Cheap vs Bonds")
        elif pe_below_median:
            score += 3;  tags.append("Below Historical P/E")

    # ══ THREE-PHASE ENTRY: Phase 1 Quality Filter ═══════════════════════════════
    net_debt   = km.get("netDebt")
    ebitda     = km.get("ebitda")
    ni_ttm     = km.get("netIncomeTTM")
    fcf        = km.get("freeCashFlowTTM") or 0
    fcf_ni     = km.get("fcfToNiRatio")
    nd_ebitda  = km.get("netDebtToEbitda")
    quality_roce   = roce_ttm >= QUALITY_ROCE_MIN
    quality_debt   = (net_debt is not None and ebitda is not None and ebitda > 0 and (net_debt / ebitda) < QUALITY_NET_DEBT_EBITDA_MAX) or (net_debt is not None and net_debt <= 0) or (ebitda is None or ebitda <= 0)

    # FCF/NI quality: Option B — only fail when BOTH (a) FCF/NI is below threshold
    # AND (b) ROCE is declining, which suggests inflated earnings not accruals.
    # If FCF/NI is low but ROCE is stable/high, treat as "Reinvesting" (capex), not a fail.
    fcf_ni_note = ""
    if ni_ttm is None or ni_ttm <= 0 or fcf_ni is None:
        quality_fcf_ni = None                # N/A — can't judge
    elif fcf_ni < 0:
        quality_fcf_ni = False               # FCF negative: fail unconditionally
        fcf_ni_note = " ⚠ neg FCF"
    elif fcf_ni >= QUALITY_FCF_NI_MIN:
        quality_fcf_ni = True                # clean pass
    else:
        # 0 ≤ FCF/NI < 80%: distinguish "accruals risk" from "heavy capex reinvestment".
        # Real accruals risk requires BOTH declining ROCE AND recent ROCE falling below
        # the quality bar (≥ 20%).  A company like MSFT with ROCE at 30% that dipped
        # from 35% is reinvesting in AI/cloud — not manipulating earnings.
        # Rule: only flag accruals if ROCE has actually decayed into marginal territory
        # (< QUALITY_ROCE_MIN * 1.25 = 25%) while also showing a declining trend.
        roce_declining = False
        if hist_roce and len(hist_roce) >= 3:
            fy_vals = [r for yr, r in hist_roce if yr != "TTM"]
            if len(fy_vals) >= 2:
                # hist_roce is newest-first; fy_vals[0] = most recent FY
                recent_roce = fy_vals[0]
                oldest_roce = fy_vals[-1]
                # Declining AND drifting into marginal ROCE territory.
                # hist_roce values are raw decimals (e.g. 0.27 = 27%),
                # so convert the % threshold to decimal before comparing.
                roce_min_decimal = QUALITY_ROCE_MIN / 100        # 20% → 0.20
                roce_declining = (
                    oldest_roce > recent_roce * 1.10
                    and recent_roce < roce_min_decimal * 1.25    # below 25% in decimal
                )
        if roce_declining:
            quality_fcf_ni = False           # marginal ROCE + low FCF/NI = accruals risk
            fcf_ni_note = " ⚠ accruals?"
        else:
            quality_fcf_ni = True            # strong ROCE + low FCF/NI = reinvesting
            fcf_ni_note = " (reinvesting)"

    quality_pass   = quality_roce and quality_debt and (quality_fcf_ni is not False)
    details["Quality Pass"] = "Y" if quality_pass else "N"
    details["Net Debt/EBITDA"] = f"{nd_ebitda:.2f}x" if nd_ebitda is not None else "—"
    details["FCF/NI"] = (f"{fcf_ni:.2f}x{fcf_ni_note}" if fcf_ni is not None else "—")
    if quality_pass:
        tags.append("Entry Quality Pass")

    # ══ THREE-PHASE ENTRY: Phase 2 FCF-yield Buy Limit ═══════════════════════════
    ev    = km.get("enterpriseValue") or 0
    fcf_yield = km.get("fcfYield")
    hurdle = (km.get("bondYield10yr") or 0.045) + HURDLE_RISK_PREMIUM
    details["FCF Yield"] = f"{fcf_yield*100:.1f}%" if fcf_yield is not None else "—"
    details["Hurdle"] = f"{hurdle*100:.1f}%"
    buy_limit_fcf = None
    if fcf and hurdle > 0 and ev is not None:
        target_ev = fcf / hurdle
        net_d = km.get("netDebt")
        if net_d is None:
            net_d = 0
        implied_equity = target_ev - net_d
        shares = km.get("sharesOutstanding") or 0
        if implied_equity > 0 and shares and shares > 0:
            buy_limit_fcf = implied_equity / float(shares)
    details["Buy Limit (FCF)"] = f"${buy_limit_fcf:.2f}" if buy_limit_fcf is not None else "—"

    # ══ THREE-PHASE ENTRY: Phase 3 Reverse DCF ═══════════════════════════════════
    # When FCF >= NI, reinvestment rate = 0 so sustainable growth = 0; show "—" (N/A)
    # rather than "Not Buy". For high FCF/NI names, practitioners rely on FCF yield vs
    # hurdle and P/E vs historical median instead of Reverse DCF.
    g_implied = None
    if ev and fcf and (ev + fcf) > 0:
        g_implied = (float(ev) * REVERSE_DCF_DISCOUNT_RATE - float(fcf)) / (float(ev) + float(fcf))
        g_implied = max(-0.5, min(0.5, g_implied))
    b_reinv = None
    if ni_ttm and ni_ttm > 0 and fcf is not None:
        b_reinv = 1 - (fcf / float(ni_ttm))
        if b_reinv is not None:
            b_reinv = max(0, min(1, b_reinv))
    roce_decimal = (roce_ttm / 100.0) if roce_ttm is not None else None
    g_sust = (roce_decimal * b_reinv) if (roce_decimal is not None and b_reinv is not None) else None
    rev_dcf_signal = "—"
    if g_implied is not None and g_sust is not None:
        if g_sust <= 0:
            # FCF >= NI (reinvestment rate 0): sustainable-growth comparison not meaningful; high FCF/NI is a quality signal, not a "Not Buy"
            rev_dcf_signal = "—"
        elif g_sust > 0:
            rev_dcf_signal = "Buy" if g_implied < g_sust else "Not Buy"
    details["Implied Growth"] = f"{g_implied*100:.1f}%" if g_implied is not None else "—"
    details["Sustainable Growth"] = f"{g_sust*100:.1f}%" if g_sust is not None else "—"
    details["Reverse DCF"] = rev_dcf_signal

    # ══ THREE-PHASE ENTRY: Phase 4 Technical ═════════════════════════════════════
    rsi14 = prof.get("rsi14")
    sma200 = prof.get("sma200")
    price_vs_sma = prof.get("priceVsSma200") or "—"
    technical_ready = (rsi14 is not None and sma200 is not None and rsi14 < RSI_OVERSOLD and price_vs_sma == "below")
    details["RSI(14)"] = f"{rsi14:.1f}" if rsi14 is not None else "—"
    details["200d SMA"] = f"${sma200:.2f}" if sma200 is not None else "—"
    details["Price vs 200d SMA"] = price_vs_sma
    details["Technical"] = "Ready" if technical_ready else ("Not yet" if (rsi14 is not None or sma200 is not None) else "—")

    # ══ DATA QUALITY ════════════════════════════════════════════════════════════
    if km.get("isStale"):
        tags.append("⚠ Stale Data")
        details["Data Quality"] = (f"⚠ Filing {km.get('filingDate','?')} "
                                   f"({km.get('dataAgeMonths','?')}mo old) — verify manually")
    else:
        details["Data Quality"] = (f"✓ Filing {km.get('filingDate','?')} "
                                   f"({km.get('dataAgeMonths','?')}mo old)")

    score = max(0, min(100, score))

    # ROCE gate: without sustained ≥20% ROCE, a stock cannot be PASS/STRONG PASS
    volatile_industry = industry_penalty >= 15
    roce_ok = (
        roce_ttm >= 20
        and (not hist_roce or hist_pts >= 16)
        and not volatile_industry
    )

    verdict = (
        "STRONG PASS" if score >= 78 and roce_ok else
        "PASS"        if score >= 62 and roce_ok else
        "WATCH"       if score >= 45             else
        "FAIL"
    )

    return score, verdict, tags, details


def _cagr_years(km):
    """Helper to show ~how many years the CAGR covers (best guess from data)."""
    return 3


def capmkt(v):
    if not v: return "—"
    v = float(v)
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


# ── HTML REPORT ───────────────────────────────────────────────────────────────
def write_html(df, path, run_time, n_scanned):
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
            f"<tr style='background:{bg}' "
            f"onmouseover=\"this.style.background='#1e293b'\" "
            f"onmouseout=\"this.style.background='{bg}'\" "
            f"onclick=\"sd('{r.Ticker}')\">"
            f"<td style='font-weight:bold;color:#f1f5f9'>{r.Ticker}</td>"
            f"<td style='color:#94a3b8;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{r.Name}</td>"
            f"<td style='color:#64748b;font-size:11px'>{r.Sector}</td>"
            f"<td style='color:{sc};font-weight:bold'>{rd.get('ROCE','')}</td>"
            f"<td style='color:#94a3b8'>{rd.get('ROCEAvg','')}</td>"
            f"<td>{rd.get('DE','')}</td>"
            f"<td style='color:#64748b'>{rd.get('RevGrowth','—')}</td>"
            f"<td style='color:#64748b'>{rd.get('EarnGrowth','—')}</td>"
            f"<td>{rd.get('CFOQuality','—')}</td>"
            f"<td>{rd.get('PE','—')}</td>"
            f"<td style='color:#64748b'>{rd.get('EarnYield','—')}</td>"
            f"<td style='color:#94a3b8'>{r.Price}</td>"
            f"<td style='color:#6ee7b7;font-size:11px' title='Buy at or below: min(price at bond yield, price at median P/E)'>{rd.get('EntryPrice','—')}</td>"
            f"<td style='color:#94a3b8;font-size:11px' title='Phase 1: ROCE>20%, Net Debt/EBITDA<1, FCF/NI>80%'>{rd.get('QualityPass','—')}</td>"
            f"<td style='color:#6ee7b7;font-size:11px' title='FCF Yield = 10Y Treasury + 3%'>{rd.get('BuyLimitFCF','—')}</td>"
            f"<td style='color:#64748b'>{r.MktCap}</td>"
            f"<td><div style='display:flex;align-items:center;gap:5px'>"
            f"<div style='width:55px;height:4px;background:#1e293b;border-radius:2px;overflow:hidden'>"
            f"<div style='width:{r.Score}%;height:100%;background:{sc}'></div></div>"
            f"<span style='color:{sc};font-weight:bold;font-size:12px'>{r.Score}</span></div></td>"
            f"<td style='color:{vc};font-weight:bold;font-size:11px'>{r.Verdict}</td>"
            f"<td>{tgs}</td></tr>"
        )

    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>
<title>Nalanda Screener v7 - {date.today()}</title>
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
  <div style='font-size:9px;color:#334155;letter-spacing:.35em;text-transform:uppercase;margin-bottom:3px'>Permanent Capital · Nalanda Framework</div>
  <h1>Nalanda Screener <span style='color:#34d399'>v8 · S&P 500</span></h1>
  <div class='meta'>Run: {run_time} · Scanned {n_scanned} stocks · yfinance · Not investment advice</div>
  <div class='note'>ROCE = EBIT / (Total Assets − Current Liabilities) · Valuation: P/E vs 4yr median + earnings yield vs 10yr bond · Growth: ~3-4yr CAGR · Market Cap ≥ $1B</div>
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
      <th onclick='st(14)' title='FCF yield = hurdle'>Buy Limit</th>
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

  // ROCE history bars
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
  <div style='color:#64748b;font-size:10px;margin-bottom:6px'>Buy when Quality Pass, Price below Buy Limit (FCF), Implied Growth &lt; Sustainable, and RSI &lt; 35 near 200d SMA.</div>
  <div class='mg'>
    <div class='mc'><div class='ml'>Quality Pass</div><div class='mv' style='color:${{d.QualityPass==="Y"?"#34d399":"#94a3b8"}}'>${{d.QualityPass||'—'}}</div></div>
    <div class='mc'><div class='ml'>Net Debt/EBITDA</div><div class='mv-sm'>${{d.NetDebtEBITDA||'—'}}</div></div>
    <div class='mc'><div class='ml'>FCF/NI</div><div class='mv-sm'>${{d.FCFtoNI||'—'}}</div></div>
  </div>
  <div class='mg'>
    <div class='mc'><div class='ml'>FCF Yield</div><div class='mv-sm'>${{d.FCFYield||'—'}}</div></div>
    <div class='mc'><div class='ml'>Hurdle</div><div class='mv-sm'>${{d.Hurdle||'—'}}</div></div>
    <div class='mc'><div class='ml'>Buy Limit (FCF)</div><div class='mv' style='color:#6ee7b7'>${{(d.BuyLimitFCF&&d.BuyLimitFCF!=='nan'?d.BuyLimitFCF:'—')}}</div></div>
    <div class='mc'><div class='ml'>Current Price</div><div class='mv-sm'>${{d.Price||'—'}}</div></div>
  </div>
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


# ── MASTER DB ─────────────────────────────────────────────────────────────────
def load_scanned():
    """Load the set of ALL tickers scanned this cycle (pass + fail)."""
    if os.path.exists(SCANNED_FILE):
        try:
            with open(SCANNED_FILE) as f:
                data = json.load(f)
            # Reset if the saved cycle date differs from today's cycle
            if data.get("cycle_started"):
                return set(data.get("tickers", []))
        except Exception:
            pass
    return set()

def save_scanned(scanned_set):
    with open(SCANNED_FILE, "w") as f:
        json.dump({"tickers": list(scanned_set)}, f)

def load_master():
    if os.path.exists(MASTER_CSV): return pd.read_csv(MASTER_CSV)
    return pd.DataFrame()

def save_master(new_df, master_df):
    if master_df.empty: combined = new_df
    else:
        old      = master_df[~master_df.Ticker.isin(new_df.Ticker)]
        combined = pd.concat([old, new_df], ignore_index=True)
    combined = combined.sort_values("Score", ascending=False).reset_index(drop=True)
    combined.to_csv(MASTER_CSV, index=False)
    return combined


# ── ENTRY MODE (single-ticker three-phase checklist) ───────────────────────────
def run_entry(ticker):
    """Print full Three-Phase Entry checklist for one ticker."""
    km, rat, prof, hist_roce = get_yf_data(ticker)
    if "__err" in km:
        print(f"{Fore.RED}Error: {km['__err']}")
        return
    score, verdict, tags, det = nalanda_score(km, rat, prof, hist_roce)
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Nalanda Entry Checklist — {ticker.upper()}")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    print(f"  Score: {score}/100  →  {verdict}")
    print(f"  Tags: {', '.join(tags)}\n")
    print(f"  {Fore.CYAN}Phase 1: Quality Filter{Style.RESET_ALL}")
    print(f"    ROCE > 20%:      {det.get('ROCE TTM','—')}")
    print(f"    Quality Pass:   {det.get('Quality Pass','—')}")
    print(f"    Net Debt/EBITDA: {det.get('Net Debt/EBITDA','—')}")
    print(f"    FCF/NI:         {det.get('FCF/NI','—')}")
    print(f"  {Fore.CYAN}Phase 2: FCF-yield Buy Limit{Style.RESET_ALL}")
    print(f"    FCF Yield:      {det.get('FCF Yield','—')}")
    print(f"    Hurdle:         {det.get('Hurdle','—')}")
    print(f"    Buy Limit (FCF): {det.get('Buy Limit (FCF)','—')}")
    print(f"    Current Price:  ${prof.get('price', 0):.2f}")
    print(f"  {Fore.CYAN}Phase 3: Reverse DCF{Style.RESET_ALL}")
    print(f"    Implied Growth:   {det.get('Implied Growth','—')}")
    print(f"    Sustainable Gr:   {det.get('Sustainable Growth','—')}")
    print(f"    Reverse DCF:      {det.get('Reverse DCF','—')}")
    print(f"  {Fore.CYAN}Phase 4: Technical{Style.RESET_ALL}")
    print(f"    RSI(14):        {det.get('RSI(14)','—')}")
    print(f"    200d SMA:       {det.get('200d SMA','—')}")
    print(f"    Price vs SMA:   {det.get('Price vs 200d SMA','—')}")
    print(f"    Technical:      {det.get('Technical','—')}")
    print(f"\n  Current price: ${prof.get('price', 0):.2f}")
    print(f"  Execute when RSI < 35 near Buy Limit price.\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if DEBUG:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        run_debug(args[0] if args else "MSFT")
        return

    entry_idx = None
    for i, a in enumerate(sys.argv):
        if a == "--entry":
            entry_idx = i
            break
    if entry_idx is not None:
        ticker = (sys.argv[entry_idx + 1] if entry_idx + 1 < len(sys.argv) else "").strip() or "AAPL"
        run_entry(ticker)
        sys.exit(0)

    today    = date.today().isoformat()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    master      = load_master()
    scanned_all = load_scanned()   # ALL tickers touched this cycle (pass + fail)

    remaining = [t for t in SP500_DEDUP if t not in scanned_all]
    if not remaining:
        print(f"{Fore.YELLOW}  Full cycle complete — resetting.")
        scanned_all = set()
        save_scanned(scanned_all)
        remaining = SP500_DEDUP
    to_scan = remaining[:MAX_STOCKS] if MAX_STOCKS else remaining

    passing_in_master = set(master.Ticker.tolist()) if not master.empty else set()

    print(f"\n{'='*60}")
    print(f"  NALANDA SCREENER v8 (yfinance) — {today}")
    print(f"  ROCE = EBIT / (Total Assets - Current Liabilities)")
    print(f"  Scanned this cycle : {len(scanned_all)}/{len(SP500_DEDUP)} stocks")
    print(f"  Scanning now       : {len(to_scan)} stocks")
    print(f"  Master DB (passing): {len(master)} stocks")
    print(f"{'='*60}\n")

    results = []
    for i, ticker in enumerate(to_scan, 1):
        print(f"  [{i:3d}/{len(to_scan)}] {ticker:<7}", end=" ", flush=True)

        km, rat, prof, hist_roce = get_yf_data(ticker)

        # Mark as scanned regardless of outcome
        scanned_all.add(ticker)
        save_scanned(scanned_all)

        if "__err" in km:
            print(f"{Fore.RED}skip: {km['__err'][:60]}")
            continue

        score, verdict, tags, det = nalanda_score(km, rat, prof, hist_roce)

        if verdict == "FAIL":
            print(f"{Fore.RED}FAIL ({score})")
            time.sleep(PAUSE)
            continue

        cap  = prof.get("mktCap") or 0
        opm  = safe(rat, "operatingProfitMarginTTM") * 100
        npm  = safe(rat, "netProfitMarginTTM")        * 100
        fcfm = safe(rat, "fcfMarginTTM")              * 100
        gm   = safe(rat, "grossProfitMarginTTM")      * 100

        # ROCE avg from history (for table display)
        fy_only  = [(yr, r) for yr, r in (hist_roce or []) if yr != "TTM"]
        roce_avg = ""
        if fy_only:
            avg = sum(r for _, r in fy_only) / len(fy_only) * 100
            roce_avg = f"{avg:.1f}% ({len(fy_only)}yr)"

        # Entry price: level at which both cheapness metrics pass (P/E < median AND earn yield > bond)
        # = min( price at which E/P = bond yield,  price at which P/E = median P/E )
        price_raw = prof.get("price") or 0
        pe_val    = safe(rat, "priceToEarningsRatioTTM")
        bond_y    = km.get("bondYield10yr") or 0.045
        hist_pe   = km.get("histPeMedian")
        entry_price_str = "—"
        if price_raw and pe_val > 1 and bond_y > 0:
            eps_ttm = price_raw / pe_val
            entry_bond  = eps_ttm / bond_y
            entry_median = (eps_ttm * hist_pe) if hist_pe else None
            if entry_median is not None:
                entry_val = min(entry_bond, entry_median)
            else:
                entry_val = entry_bond
            entry_price_str = f"${entry_val:.2f}"

        results.append({
            "Ticker":      ticker,
            "Name":        prof.get("companyName", ticker),
            "Sector":      prof.get("sector",      "--"),
            "Industry":    prof.get("industry",    "--"),
            "ROCE":        det["ROCE TTM"],
            "ROCEAvg":     roce_avg,
            "DE":          det["D/E"],
            "RevGrowth":   det.get("Revenue CAGR",  "—"),
            "EarnGrowth":  det.get("Earnings CAGR", "—"),
            "CFOQuality":  det.get("CFO / Net Income", "—"),
            "FCFMargin":   f"{fcfm:.1f}%",
            "GrossMargin": f"{gm:.1f}%",
            "OpMargin":    f"{opm:.1f}%",
            "NetMargin":   f"{npm:.1f}%",
            "PE":          det.get("P/E TTM", "—"),
            "PEMedian":    det.get("P/E Median (4yr)", "—"),
            "EarnYield":   det.get("Earnings Yield", "—"),
            "BondYield":   det.get("Bond Yield (10yr)", "—"),
            "Price":       f"${price_raw:.2f}",
            "EntryPrice":  entry_price_str,
            "QualityPass": det.get("Quality Pass", "—"),
            "NetDebtEBITDA": det.get("Net Debt/EBITDA", "—"),
            "FCFtoNI":     det.get("FCF/NI", "—"),
            "FCFYield":    det.get("FCF Yield", "—"),
            "Hurdle":      det.get("Hurdle", "—"),
            "BuyLimitFCF": det.get("Buy Limit (FCF)", "—"),
            "ImpliedGrowth": det.get("Implied Growth", "—"),
            "SustGrowth":  det.get("Sustainable Growth", "—"),
            "ReverseDCF":  det.get("Reverse DCF", "—"),
            "RSI14":       det.get("RSI(14)", "—"),
            "SMA200":      det.get("200d SMA", "—"),
            "PriceVsSma200": det.get("Price vs 200d SMA", "—"),
            "TechnicalReady": det.get("Technical", "—"),
            "MktCap":      capmkt(cap),
            "Score":       score,
            "Verdict":     verdict,
            "Tags":        ", ".join(tags),
            "ROCEHistory": det.get("ROCE History", ""),
            "DataQuality": det.get("Data Quality", ""),
        })
        col = Fore.GREEN if score >= 78 else Fore.YELLOW if score >= 62 else Fore.MAGENTA
        print(f"{col}{score:3d}  ->  {verdict}")
        time.sleep(PAUSE)

    if not results and master.empty:
        print(f"\n{Fore.RED}No results.")
        return

    new_df = pd.DataFrame(results) if results else pd.DataFrame()
    df_out = save_master(new_df, master) if not new_df.empty else master.sort_values("Score", ascending=False)

    print(f"\n{'='*60}")
    print(f"{Fore.GREEN}  This run: {len(results)} passed | Master DB: {len(df_out)} total")
    for v, c in [("STRONG PASS", Fore.GREEN), ("PASS", Fore.LIGHTGREEN_EX), ("WATCH", Fore.YELLOW)]:
        n = len(df_out[df_out.Verdict == v])
        if n: print(f"  {c}{v:<14}{Style.RESET_ALL} {n}")
    print(f"{'='*60}\n")

    show_cols = [c for c in ["Ticker","Name","Sector","ROCE","ROCEAvg","DE","RevGrowth","EarnGrowth","CFOQuality","Price","EntryPrice","Score","Verdict"] if c in df_out.columns]
    if HAS_TAB: print(tabulate(df_out[show_cols].head(40), headers="keys", tablefmt="rounded_outline", showindex=True))
    else:       print(df_out[show_cols].head(40).to_string())

    html_path = os.path.join(OUTPUT_DIR, f"nalanda_results_{today}.html")
    write_html(df_out, html_path, run_time, len(to_scan))

    print(f"\n{Fore.GREEN}  Master CSV  -> {MASTER_CSV}")
    print(f"  HTML Report -> {html_path}")
    print(f"\n  Open {html_path} in a browser for the full interactive report.")
    left = len(SP500_DEDUP) - len(scanned_all)
    print(f"  {left} stocks remaining in this cycle ({len(scanned_all)}/{len(SP500_DEDUP)} scanned).\n")

if __name__ == "__main__":
    main()