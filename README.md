# Nalanda-Style US Stock Screener

A free, fully local stock screener for the S&P 500 built on [Pulak Prasad / Nalanda Capital](https://www.nalandainvest.com) principles.  
No API key required. Data from [yfinance](https://github.com/ranaroussi/yfinance) + SEC EDGAR.

---

## What it does

### Screening philosophy

Nalanda Capital (Pulak Prasad) invests in businesses with durable competitive advantages, zero debt, and no dependence on the capital markets. The screener operationalises three core ideas:

1. **Historical ROCE is the single primary filter.**  
   ROCE = EBIT / (Total Assets − Current Liabilities) — pre-tax, Nalanda definition.  
   Only businesses that have compounded at ≥ 20% ROCE over multiple years qualify.

2. **Debt is near-disqualifying.**  
   Near-zero D/E preferred; D/E > 1x is a severe penalty.

3. **Valuation bonus activates only after quality is confirmed.**  
   P/E vs 4-year median and earnings yield vs 10-year Treasury yield.

### Scoring (100 pts max)

| Block | Max pts |
|-------|---------|
| ROCE TTM level | 25 |
| ROCE historical consistency | 30 |
| Balance sheet / debt | 20 |
| Revenue + earnings growth (~3-4yr CAGR) | 10 |
| CFO quality (CFO / Net Income) | 5 |
| Valuation (cheapness test) | 10 |
| Industry quality penalty | −0 to −20 |

**Verdicts:** STRONG PASS (≥ 78), PASS (≥ 62), WATCH (≥ 45), FAIL (< 45 or ROCE gate fails).

### Three-Phase Entry Algorithm (per-stock)

When you click a stock in the HTML report (or run `--entry TICKER`), the screener computes a full entry checklist:

| Phase | What it checks |
|-------|---------------|
| **1 — Quality filter** | ROCE > 20%, Net Debt/EBITDA < 1.0, FCF/NI > 80% |
| **2 — FCF-yield Buy Limit** | Price where FCF Yield = 10Y Treasury + 3% (the "cash anchor") |
| **3 — Reverse DCF** | Implied growth vs Sustainable growth (ROCE × reinvestment rate) |
| **4 — Technical** | RSI(14) < 35, Price vs 200-day SMA |

**Buy signal:** Quality Pass + Price below Buy Limit (FCF) + Implied Growth < Sustainable + RSI < 35 near 200d SMA.

> **Note on Reverse DCF:** When FCF ≥ Net Income (high cash conversion), reinvestment rate = 0 and Sustainable Growth = 0, so Reverse DCF shows "—" (N/A). For those names, rely on FCF yield vs hurdle and P/E vs history instead.

### Output

- **Interactive HTML report** — sortable table with ROCE, growth, valuation, entry prices, quality filter, buy limit, RSI and 200d SMA. Click any row for the full Three-Phase Entry panel.
- **Master CSV** (`nalanda_master.csv`) — cumulative database of all stocks that have ever passed.
- **Console output** — color-coded progress with score and verdict per ticker.

### Cross-verification with SEC EDGAR (`nalanda_verify.py`)

A separate script cross-checks ROCE between yfinance and SEC 10-K filings (up to 10 years). No API key.

---

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip

---

## Setup

### With uv (recommended)

```bash
cd nalanda_screener
uv venv .venv
uv pip install --python .venv/bin/python yfinance pandas tabulate colorama requests
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### With pip

```bash
cd nalanda_screener
python -m venv .venv
source .venv/bin/activate
pip install yfinance pandas tabulate colorama requests
```

---

## Running the screener

### Full S&P 500 scan (resumes from last run automatically)

```bash
python nalanda_screener.py
```

- Scans remaining stocks from the current cycle, then resets automatically.
- Results saved to `nalanda_results_YYYY-MM-DD.html` and `nalanda_master.csv`.
- Open the HTML file in any browser for the interactive report.

### Single-ticker entry checklist (Phases 1–4)

```bash
python nalanda_screener.py --entry AAPL
python nalanda_screener.py --entry ADBE
```

Prints the full Three-Phase Entry checklist including ROCE, quality pass, FCF-yield buy limit, implied vs sustainable growth, RSI and 200d SMA.

### Debug mode (raw yfinance data for one ticker)

```bash
python nalanda_screener.py --debug
python nalanda_screener.py --debug MSFT
```

Dumps all raw yfinance fields plus computed metrics: ROCE per year, growth, P/E, FCF, D/E.

### Cross-verify with SEC EDGAR (10-year ROCE)

```bash
python nalanda_verify.py AAPL
python nalanda_verify.py AAPL MSFT GOOGL
```

Fetches annual 10-K data from `data.sec.gov` (no API key) and compares ROCE year-by-year against yfinance.

---

## Configuration

Edit the constants near the top of `nalanda_screener.py`:

```python
PAUSE                       = 0.6   # seconds between requests
MAX_STOCKS                  = None  # None = full S&P 500; set to e.g. 10 for a quick test
HURDLE_RISK_PREMIUM         = 0.03  # FCF yield hurdle = 10Y Treasury + this
REVERSE_DCF_DISCOUNT_RATE   = 0.10  # discount rate for reverse DCF
RSI_OVERSOLD                = 35    # RSI threshold for "Technical Ready"
QUALITY_ROCE_MIN            = 20    # % ROCE for Phase 1 quality pass
QUALITY_NET_DEBT_EBITDA_MAX = 1.0   # max Net Debt/EBITDA for quality pass
QUALITY_FCF_NI_MIN          = 0.80  # min FCF/NI for quality pass
```

---

## Key formulas

| Metric | Formula |
|--------|---------|
| ROCE | EBIT / (Total Assets − Current Liabilities) |
| FCF Yield | Free Cash Flow / Enterprise Value |
| Buy Limit (FCF) | EPS-based: Target EV = FCF / hurdle; implied equity = Target EV − Net Debt; price = implied equity / shares |
| Implied growth | (EV × 0.10 − FCF) / (EV + FCF)  *(perpetual DCF, r = 10%)* |
| Sustainable growth | ROCE × (1 − FCF/NI)  *(0 when FCF ≥ NI)* |
| P/E entry | min(EPS / bond yield, EPS × median P/E) |
| RSI(14) | Standard Wilder RSI on daily close prices |

---

## File structure

```
nalanda_screener/
├── nalanda_screener.py        # main screener
├── nalanda_verify.py          # SEC EDGAR cross-verification
├── nalanda_master.csv         # cumulative results database (auto-generated)
├── nalanda_results_*.html     # daily HTML reports (auto-generated)
├── .scanned_tickers.json      # progress tracking (auto-generated)
├── .sp500_cache.json          # S&P 500 list cache (auto-generated)
└── .sec_cache/                # SEC EDGAR JSON cache (auto-generated)
```

---

## Disclaimer

This is a personal research tool for educational purposes only. It is not investment advice. Always do your own research before making any investment decision. Data from yfinance and SEC EDGAR may be delayed or incorrect. Verify numbers before acting on them.

---

## Credits

Investing framework: [Pulak Prasad — *What I Learned About Investing from Darwin*](https://www.nalandainvest.com)  
Data: [yfinance](https://github.com/ranaroussi/yfinance), [SEC EDGAR](https://www.sec.gov/developer)
