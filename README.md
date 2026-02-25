# Nalanda-Style US Stock Screener

A free, fully local stock screener for the S&P 500 built on [Pulak Prasad / Nalanda Capital](https://www.nalandainvest.com) principles.  
No API key required. Data from [yfinance](https://github.com/ranaroussi/yfinance) + SEC EDGAR.

---

## What it does

### Screening philosophy

Nalanda Capital (Pulak Prasad) invests in businesses with durable competitive advantages, near-zero debt, and no dependence on capital markets. The screener operationalises three core ideas:

1. **Historical ROCE is the single primary filter.**  
   ROCE = EBIT / (Total Assets − Current Liabilities) — pre-tax, Nalanda definition.  
   Only businesses that have compounded at ≥ 20% ROCE over multiple years qualify.

2. **Debt is near-disqualifying.**  
   Near-zero D/E preferred; D/E > 1× is a severe penalty.

3. **Valuation bonus activates only after quality is confirmed.**  
   P/E vs 4-year median and earnings yield vs 10-year Treasury yield.

### Scoring (100 pts max)

| Block | Max pts |
|-------|---------|
| ROCE TTM level | 25 |
| ROCE historical consistency | 30 |
| Balance sheet / debt | 20 |
| Revenue + earnings growth (3–4yr CAGR) | 10 |
| CFO quality (CFO / Net Income) | 5 |
| Valuation (cheapness test) | 10 |
| Industry quality penalty | −0 to −20 |

**Verdicts:** STRONG PASS (≥ 78), PASS (≥ 62), WATCH (≥ 45), FAIL (< 45 or ROCE gate fails).

---

### Three-Phase Entry Algorithm (per-stock)

When you click a stock in the HTML report (or run `--entry TICKER`), the screener runs a full valuation and entry checklist across four phases:

#### Phase 1 — Quality Filter

| Check | Threshold |
|-------|-----------|
| ROCE TTM | > 20% |
| Net Debt / EBITDA | < 1.0× |
| FCF / NI | > 80% — or "reinvesting" if ROCE is stable (Option B logic) |

**FCF/NI Option B:** A low FCF/NI ratio (e.g. 45%) only fails quality if ROCE is *also* declining into marginal territory (< 25%). A company like MSFT with ROCE at 30% and low FCF/NI is classified as "reinvesting" — heavy capex, not earnings manipulation.

#### Phase 2 — FCF Valuation (two prices + range)

Two prices are always shown, so you can see both the floor and the realistic entry range:

| Price | Formula | What it means |
|-------|---------|---------------|
| **FCF Floor** | FCF ÷ hurdle | Zero-growth backstop. At this price, FCF yield = hurdle with no growth assumed. Never a buy target — just the absolute floor. |
| **FCF Fair Value** | Range: conservative to optimistic | Shown as `$low – $high (mult_low× – mult_high× FCF)`. Two methods depending on reinvestment: |

**Method selection (FCF/NI threshold = 0.90):**

- **FCF/NI ≤ 0.90** (meaningful reinvestment) → **Gordon Growth Model**  
  `Fair EV = FCF ÷ (hurdle − g)`  
  where `g = ROCE × reinvestment_rate`, capped at `hurdle − 3%` (conservative) and `hurdle − 1%` (optimistic).  
  Range reflects sensitivity to the growth assumption.

- **FCF/NI > 0.90** (near-zero reinvestment — SaaS, payments, software) → **EV/FCF Median Reversion**  
  `Fair EV = FCF × historical_median_EV/FCF`  
  Conservative = median × 0.80, optimistic = median.  
  Historical median reconstructed from annual price × shares + debt − cash over up to 5 years.

The implied **EV/FCF multiple** is shown in parentheses alongside each price so you can judge whether the multiple is historically reasonable.

#### Phase 3 — Reverse DCF

`Implied Growth = (EV × 0.10 − FCF) / (EV + FCF)` — growth rate baked into current price.  
`Sustainable Growth = ROCE × (1 − FCF/NI)` — maximum growth from reinvested earnings.

- **Buy** if Implied Growth < Sustainable Growth (market is underpricing the business)
- **Not Buy** if Implied Growth > Sustainable Growth (market expects more than the business can deliver)
- **— (N/A)** if FCF/NI > 0.90 (reinvestment rate ≈ 0, formula not meaningful). Use FCF yield vs hurdle and EV/FCF vs history instead.

#### Phase 4 — Technical

| Check | Threshold |
|-------|-----------|
| RSI(14) | < 35 (oversold) |
| Price vs 200-day SMA | below |

Both must be true for **Technical: Ready**.

**Buy signal:** Quality Pass + Price ≤ conservative FCF Fair Value + Implied < Sustainable + RSI < 35 below 200d SMA.

---

### Output

- **Interactive HTML report** — sortable table with ROCE, growth, valuation, entry prices, quality filter, FCF Fair Value range, RSI and 200d SMA. Click any row for the full Three-Phase Entry panel with all metrics.
- **Master CSV** (`nalanda_master.csv`) — cumulative database of all stocks that have ever passed.
- **Console output** — colour-coded progress with score and verdict per ticker.

---

### Cross-verification with SEC EDGAR (`nalanda_verify.py`)

A separate script cross-checks ROCE between yfinance and SEC 10-K filings (up to 10 years). No API key needed.

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

### Full S&P 500 scan

```bash
python nalanda_screener.py
```

Scans remaining stocks from the current cycle, then resets automatically.  
Results saved to `nalanda_results_YYYY-MM-DD.html` and `nalanda_master.csv`.  
Open the HTML file in any browser for the interactive report.

### Single-ticker entry checklist (Phases 1–4)

```bash
python nalanda_screener.py --entry AAPL
python nalanda_screener.py --entry EXLS
```

Prints the full Three-Phase Entry checklist:
- Phase 1: ROCE, quality pass, Net Debt/EBITDA, FCF/NI
- Phase 2: FCF Yield, Hurdle, FCF Fair Value range with multiples, FCF Floor
- Phase 3: Implied Growth vs Sustainable Growth, Reverse DCF signal
- Phase 4: RSI(14), 200d SMA, Technical Ready/Not yet

### Debug mode (raw data + all computed metrics)

```bash
python nalanda_screener.py --debug MSFT
```

Dumps all raw yfinance fields plus every computed metric: ROCE per year, growth CAGRs, P/E history, FCF, EV/FCF, valuation range. **Use this for deep-dive analysis** — paste the output into an LLM for a narrative breakdown (see AI Analysis below).

### Cross-verify with SEC EDGAR (10-year ROCE)

```bash
python nalanda_verify.py AAPL
python nalanda_verify.py AAPL MSFT GOOGL
```

Fetches annual 10-K data from `data.sec.gov` (no API key) and compares ROCE year-by-year against yfinance.

---

## AI-assisted analysis (free, no API key)

The `--debug` output contains all the data needed for a full Nalanda-style investment analysis. Paste it into any LLM (e.g. Cursor, Claude, ChatGPT) for a narrative breakdown covering:

- Business moat and quality
- Valuation vs FCF Fair Value range and historical multiples
- Key risks (debt, ROCE trend, industry disruption)
- Technical setup and entry timing
- Verdict

### Quick workflow

Add these aliases to your `~/.bashrc`:

```bash
# Full debug — rich data for deep analysis
alias nscreen='f(){
  cd ~/Documents/nalanda_screener && source .venv/bin/activate
  python3 nalanda_screener.py --debug "$1" 2>&1 | tee /tmp/nscreen_last.txt
  xclip -selection clipboard < /tmp/nscreen_last.txt
  echo "\n✓ Output copied to clipboard — paste into your LLM chat"
}; f'

# Entry checklist only — quick numbers
alias nentry='f(){
  cd ~/Documents/nalanda_screener && source .venv/bin/activate
  python3 nalanda_screener.py --entry "$1" 2>&1 | tee /tmp/nentry_last.txt
  xclip -selection clipboard < /tmp/nentry_last.txt
  echo "\n✓ Copied"
}; f'
```

Reload: `source ~/.bashrc`

Usage:
```bash
nscreen EXLS   # deep analysis
nentry LULU    # quick entry check
```

Then paste the clipboard output into your LLM and ask "analyze this" or "is this a buy?".

---

## Configuration

Edit the constants near the top of `nalanda_screener.py`:

```python
PAUSE                       = 0.6    # seconds between requests (be polite to Yahoo)
MAX_STOCKS                  = None   # None = full S&P 500; set to e.g. 10 for a quick test
HURDLE_RISK_PREMIUM         = 0.03   # FCF yield hurdle = 10Y Treasury + this (default: +3%)
REVERSE_DCF_DISCOUNT_RATE   = 0.10   # discount rate for Reverse DCF (default: 10%)
RSI_OVERSOLD                = 35     # RSI threshold for "Technical Ready"
QUALITY_ROCE_MIN            = 20     # % ROCE required for Phase 1 quality pass
QUALITY_NET_DEBT_EBITDA_MAX = 1.0    # max Net Debt/EBITDA for quality pass
QUALITY_FCF_NI_MIN          = 0.80   # min FCF/NI for quality pass (Option B logic applies)
```

---

## Key formulas

| Metric | Formula |
|--------|---------|
| ROCE | EBIT / (Total Assets − Current Liabilities) |
| FCF | Operating Cash Flow − Capital Expenditure |
| FCF Yield | FCF / Enterprise Value |
| Hurdle Rate | 10Y Treasury Yield + HURDLE_RISK_PREMIUM |
| FCF Floor | (FCF / hurdle − Net Debt) / Shares |
| FCF Fair Value (Gordon) | (FCF / (hurdle − g) − Net Debt) / Shares; g = ROCE × (1 − FCF/NI), capped |
| FCF Fair Value (EV/FCF) | (FCF × hist\_median\_EV\_FCF − Net Debt) / Shares; used when FCF/NI > 0.90 |
| Implied Growth | (EV × 0.10 − FCF) / (EV + FCF) |
| Sustainable Growth | ROCE × (1 − FCF/NI); 0 when FCF ≥ NI |
| EV/FCF Multiple | Enterprise Value / Free Cash Flow |
| P/E Entry | min(EPS / bond yield, EPS × median P/E) |
| RSI(14) | Standard Wilder RSI on 14-day daily close prices |

---

## Reading the FCF Fair Value range

```
LULU:  FCF Fair Value  $256 – $782  (33× – 80× FCF)
       FCF Floor       $106  (14× FCF)
       Current price   $179  (21× FCF)
```

| Price | Multiple | Interpretation |
|-------|---------|----------------|
| $106 Floor | 14× | Pay this if FCF never grows — the absolute backstop |
| $179 Current | 21× | Market pricing ~2% perpetual FCF growth |
| $256 Conservative | 33× | Fair if FCF grows at 4% forever (hurdle − 3%) |
| $782 Optimistic | 80× | Fair if FCF grows at 6% forever (hurdle − 1%) |

The multiple column is the key sanity check. 80× FCF requires exceptional long-run confidence. 33× is historically grounded for a quality compounder. Current price below conservative fair value is the signal.

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
