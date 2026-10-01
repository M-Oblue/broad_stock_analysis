# broad_stock_analysis

Multi-horizon stock analysis and suggestion tool for NSE (Indian equities),
screening the Nifty 500 across three distinct holding periods.

**Educational project only. Nothing here is investment advice.** Every output
requires independent confirmation before you risk money on it.

---

## The three horizons

| Horizon | Hold | Primary method | Signal mix |
|---|---|---|---|
| **Long** | 2-5 years | Quality compounding + GARP | Fundamentals only |
| **Medium** | 1-2 years | Earnings momentum (PEAD) + 12-1 relative strength | ~50/50 |
| **Short** | weeks-months | Trend following + ATR risk control | Technicals; fundamentals as a filter |

These are genuinely different problems, not one model with three lookbacks:

- Over **2-5 years** return is approximately earnings growth ± multiple change.
  Chart patterns are noise, and recent-winner momentum actively *hurts* —
  equities mean-revert at long horizons. The long engine therefore excludes
  RSI, MACD, breakouts **and** 12-month momentum.
- Over **1-2 years** post-earnings-announcement drift and 12-1 cross-sectional
  momentum are at their most reliable. A quality floor keeps leveraged junk out.
- Over **weeks-months** risk management *is* the edge: ATR stops, R-multiple
  targets, position sizing, and staying out of earnings-date landmines.

That separation is visible in the output. Across the top 50 names, **long and
short overlap by only 2%**:

| Stock | Long rank | Medium rank | Short rank |
|---|---|---|---|
| ITC | **9** | 353 | 397 |
| INFY | 23 | 403 | 422 |
| CPPLUS | 261 | **1** | 62 |
| DIVISLAB | 149 | **7** | 49 |

---

## Does it actually work? Measured, not assumed

Walk-forward validation, scoring each symbol using only data available at each
rebalance date.

| | Short (63-day forward) | Medium (252-day forward) |
|---|---|---|
| Periods tested | 45 | 36 |
| Mean IC | **-0.027** | **+0.074** |
| Periods with positive IC | 37.8% | 61.1% |
| Quintile returns monotonic | No | **Yes** |
| Top-20 vs universe | **-0.97pp** | **+8.53pp** |

**The medium-term engine shows a useful edge.** Quintile returns rise
monotonically (27.4% -> 28.2% -> 33.2% -> 40.5% -> 44.4%), which matters more
than the headline IC: it means the relationship holds across the whole score
range rather than being driven by a few outliers.

**The short-term engine shows no edge, and slightly negative.** Its top 20
*underperformed* the universe. This is reported rather than tuned away — the
weights were not refitted until the backtest looked good, because that would
have produced an overfitted number with no out-of-sample meaning. Use the
short-term screen as a starting list to investigate, not as a signal.

**The long-term engine is deliberately not backtested.** Five years of data
yields too few independent 2-5 year observations to distinguish skill from luck,
and yfinance carries no point-in-time fundamentals, so any fundamental backtest
would suffer look-ahead and survivorship bias. A backtest that cannot be trusted
is worse than none: it lends false confidence to the weights. The long engine's
claim to attention is that **every point it awards is explained**, so the
reasoning can be audited even though the weights cannot be validated.

---

## Setup

```powershell
pip install -r requirements.txt
```

Fetch data once (~35 minutes total; all cached and incremental thereafter):

```powershell
python -m src.data.fetch_prices          # ~1 min   561k rows, 500 symbols, 5y
python -m src.data.fetch_fundamentals    # ~30 min  95k statement rows
python -m src.data.fetch_ownership       # ~6 min   promoter holding + pledging
python -m src.data.validate              # data-quality report
```

## Usage

```powershell
python analyze.py --horizon long               # 2-5 year candidates
python analyze.py --horizon medium             # 1-2 year candidates
python analyze.py --horizon short              # weeks-to-months candidates
python analyze.py --horizon all

python analyze.py --horizon long --top 30 --detail 10
python analyze.py --horizon long --ticker INFY.NS ITC.NS
python analyze.py --horizon medium --min-coverage 0.7
python analyze.py --refresh                    # refetch everything first
```

Each run prints a ranked table, a full explanation of the top candidates, and
writes `data/processed/<YYYY-MM-DD>/ranked_<horizon>.csv`.

### What the output looks like

```
TRAVELFOOD.NS  --  score 92.5/100 (strong)
  sector: Consumer Services   model: generic   coverage: 80%
  What earned points:
    + ROCE above 25% (5y avg) -- excellent capital efficiency
    + Very stable ROCE (5y std <3pp)
    + Operating cash flow exceeds reported profit
    + Deleveraging over time
  What did not:
    - P/E near the top of its own 5y range -- little re-rating room
  Caveats:
    * Only 4 years of financial history -- a 2-5 year judgement is being
      made on a short record
```

The explanation is the product. A score is a summary of reasoning that can be
wrong in ways the number cannot express, so the report leads with *why*.

---

## How it works

```
src/
  data/       universe, prices, fundamentals, ownership, validation
  metrics/    technical, fundamental, valuation, governance, risk, regime
  scoring/    base + long_term / medium_term / short_term / financials
  portfolio/  ATR sizing, concentration and correlation constraints
  report/     console + CSV with per-stock explanations
  backtest/   walk-forward IC and quantile spreads (short & medium only)
  common/     vendored indicator math, dated run folders, console guard
analyze.py    unified CLI
```

### Design decisions that matter

**Missing data is not scored as zero.** A criterion whose input is unavailable
is dropped from both the earned and available totals, so lenders and
recently-listed companies are not sunk for data that does not exist. Every score
reports its `coverage`.

**...but a wholly unmeasurable block scores neutral, not skipped.** Normalising
away a fifth of the score silently rescores a company on its strengths alone.
Observed live: INFY topped the long screen at 94/100 *because* its valuation is
currency-suppressed, so the one dimension that might have held it back never
applied. It now scores neutral there and ranks 33rd.

**Valuation is never absolute.** A P/E of 45 is expensive for a steel mill and
cheap for a consumer compounder, so screening on raw P/E finds cheap *sectors*
and ranks the index by industry every run. Every multiple is expressed as an
own-history percentile and a sector z-score (median/MAD, not mean/stdev, so one
200x outlier cannot drag a peer group).

**Penalties sit outside the weighted average.** A severely pledged promoter must
not be offset by a good margin trend.

**Four business models, not one.** NSE's "Financial Services" bucket holds 101
names spanning lenders, insurers and fee businesses. Routing is 60 lenders /
13 insurers / 428 generic, split on yfinance sub-industry and confirmed by EBIT
availability (0 of 26 banks report it; 9 of 11 asset managers do).

**Market regime gates the momentum horizons.** Nifty vs its 200DMA; in a
risk-off tape the short and medium reports warn that the ranking is relative,
not absolute — the best of a falling list is still falling.

---

## Known data hazards (all guarded, all found in live data)

1. **Currency mismatch.** INFY and HCLTECH report statements in **USD** while
   priced in **INR**, so yfinance's `enterpriseToEbitda` reads 936 and 1168
   instead of ~10.6. Statement-over-statement metrics (ROCE, margins, CAGR, D/E)
   are unaffected; price-over-statement ones break. A random 60-ticker sample
   found **zero** instances — hence a systematic guard rather than spot-checking.

2. **Promoter pledging field selection.** NSE's `numSharesPledged` includes
   non-promoter and NBFC pledges: SUZLON reports 943,080,314 encumbered shares
   against **0.00%** promoter pledging, and ITC/HDFCBANK/YESBANK look equally
   alarming while their promoters have pledged nothing. The correct field is
   **`percPromoterShares`**.

3. **Two NSE endpoints disagree on promoter holding.** The pledge endpoint says
   HDFCBANK 13.32% and INFY 20.76%; the official shareholding-pattern filing
   says 0.00% and 13.82%. SHP is authoritative — HDFC Bank has *no promoter*
   post-merger, so the pledge endpoint would invent a 13% stake.

4. **Quarterly series have holes.** RELIANCE is missing 2025-09-30 entirely, so
   matching "the same quarter last year" by position compared Jun-2026 against
   Mar-2025 and reported an 18.4% YoY where the truth was 27.0%. Year-ago
   quarters are matched by **date**, within a 45-day tolerance.

5. **Splits rewrite history retroactively.** `auto_adjust` re-adjusts all prior
   bars, so a naive incremental append stacks adjusted new bars onto stale old
   ones and produces a price cliff mid-series. Each run re-fetches an overlap
   window and re-downloads any symbol whose closes drifted more than 1%.

6. **NaN is truthy in Python.** A missing `financialCurrency` compared as a
   string against `"INR"` reads as a mismatch; JSWDULUX.NS had every multiple
   suppressed on that basis. Absent data now means "cannot determine".

---

## Limitations

- **Asset quality is not assessed for lenders.** GNPA, provision coverage,
  capital adequacy, CASA and credit cost are unavailable from this data source.
  A bank can post excellent ROA right up until the credit cycle turns. Every
  lender score says so explicitly.
- **Insurers are scored roughly.** Embedded value, VNB margin, solvency ratio
  and persistency are unavailable, so insurance candidates carry a caveat that
  the score is a rough quality and valuation read only.
- **Backtests model no costs, slippage or position sizing.** They validate
  signal quality, not achievable returns.
- **42 of 500 names have under 500 bars** of history; long-horizon valuation
  percentiles and 12-1 momentum are unavailable for them and report as missing
  rather than being computed from a stub.
- **The tool ranks and explains; it does not trade.** It will tell you a
  position breaches a concentration cap. It will not tell you which of two
  overweight names to trim.

---

## Relationship to `stock_analysis`

A standalone sibling of the older `stock_analysis` project, which is left
untouched. No cross-repo imports. Three utilities are **vendored** into
`src/common/` with attribution because they are objective infrastructure
carrying no strategy bias: `indicators.py` (Wilder-correct RSI/ATR/ADX/MACD),
`run_paths.py` (dated run folders) and `io_console.py` (Windows cp1252 guard).
All scoring logic, the fundamentals schema and the pipeline structure were
rebuilt from first principles.

## Tests

```powershell
python -m pytest tests/ -q     # 249 tests
```