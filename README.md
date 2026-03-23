# P2-ETF-PCMCI-ENGINE

**Causal Discovery ETF Signal Engine using PCMCI+**

Rather than asking *"which ETF had the best return recently?"*, this engine asks:
**"which ETF is causally driving the others right now?"**

The causal hub — the asset with the most outgoing causal links — is the leading
asset in the system. Holding the leader front-runs the propagation to the followers.

---

## Research Foundation

**PCMCI+** — Runge et al. (2019), *Science Advances*
> "Detecting and quantifying causal associations in large nonlinear time series datasets"

The gold standard for causal discovery on short, noisy financial time series.
Explicitly separates spurious correlation from true Granger causality.

---

## ETF Universe

### Option A — Fixed Income / Alternatives (benchmark: AGG)
TLT · LQD · HYG · VNQ · GLD · SLV · PFF · MBB

### Option B — Equity Sectors (benchmark: SPY)
SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME

---

## Architecture

```
Shared HF Dataset (p2-etf-deepm-data)
  etf_returns.parquet + macro_fred.parquet
          │
          ▼
PCMCI+ Causal Discovery
  ETF returns + macro conditioning variables
  → causal graph (who drives whom)
          │
          ▼
Causal Centrality Scoring
  out-degree = causal DRIVER score per ETF
          │
          ▼
Momentum Overlay (30%)
  5d + 21d + 63d cross-sectional rank
          │
          ▼
Fixed Window + Shrinking Windows (8)
  Winner = highest OOS annualised return
          │
          ▼
Daily Signal → HF Results Repo
```

---

## Data Source

Reads from shared `P2SAMAPA/p2-etf-deepm-data` — no separate seeding needed.
Results written to `P2SAMAPA/p2-etf-pcmci-results`.

---

## Setup

### Secrets

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | HuggingFace write token |
| `HF_DATASET_REPO` | `P2SAMAPA/p2-etf-deepm-data` |
| `HF_RESULTS_REPO` | `P2SAMAPA/p2-etf-pcmci-results` |

### First run

```
GitHub Actions → Daily PCMCI+ Signal → Run workflow
```

Takes ~15-30 minutes (PCMCI+ is fast on CPU — no GPU needed).

---

## Disclaimer

Research and educational purposes only. Not financial advice.
