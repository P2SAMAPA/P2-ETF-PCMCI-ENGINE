# config.py — P2-ETF-PCMCI-ENGINE
# Causal Discovery ETF Signal Engine
# Reads data from shared p2-etf-deepm-data dataset

import os

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-deepm-data")
HF_RESULTS_REPO = os.environ.get("HF_RESULTS_REPO", "P2SAMAPA/p2-etf-pcmci-results")

# ── Data files (from shared DeePM dataset) ─────────────────────────────────────
FILE_MASTER      = "data/master.parquet"
FILE_ETF_RETURNS = "data/etf_returns.parquet"
FILE_MACRO_FRED  = "data/macro_fred.parquet"

# ── Option A — Fixed Income / Alternatives ─────────────────────────────────────
FI_ETFS = [
    "TLT",   # 20+ Year Treasury Bond
    "LQD",   # Investment Grade Corporate Bond
    "HYG",   # High Yield Corporate Bond
    "VNQ",   # Real Estate (REITs)
    "GLD",   # Gold
    "SLV",   # Silver
    "PFF",   # Preferred Stock
    "MBB",   # Mortgage-Backed Securities
]
FI_BENCHMARK = "AGG"

# ── Option B — Equity Sectors ──────────────────────────────────────────────────
EQ_ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # NASDAQ 100
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "GDX",   # Gold Miners
    "IWF",
    "IWM",
    "XSD",
    "XBI",
    "XME",   # Metals & Mining
]
EQ_BENCHMARK = "SPY"

# ── Macro features used as conditioning variables ──────────────────────────────
# PCMCI+ uses these as context nodes in the causal graph
MACRO_VARS = ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]

# ── PCMCI+ hyperparameters ─────────────────────────────────────────────────────
# Rolling window for causal graph estimation
ROLLING_WINDOW  = 252        # ~1 year of trading days
MIN_WINDOW      = 120        # minimum days needed to fit
MAX_LAG         = 5          # maximum causal lag (days) to test
PC_ALPHA        = 0.05       # significance threshold for causal links
COND_IND_TEST   = "parcorr"  # conditional independence test (parcorr = partial correlation)

# ── Shrinking windows ──────────────────────────────────────────────────────────
WINDOWS = [
    {"id": 1, "start": "2008-01-01"},
    {"id": 2, "start": "2010-01-01"},
    {"id": 3, "start": "2012-01-01"},
    {"id": 4, "start": "2014-01-01"},
    {"id": 5, "start": "2016-01-01"},
    {"id": 6, "start": "2018-01-01"},
    {"id": 7, "start": "2020-01-01"},
    {"id": 8, "start": "2022-01-01"},
]
TRAIN_END  = "2024-12-31"
LIVE_START = "2025-01-01"

# ── Centrality scoring ─────────────────────────────────────────────────────────
# How to score each ETF from the causal graph
# "out_degree"  — assets that causally drive others (leaders)
# "in_degree"   — assets driven by others (followers)
# "net_degree"  — out minus in (pure leaders)
CENTRALITY_METRIC = "out_degree"

# Minimum causal strength to count a link
MIN_LINK_STRENGTH = 0.05

# ── Signal generation ──────────────────────────────────────────────────────────
# Combine causal centrality with momentum for final signal
MOMENTUM_WINDOWS   = [5, 21, 63]   # days
MOMENTUM_WEIGHT    = 0.3            # 30% momentum, 70% causal centrality
CONVICTION_THRESHOLD = 0.15         # min score gap to have conviction

# ── Local dirs ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
DATA_DIR    = "data"
