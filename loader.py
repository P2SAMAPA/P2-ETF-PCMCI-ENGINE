# loader.py — Loads data from shared p2-etf-deepm-data HF dataset

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config as cfg


def _load(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=cfg.HF_TOKEN or None,
        force_download=True,
    )
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Drop residual index columns
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ("date", "index", "level_0"):
            df = df.drop(columns=[col])
    return df.sort_index()


def get_option_data(option: str) -> dict:
    """
    Load and prepare data for Option A (FI) or Option B (Equity).

    Returns dict with:
        returns     — log returns for ETF universe
        macro       — raw FRED macro series
        tickers     — list of ETF tickers
        benchmark   — benchmark ticker string
        dates       — full date index
        train_ret   — returns up to TRAIN_END
        oos_ret     — returns from LIVE_START onward
        train_macro — macro up to TRAIN_END
        oos_macro   — macro from LIVE_START onward
    """
    tickers   = cfg.FI_ETFS   if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK

    print(f"[loader] Loading data for Option {option} ({len(tickers)} ETFs)...")

    master = _load(cfg.FILE_MASTER)

    # Log returns for ETFs
    logret_cols = [f"{t}_logret" for t in tickers if f"{t}_logret" in master.columns]
    if not logret_cols:
        # Fallback: compute from close prices
        close_cols = [f"{t}_Close" for t in tickers if f"{t}_Close" in master.columns]
        prices = master[close_cols].copy()
        prices.columns = [c.replace("_Close", "") for c in prices.columns]
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = master[logret_cols].copy()
        returns.columns = [c.replace("_logret", "") for c in returns.columns]

    # Macro variables
    macro_cols = [c for c in cfg.MACRO_VARS if c in master.columns]
    macro = master[macro_cols].copy().ffill()

    # Align
    common = returns.index.intersection(macro.index)
    returns = returns.reindex(common).dropna(how="all")
    macro   = macro.reindex(common).ffill().dropna(how="all")
    common  = returns.index.intersection(macro.index)
    returns = returns.reindex(common)
    macro   = macro.reindex(common)

    # Splits
    train_mask = common <= cfg.TRAIN_END
    oos_mask   = common >= cfg.LIVE_START

    print(f"[loader] Option {option}: {len(common)} days | "
          f"train={train_mask.sum()} | OOS={oos_mask.sum()}")
    print(f"[loader] Range: {common[0].date()} → {common[-1].date()}")

    return {
        "option":     option,
        "tickers":    tickers,
        "benchmark":  benchmark,
        "returns":    returns,
        "macro":      macro,
        "dates":      common,
        "train_ret":  returns[train_mask],
        "oos_ret":    returns[oos_mask],
        "train_macro":macro[train_mask],
        "oos_macro":  macro[oos_mask],
    }
