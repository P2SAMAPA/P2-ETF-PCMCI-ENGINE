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
    """
    tickers = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK

    print(f"[loader] Loading data for Option {option} ({len(tickers)} ETFs)...")

    master = _load(cfg.FILE_MASTER)

    # ── Debug: show what's in master ──────────────────────────────────────────
    print(f"[loader] master shape: {master.shape}, "
          f"range: {master.index[0].date()} → {master.index[-1].date()}")

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

    print(f"[loader] returns shape: {returns.shape}, "
          f"range: {returns.index[0].date()} → {returns.index[-1].date()}")

    # Macro variables
    macro_cols = [c for c in cfg.MACRO_VARS if c in master.columns]
    macro = master[macro_cols].copy() if macro_cols else pd.DataFrame(index=master.index)
    print(f"[loader] macro cols found: {macro_cols}")

    # ── Alignment: use returns as the date authority ───────────────────────────
    # Step 1: intersect on dates present in both
    common = returns.index.intersection(macro.index) if not macro.empty else returns.index

    returns = returns.reindex(common)
    macro   = macro.reindex(common)

    # Step 2: ffill + bfill macro so leading/trailing NaNs don't drop rows
    if not macro.empty:
        macro = macro.ffill().bfill()

    # Step 3: drop only rows where ALL return columns are NaN (truly missing data)
    valid = returns.notna().any(axis=1)
    common = common[valid]
    returns = returns.reindex(common)
    macro   = macro.reindex(common) if not macro.empty else pd.DataFrame(index=common)

    # ── Splits ────────────────────────────────────────────────────────────────
    train_mask = common <= cfg.TRAIN_END
    oos_mask   = common >= cfg.LIVE_START

    print(f"[loader] Option {option}: {len(common)} days | "
          f"train={train_mask.sum()} | OOS={oos_mask.sum()}")
    print(f"[loader] Range: {common[0].date()} → {common[-1].date()}")

    if train_mask.sum() == 0:
        raise ValueError(
            f"[loader] FATAL: train set is empty for Option {option}. "
            f"Data range {common[0].date()} → {common[-1].date()} has no dates "
            f"<= TRAIN_END={cfg.TRAIN_END}. Check master.parquet in HuggingFace dataset."
        )

    return {
        "option":      option,
        "tickers":     tickers,
        "benchmark":   benchmark,
        "returns":     returns,
        "macro":       macro,
        "dates":       common,
        "train_ret":   returns[train_mask],
        "oos_ret":     returns[oos_mask],
        "train_macro": macro[train_mask],
        "oos_macro":   macro[oos_mask],
    }
