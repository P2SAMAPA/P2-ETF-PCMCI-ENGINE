# train_and_predict.py — PCMCI+ daily pipeline
# Runs causal discovery on both fixed window and shrinking windows,
# picks best signal, saves results to HuggingFace.
#
# Usage:
#   python train_and_predict.py --option both

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from huggingface_hub import HfApi, hf_hub_download

import config as cfg
import loader
from causal_engine import run_window_analysis, build_signal_scores, run_pcmci

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


# ── Next trading day ───────────────────────────────────────────────────────────

def next_trading_day(from_date: str = None) -> str:
    nyse = mcal.get_calendar("NYSE")
    base = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    schedule = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else str((base + pd.Timedelta(days=1)).date())


# ── OOS evaluation ─────────────────────────────────────────────────────────────

def evaluate_oos(pick: str, oos_returns: pd.DataFrame) -> dict:
    """Compute OOS performance metrics for a static pick."""
    if pick not in oos_returns.columns or oos_returns.empty:
        return {"ann_return": 0.0, "sharpe": 0.0, "hit_rate": 0.0, "max_dd": 0.0}

    r = oos_returns[pick].fillna(0.0)
    ar = float(r.mean() * 252)
    av = float(r.std() * np.sqrt(252))
    sh = ar / (av + 1e-8)
    curve = (1 + r).cumprod()
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    hr = float((r > 0).mean())

    return {
        "ann_return": round(ar, 4),
        "ann_vol":    round(av, 4),
        "sharpe":     round(sh, 4),
        "hit_rate":   round(hr, 4),
        "max_dd":     round(dd, 4),
    }


# ── History management with HF persistence ─────────────────────────────────────

def _load_remote_history(option: str) -> list:
    """Download existing history from Hugging Face, if available."""
    try:
        # Attempt to download the history file from the results repo
        local_path = hf_hub_download(
            repo_id=cfg.HF_RESULTS_REPO,
            filename=f"results/signal_history_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            local_dir=cfg.RESULTS_DIR,
            local_dir_use_symlinks=False,
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception:
        # No existing history – start fresh
        return []


def update_history(result: dict, option: str) -> None:
    """Append today's signal to history, preserving remote history."""
    # Load existing history from HF (if any)
    history = _load_remote_history(option)

    record = {
        "signal_date":  result["signal_date"],
        "pick":         result["pick"],
        "conviction":   result["conviction"],
        "source":       result["source"],
        "generated_at": result["generated_at"],
    }

    # Avoid duplicates by date
    if record["signal_date"] not in {r["signal_date"] for r in history}:
        history.append(record)

    # Save locally (will be uploaded later by upload_results.py)
    history_path = os.path.join(cfg.RESULTS_DIR, f"signal_history_{option}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[predict] History: {len(history)} records for Option {option}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_option(option: str) -> dict:
    """
    Full PCMCI+ pipeline for one option.
    1. Fixed window — full training history
    2. Shrinking windows — 8 windows, pick best OOS return
    3. Combine into final signal
    """
    print(f"\n{'='*60}")
    print(f"PCMCI+ Pipeline — Option {'A (FI)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    # Load data
    data = loader.get_option_data(option)
    tickers    = data["tickers"]
    returns    = data["returns"]
    macro      = data["macro"]
    train_ret  = data["train_ret"]
    train_mac  = data["train_macro"]
    oos_ret    = data["oos_ret"]
    last_date  = str(returns.index[-1].date())
    signal_date = next_trading_day(last_date)

    # ── 1. Fixed window — full training history ────────────────────────────────
    print(f"\n[1/2] Fixed window PCMCI+ ({cfg.TRAIN_END} training cutoff)...")
    fixed_result = run_window_analysis(
        returns=train_ret,
        macro=train_mac,
        tickers=tickers,
        window_start=str(train_ret.index[0].date()),
        window_end=cfg.TRAIN_END,
    )

    # Fixed window — evaluate on its actual test set (last 15% of full data)
    n_total    = len(returns)
    n_test     = int(n_total * 0.15)
    test_start = returns.index[-n_test]
    test_ret   = returns[returns.index >= test_start]

    fixed_pick  = fixed_result["top_pick"] if fixed_result else tickers[0]
    fixed_scores = fixed_result["scores"]  if fixed_result else {}
    fixed_oos   = evaluate_oos(fixed_pick, test_ret)

    print(f"  Fixed pick: {fixed_pick} | "
          f"Test return ({test_start.date()}→today): {fixed_oos['ann_return']*100:.2f}%")

    # ── 2. Shrinking windows ───────────────────────────────────────────────────
    print(f"\n[2/2] Shrinking windows ({len(cfg.WINDOWS)} windows)...")
    window_results = []
    best_window    = None
    best_return    = -float("inf")

    for w in cfg.WINDOWS:
        print(f"  Window {w['id']}: {w['start']} → {cfg.TRAIN_END}...")
        try:
            result = run_window_analysis(
                returns=returns,
                macro=macro,
                tickers=tickers,
                window_start=w["start"],
                window_end=cfg.TRAIN_END,
            )
            if result is None:
                print(f"  Window {w['id']}: insufficient data, skipping")
                continue

            oos_perf = evaluate_oos(result["top_pick"], oos_ret)
            result["oos_ann_return"] = oos_perf["ann_return"]
            result["oos_ann_vol"]    = oos_perf["ann_vol"]
            result["oos_sharpe"]     = oos_perf["sharpe"]
            result["oos_hit_rate"]   = oos_perf["hit_rate"]
            result["oos_max_dd"]     = oos_perf["max_dd"]
            result["window_id"]      = w["id"]
            window_results.append(result)

            print(f"  Window {w['id']}: pick={result['top_pick']} | "
                  f"OOS return={oos_perf['ann_return']*100:.2f}% | "
                  f"method={result['method']}")

            if oos_perf["ann_return"] > best_return:
                best_return = oos_perf["ann_return"]
                best_window = result

        except Exception as e:
            print(f"  Window {w['id']} failed: {e}")
            continue

    if best_window is None:
        best_window = fixed_result or {
            "top_pick": tickers[0], "scores": {}, "window_id": 0,
            "oos_ann_return": 0.0, "oos_sharpe": 0.0,
            "window_start": cfg.TRAIN_END, "window_end": cfg.TRAIN_END,
        }

    # ── Best overall signal ────────────────────────────────────────────────────
    # Pure comparison: highest OOS ann return wins
    fixed_ann  = fixed_oos.get("ann_return", -999)
    window_ann = best_window.get("oos_ann_return", -999)

    print(f"\n  [debug] Fixed window: pick={fixed_pick} | OOS return={fixed_ann*100:.2f}%")
    print(f"  [debug] Best shrinking: pick={best_window.get('top_pick','?')} | OOS return={window_ann*100:.2f}%")

    if window_ann > fixed_ann:
        best_pick    = best_window["top_pick"]
        best_source  = f"Shrinking Window {best_window['window_id']}"
        best_scores  = best_window["scores"]
        best_ann_ret = window_ann
        print(f"  [debug] Winner: Shrinking Window ({best_pick})")
    else:
        best_pick    = fixed_pick
        best_source  = "Fixed Window"
        best_scores  = fixed_scores
        best_ann_ret = fixed_ann
        print(f"  [debug] Winner: Fixed Window ({best_pick})")

    # Conviction = score gap between top and second pick
    scores_series = pd.Series(best_scores).sort_values(ascending=False)
    conviction = float(scores_series.iloc[0] - scores_series.iloc[1]) \
                 if len(scores_series) > 1 else 0.5

    # ── Causal graph on latest rolling window (for daily insight) ──────────────
    rolling_ret = returns.iloc[-cfg.ROLLING_WINDOW:]
    rolling_mac = macro.iloc[-cfg.ROLLING_WINDOW:]
    rolling_result = run_pcmci(rolling_ret, rolling_mac, tickers)

    # Build causal network summary
    mat = rolling_result["causal_matrix"]
    causal_links = []
    for i, src in enumerate(tickers):
        for j, tgt in enumerate(tickers):
            if i != j and mat[i, j] > cfg.MIN_LINK_STRENGTH:
                causal_links.append({
                    "from": src, "to": tgt,
                    "strength": round(float(mat[i, j]), 4)
                })
    causal_links.sort(key=lambda x: x["strength"], reverse=True)

    # ── Assemble output ────────────────────────────────────────────────────────
    result = {
        "option":          option,
        "option_name":     "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":     signal_date,
        "last_data_date":  last_date,
        "generated_at":    datetime.utcnow().isoformat(),
        "pick":            best_pick,
        "conviction":      round(conviction, 4),
        "source":          best_source,
        "scores":          {k: round(float(v), 4) for k, v in best_scores.items()},
        "causal_method":   rolling_result["method"],
        "top_causal_links": causal_links[:10],

        "fixed_window": {
            "pick":       fixed_pick,
            "scores":     {k: round(float(v), 4) for k, v in fixed_scores.items()},
            "test_start": str(test_start.date()),
            "oos_return": fixed_oos["ann_return"],
            "oos_vol":    fixed_oos["ann_vol"],
            "oos_sharpe": fixed_oos["sharpe"],
            "hit_rate":   fixed_oos["hit_rate"],
            "max_dd":     fixed_oos["max_dd"],
        },

        "shrinking_window": {
            "winning_window":      best_window.get("window_id", 0),
            "winning_train_start": best_window.get("window_start", ""),
            "winning_train_end":   best_window.get("window_end", ""),
            "pick":                best_window.get("top_pick", ""),
            "scores":              {k: round(float(v), 4)
                                    for k, v in best_window.get("scores", {}).items()},
            "oos_return":          best_window.get("oos_ann_return", 0.0),
            "oos_vol":             best_window.get("oos_ann_vol", 0.0),
            "oos_sharpe":          best_window.get("oos_sharpe", 0.0),
            "oos_hit_rate":        best_window.get("oos_hit_rate", 0.0),
            "oos_max_dd":          best_window.get("oos_max_dd", 0.0),
            "all_windows":         [
                {
                    "window_id":    w["window_id"],
                    "train_start":  w["window_start"],
                    "pick":         w["top_pick"],
                    "oos_return":   w.get("oos_ann_return", 0.0),
                    "oos_sharpe":   w.get("oos_sharpe", 0.0),
                    "method":       w.get("method", ""),
                }
                for w in window_results
            ],
        },
    }

    print(f"\n  Final pick: {best_pick} | Source: {best_source} | "
          f"Conviction: {conviction:.3f} | OOS return: {best_ann_ret*100:.2f}%")

    return result


def save_results(result_A: dict = None, result_B: dict = None) -> None:
    """Save all results locally."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "option_A":     result_A,
        "option_B":     result_B,
    }
    with open(os.path.join(cfg.RESULTS_DIR, "latest_signals.json"), "w") as f:
        json.dump(combined, f, indent=2)

    for res, name, opt in [
        (result_A, "signal_A.json", "A"),
        (result_B, "signal_B.json", "B"),
    ]:
        if res:
            with open(os.path.join(cfg.RESULTS_DIR, name), "w") as f:
                json.dump(res, f, indent=2)
            update_history(res, opt)

    print(f"[predict] Results saved to {cfg.RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCMCI+ ETF Signal Pipeline")
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    res_A = res_B = None

    if args.option in ("A", "both"):
        res_A = run_option("A")

    if args.option in ("B", "both"):
        res_B = run_option("B")

    save_results(res_A, res_B)

    print("\n" + "="*60)
    print("PCMCI+ PIPELINE COMPLETE")
    if res_A:
        print(f"  Option A: {res_A['pick']} on {res_A['signal_date']} "
              f"(conviction={res_A['conviction']:.3f} | {res_A['source']})")
    if res_B:
        print(f"  Option B: {res_B['pick']} on {res_B['signal_date']} "
              f"(conviction={res_B['conviction']:.3f} | {res_B['source']})")
    print("="*60)
