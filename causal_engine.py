# causal_engine.py — PCMCI+ Causal Discovery Engine
#
# Key idea: instead of asking "which ETF had the best return recently?",
# ask "which ETF is causally DRIVING the others right now?"
#
# The causal hub — the asset with most outgoing causal links — is the
# leading asset in the system. Holding the leader front-runs the propagation
# to the followers.
#
# Uses tigramite PCMCI+ which:
#   - Handles time-series autocorrelation correctly
#   - Tests Granger causality with proper conditioning
#   - Is unbiased for lag selection up to MAX_LAG

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg

warnings.filterwarnings("ignore")

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    print("[causal] WARNING: tigramite not installed — falling back to Granger causality")


# ── Fallback: simple Granger causality ────────────────────────────────────────

def _granger_matrix(data: np.ndarray, max_lag: int = 5, alpha: float = 0.05) -> np.ndarray:
    """
    Simple Granger causality matrix as fallback when tigramite unavailable.
    Returns (n_vars, n_vars) matrix where [i,j] = strength of i→j causal link.
    Uses F-test on OLS residuals.
    """
    from sklearn.linear_model import LinearRegression

    n, p = data.shape
    causal_matrix = np.zeros((p, p))

    for target in range(p):
        y = data[max_lag:, target]

        # Restricted model: target ~ own lags only
        X_restricted = np.column_stack([
            data[max_lag - lag - 1: n - lag - 1, target]
            for lag in range(max_lag)
        ])
        reg_r = LinearRegression().fit(X_restricted, y)
        rss_r = np.sum((y - reg_r.predict(X_restricted)) ** 2)

        for source in range(p):
            if source == target:
                continue
            # Unrestricted model: target ~ own lags + source lags
            X_unrestricted = np.column_stack([
                X_restricted,
                *[data[max_lag - lag - 1: n - lag - 1, source]
                  for lag in range(max_lag)]
            ])
            reg_u = LinearRegression().fit(X_unrestricted, y)
            rss_u = np.sum((y - reg_u.predict(X_unrestricted)) ** 2)

            # F-statistic
            df1 = max_lag
            df2 = n - max_lag - 2 * max_lag - 1
            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                # Convert to causal strength (bounded 0-1)
                strength = min(f_stat / (f_stat + df2 / df1 + 1e-8), 1.0)
                if strength > cfg.MIN_LINK_STRENGTH:
                    causal_matrix[source, target] = strength

    return causal_matrix


# ── PCMCI+ causal discovery ────────────────────────────────────────────────────

def run_pcmci(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list,
) -> dict:
    """
    Run PCMCI+ on ETF returns conditioned on macro variables.

    Returns dict with:
        causal_matrix   — (n_etfs, n_etfs) strength of i→j links
        p_values        — (n_etfs, n_etfs) p-values of links
        val_matrix      — (n_etfs, n_etfs, max_lag) full lagged coefficients
        tickers         — list of ETF names
        n_obs           — number of observations used
        method          — "pcmci+" or "granger" (fallback)
    """
    # Combine ETF returns + macro as conditioning variables
    all_vars   = list(tickers) + list(macro.columns)
    n_etfs     = len(tickers)

    # Align and stack
    common_idx = returns.index.intersection(macro.index)
    ret_arr    = returns.reindex(common_idx).values
    mac_arr    = macro.reindex(common_idx).values

    # Standardise (PCMCI+ works better on standardised data)
    scaler     = StandardScaler()
    ret_scaled = scaler.fit_transform(ret_arr)
    mac_scaled = StandardScaler().fit_transform(mac_arr)

    data_arr   = np.column_stack([ret_scaled, mac_scaled])
    n_obs      = len(data_arr)

    if n_obs < cfg.MIN_WINDOW:
        raise ValueError(f"Insufficient data: {n_obs} obs < {cfg.MIN_WINDOW} minimum")

    if TIGRAMITE_AVAILABLE:
        method = "pcmci+"
        try:
            dataframe = pp.DataFrame(
                data_arr,
                var_names=all_vars,
                datatime=np.arange(n_obs),
            )
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(significance="analytic"),
                verbosity=0,
            )
            results = pcmci.run_pcmciplus(
                tau_min=1,
                tau_max=cfg.MAX_LAG,
                pc_alpha=cfg.PC_ALPHA,
            )

            # Extract ETF-only submatrix (ignore macro→macro links)
            val_full = results["val_matrix"]         # (n_vars, n_vars, max_lag+1)
            p_full   = results["p_matrix"]

            # Aggregate across lags: max absolute value
            val_etf = np.max(np.abs(val_full[:n_etfs, :n_etfs, 1:]), axis=2)
            p_etf   = np.min(p_full[:n_etfs, :n_etfs, 1:], axis=2)

            # Zero out non-significant links
            val_etf[p_etf > cfg.PC_ALPHA] = 0.0

        except Exception as e:
            print(f"[causal] PCMCI+ failed ({e}) — falling back to Granger")
            method  = "granger"
            val_etf = _granger_matrix(ret_scaled, cfg.MAX_LAG, cfg.PC_ALPHA)
            p_etf   = np.zeros_like(val_etf)

    else:
        method  = "granger"
        val_etf = _granger_matrix(ret_scaled, cfg.MAX_LAG, cfg.PC_ALPHA)
        p_etf   = np.zeros_like(val_etf)

    return {
        "causal_matrix": val_etf,
        "p_values":      p_etf,
        "tickers":       list(tickers),
        "n_obs":         n_obs,
        "method":        method,
    }


# ── Centrality scoring ─────────────────────────────────────────────────────────

def compute_centrality(causal_result: dict) -> pd.Series:
    """
    Score each ETF by its causal centrality in the network.

    out_degree  = sum of outgoing causal link strengths (causal DRIVERS)
    in_degree   = sum of incoming causal link strengths (causal FOLLOWERS)
    net_degree  = out - in (pure leaders vs pure followers)

    Returns pd.Series indexed by ticker, sorted descending.
    """
    mat     = causal_result["causal_matrix"]   # (n, n) where [i,j] = i causes j
    tickers = causal_result["tickers"]
    n       = len(tickers)

    out_deg = np.zeros(n)
    in_deg  = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] > cfg.MIN_LINK_STRENGTH:
                out_deg[i] += mat[i, j]   # i → j: i is a driver
                in_deg[j]  += mat[i, j]   # i → j: j is a follower

    if cfg.CENTRALITY_METRIC == "out_degree":
        scores = out_deg
    elif cfg.CENTRALITY_METRIC == "in_degree":
        scores = in_deg
    else:  # net_degree
        scores = out_deg - in_deg

    # Normalise to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    return pd.Series(scores, index=tickers).sort_values(ascending=False)


# ── Momentum overlay ───────────────────────────────────────────────────────────

def compute_momentum(returns: pd.DataFrame, tickers: list) -> pd.Series:
    """
    Cross-sectional momentum score for each ETF.
    Blends 5d, 21d, 63d returns, rank-normalised to [0,1].
    """
    scores = pd.Series(0.0, index=tickers)
    weights = [0.5, 0.3, 0.2]   # 5d, 21d, 63d

    for window, w in zip(cfg.MOMENTUM_WINDOWS, weights):
        if len(returns) < window:
            continue
        ret_w = returns[tickers].iloc[-window:].sum()
        rank  = ret_w.rank(pct=True)
        scores += w * rank

    # Normalise
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    return scores.sort_values(ascending=False)


# ── Combined signal ────────────────────────────────────────────────────────────

def build_signal_scores(
    causal_result: dict,
    returns: pd.DataFrame,
    tickers: list,
) -> pd.Series:
    """
    Blend causal centrality (70%) + momentum (30%) into final score.
    Returns pd.Series indexed by ticker, sorted descending.
    """
    centrality = compute_centrality(causal_result)
    momentum   = compute_momentum(returns, tickers)

    w_causal = 1.0 - cfg.MOMENTUM_WEIGHT
    w_mom    = cfg.MOMENTUM_WEIGHT

    combined = (w_causal * centrality + w_mom * momentum).reindex(tickers)
    combined = combined.fillna(0.0)

    # Normalise
    if combined.max() > combined.min():
        combined = (combined - combined.min()) / (combined.max() - combined.min())

    return combined.sort_values(ascending=False)


# ── Window-based analysis ──────────────────────────────────────────────────────

def run_window_analysis(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list,
    window_start: str,
    window_end: str = None,
) -> dict:
    end = window_end or cfg.TRAIN_END
    ret_w = returns[(returns.index >= window_start) & (returns.index <= end)]
    mac_w = macro[(macro.index >= window_start) & (macro.index <= end)]

    if len(ret_w) < cfg.MIN_WINDOW:
        return None

    causal_result = run_pcmci(ret_w, mac_w, tickers)
    centrality    = compute_centrality(causal_result)

    # Use window-sliced returns for momentum too — fixes length mismatch
    scores = build_signal_scores(causal_result, ret_w, tickers)

    return {
        "window_start":   window_start,
        "window_end":     end,
        "n_obs":          causal_result["n_obs"],
        "method":         causal_result["method"],
        "causal_matrix":  causal_result["causal_matrix"].tolist(),
        "centrality":     centrality.to_dict(),
        "scores":         scores.to_dict(),
        "top_pick":       scores.index[0],
        "conviction":     float(scores.iloc[0] - scores.iloc[1])
                          if len(scores) > 1 else float(scores.iloc[0]),
    }
