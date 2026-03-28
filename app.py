# app.py — P2-ETF-PCMCI-ENGINE Streamlit Dashboard
# Causal Discovery ETF Signal Engine
# Two tabs: Option A (FI) | Option B (Equity)

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
import pandas_market_calendars as mcal

import config as cfg

st.set_page_config(
    page_title="PCMCI+ — Causal ETF Engine",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background-color: #ffffff; }

  .hero-card {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 14px; padding: 28px 32px 22px 32px; margin-bottom: 24px;
  }
  .hero-ticker { font-size: 64px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
  .hero-conv   { font-size: 26px; font-weight: 500; color: #16a34a; margin-top: 6px; }
  .hero-date   { font-size: 15px; color: #6b7280; margin-top: 8px; }
  .hero-source { font-size: 14px; color: #15803d; font-weight: 600;
                 background: #dcfce7; border-radius: 20px;
                 padding: 3px 12px; display: inline-block; margin-top: 8px; }
  .runner-up   { font-size: 18px; color: #374151; margin-top: 14px;
                 padding-top: 14px; border-top: 1px solid #e5e7eb; }

  .label-fixed  { display:inline-block; font-size:14px; font-weight:700;
                  color:#374151; text-transform:uppercase; letter-spacing:.07em;
                  background:#f3f4f6; border-radius:6px;
                  padding:5px 14px; margin-bottom:12px; }
  .label-window { display:inline-block; font-size:14px; font-weight:700;
                  color:#15803d; text-transform:uppercase; letter-spacing:.07em;
                  background:#dcfce7; border-radius:6px;
                  padding:5px 14px; margin-bottom:12px; }
  .window-badge { font-size:14px; color:#15803d; background:#dcfce7;
                  border:1px solid #86efac; border-radius:20px;
                  padding:4px 14px; display:inline-block; margin-bottom:12px; }
  .period-badge { font-size:14px; color:#374151; background:#f3f4f6;
                  border:1px solid #e5e7eb; border-radius:20px;
                  padding:4px 14px; display:inline-block; margin-bottom:12px; }

  .metric-row { display:flex; gap:12px; margin:10px 0 16px 0; }
  .metric-box { flex:1; background:#fff; border:1px solid #e5e7eb;
                border-radius:10px; padding:14px 10px; text-align:center; }
  .metric-label { font-size:12px; color:#6b7280; text-transform:uppercase;
                  letter-spacing:.05em; margin-bottom:6px; }
  .metric-value { font-size:24px; font-weight:600; color:#111827; }
  .pos { color:#059669; } .neg { color:#dc2626; }

  .causal-link { display:flex; align-items:center; justify-content:space-between;
                 padding:8px 12px; border-radius:8px; margin-bottom:6px;
                 background:#f8faff; border:1px solid #e0e7ff; font-size:14px; }
  .causal-from { font-weight:700; color:#1d4ed8; }
  .causal-to   { font-weight:700; color:#16a34a; }
  .causal-str  { color:#6b7280; font-size:13px; }

  .hit-line { font-size:16px; color:#374151; margin-bottom:10px; }
  .fn       { font-size:13px; color:#9ca3af; margin-top:8px; }
  .sec-hdr  { font-size:20px; font-weight:700; color:#1a1a2e; margin:28px 0 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helper: next trading day (with NYSE calendar) ─────────────────────────────

nyse = mcal.get_calendar("NYSE")

def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next NYSE trading day after the given date."""
    schedule = nyse.schedule(start_date=date, end_date=date + timedelta(days=10))
    trading_days = schedule.index
    next_days = trading_days[trading_days > date]
    if len(next_days) > 0:
        return next_days[0]
    # fallback: skip weekends only
    d = date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_signals() -> dict:
    try:
        import requests
        url = f"https://huggingface.co/datasets/{cfg.HF_RESULTS_REPO}/resolve/main/results/latest_signals.json"
        headers = {"Authorization": f"Bearer {cfg.HF_TOKEN}"} if cfg.HF_TOKEN else {}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        raw = r.json()
        return {
            "A": raw.get("option_A") or {},
            "B": raw.get("option_B") or {},
        }
    except Exception as e:
        st.error(f"Could not load signals: {e}")
        return {"A": {}, "B": {}}


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=cfg.FILE_MASTER,
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error(f"Could not load master dataset: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_history(option: str) -> pd.DataFrame:
    try:
        import requests
        url = (f"https://huggingface.co/datasets/{cfg.HF_RESULTS_REPO}"
               f"/resolve/main/results/signal_history_{option}.json")
        headers = {"Authorization": f"Bearer {cfg.HF_TOKEN}"} if cfg.HF_TOKEN else {}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


# ── Backtest ───────────────────────────────────────────────────────────────────

def build_bt(pick: str, master: pd.DataFrame, option: str,
             start_date: str = None) -> dict:
    if not pick or master.empty:
        return {}

    benchmark    = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK
    period_start = start_date or cfg.LIVE_START
    oos = master[master.index >= period_start].copy()
    if oos.empty:
        return {}

    bench_ret = oos.get(f"{benchmark}_ret",
                        pd.Series(0.0, index=oos.index)).fillna(0.0)
    ret_col   = f"{pick}_ret"
    pick_rets = oos.get(ret_col, pd.Series(0.0, index=oos.index)).fillna(0.0)

    sc = (1 + pick_rets).cumprod()
    bc = (1 + bench_ret).cumprod()

    def ar(r): return float(r.mean() * 252)
    def av(r): return float(r.std() * np.sqrt(252))
    def sh(r): return ar(r) / (av(r) + 1e-8)
    def dd(c): return float(((c - c.cummax()) / c.cummax()).min())
    def hr(r): return float((r > 0).mean())

    return {
        "dates": oos.index, "sc": sc, "bc": bc,
        "pick": pick, "benchmark": benchmark,
        "m": {"ar": ar(pick_rets), "av": av(pick_rets),
              "sh": sh(pick_rets), "dd": dd(sc), "hr": hr(pick_rets)},
    }


# ── UI components ──────────────────────────────────────────────────────────────

def render_hero(signal: dict, option: str, master: pd.DataFrame):
    if not signal or "pick" not in signal:
        st.info("Signal not available yet — run the workflow first.")
        return

    tickers = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    scores  = signal.get("scores", {})
    sorted_picks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    pick       = signal["pick"]
    conviction = signal.get("conviction", 0)
    source     = signal.get("source", "—")
    gen        = signal.get("generated_at", "")
    method     = signal.get("causal_method", "pcmci+")

    # Compute correct next trading day from the master dataset's last date
    if not master.empty:
        last_data_date = master.index[-1]
        next_day = next_trading_day(last_data_date)
        sig_date = next_day.strftime("%Y-%m-%d")
    else:
        # fallback: use signal's last_data_date if master not available
        last = signal.get("last_data_date")
        if last:
            try:
                last_date = pd.Timestamp(last)
                next_day = next_trading_day(last_date)
                sig_date = next_day.strftime("%Y-%m-%d")
            except Exception:
                sig_date = last
        else:
            sig_date = "—"

    try:
        gen = datetime.fromisoformat(gen).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    t2 = sorted_picks[1] if len(sorted_picks) > 1 else None
    t3 = sorted_picks[2] if len(sorted_picks) > 2 else None
    runner = ""
    if t2: runner += f"<span style='color:#6b7280'>2nd:</span> <b>{t2[0]}</b> {t2[1]:.3f}"
    if t3: runner += f"&nbsp;&nbsp;<span style='color:#6b7280'>3rd:</span> <b>{t3[0]}</b> {t3[1]:.3f}"

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-ticker">{pick}</div>
      <div class="hero-conv">Causal score: {conviction:.3f}</div>
      <div class="hero-date">Signal for {sig_date} &nbsp;·&nbsp; Generated {gen}
        &nbsp;·&nbsp; Method: {method}</div>
      <div class="hero-source">Source: {source}</div>
      <div class="runner-up">{runner}</div>
    </div>
    """, unsafe_allow_html=True)


def render_causal_links(signal: dict):
    links = signal.get("top_causal_links", [])
    if not links:
        st.caption("No causal links detected.")
        return

    st.markdown("<div class='sec-hdr' style='font-size:16px;margin:0 0 10px 0;'>"
                "Top causal links — who drives whom</div>",
                unsafe_allow_html=True)

    for link in links[:8]:
        strength = link["strength"]
        bar_w    = int(strength * 100)
        st.markdown(f"""
        <div class="causal-link">
          <span class="causal-from">{link['from']}</span>
          <span style="color:#9ca3af;font-size:16px;">→</span>
          <span class="causal-to">{link['to']}</span>
          <div style="flex:1;margin:0 12px;background:#e5e7eb;
                      border-radius:4px;height:8px;">
            <div style="width:{bar_w}%;background:#3a5bd9;
                        border-radius:4px;height:8px;"></div>
          </div>
          <span class="causal-str">{strength:.3f}</span>
        </div>
        """, unsafe_allow_html=True)


def render_metrics(bt: dict):
    if not bt:
        st.caption("Metrics available after first run.")
        return
    m  = bt["m"]
    fp = lambda v: f"{v*100:.1f}%"
    c  = lambda v: "pos" if v >= 0 else "neg"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Ann Return</div>
        <div class="metric-value {c(m['ar'])}">{fp(m['ar'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Ann Vol</div>
        <div class="metric-value">{fp(m['av'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Sharpe</div>
        <div class="metric-value {c(m['sh'])}">{m['sh']:.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max DD (peak→trough)</div>
        <div class="metric-value neg">{fp(m['dd'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Hit Rate</div>
        <div class="metric-value">{fp(m['hr'])}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_curve(bt: dict, key: str = ""):
    if not bt:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["sc"].values,
        name=f"PCMCI+ ({bt['pick']})",
        line=dict(color="#16a34a", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["bc"].values,
        name=bt["benchmark"],
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=13)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=12),
                   tickformat=".2f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=f"curve_{key}")


def render_history(hist_df: pd.DataFrame, master: pd.DataFrame):
    if hist_df.empty:
        st.info("Signal history will appear after the first run.")
        return

    # Recalculate actual_return if not present or if last_data_date exists
    if not master.empty:
        def get_ret(row):
            try:
                pick = row.get("pick", "")
                if not pick:
                    return np.nan

                col = f"{pick}_ret"
                if col not in master.columns:
                    return np.nan

                # Use last_data_date for lookup if available (preferred)
                # Otherwise fall back to signal_date
                date_str = row.get("last_data_date") or row.get("signal_date")
                if not date_str:
                    return np.nan

                date = pd.Timestamp(date_str)
                if date in master.index:
                    return master.loc[date, col]
            except Exception:
                pass
            return np.nan

        # Only recalculate if actual_return is missing for any row
        if "actual_return" not in hist_df.columns or hist_df["actual_return"].isna().any():
            hist_df["actual_return"] = hist_df.apply(get_ret, axis=1)

    if "hit" not in hist_df.columns and "actual_return" in hist_df.columns:
        hist_df["hit"] = hist_df["actual_return"].apply(
            lambda x: "✓" if (not np.isnan(x) and x > 0)
                      else ("✗" if not np.isnan(x) else "—")
        )

    disp = hist_df.sort_values("signal_date", ascending=False).copy()
    col_map = {
        "signal_date":   "Date",
        "pick":          "Pick",
        "conviction":    "Causal Score",
        "source":        "Source",
        "actual_return": "Actual Return",
        "hit":           "Hit",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    if "Causal Score" in disp.columns:
        disp["Causal Score"] = disp["Causal Score"].apply(lambda x: f"{x:.3f}")
    if "Actual Return" in disp.columns:
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "—"
        )

    if "Hit" in disp.columns:
        hits  = (disp["Hit"] == "✓").sum()
        total = disp["Hit"].isin(["✓", "✗"]).sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div class='hit-line'>Hit rate: <b>{hr:.1%}</b>"
            f" &nbsp;({hits}/{total} signals)</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Option renderer ────────────────────────────────────────────────────────────

def render_option(option: str, signals: dict, master: pd.DataFrame):
    signal = signals.get(option, {})
    hist   = load_history(option)

    # Hero (pass master for date calculation)
    render_hero(signal, option, master)

    # Causal links — full width
    if signal.get("top_causal_links"):
        render_causal_links(signal)
        st.markdown("---")

    # Fixed window uses its actual test period (last 15% of full history)
    # Shrinking window uses fixed OOS (2025-01-01 → today)
    fw  = signal.get("fixed_window", {})
    sw  = signal.get("shrinking_window", {})

    n_total    = len(master) if not master.empty else 4582
    n_test     = int(n_total * 0.15)
    test_start = fw.get("test_start") or \
                 (str(master.index[-n_test].date()) if not master.empty else "2023-01-01")
    oos_start  = cfg.LIVE_START
    oos_end    = str(master.index[-1].date()) if not master.empty else "today"

    bt_f = build_bt(fw.get("pick", ""), master, option, start_date=test_start)
    bt_w = build_bt(sw.get("pick", ""), master, option, start_date=oos_start)

    col_f, col_w = st.columns(2, gap="large")

    with col_f:
        st.markdown("<div class='label-fixed'>Fixed Window</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='period-badge'>Test: {test_start} → {oos_end}</div>",
            unsafe_allow_html=True,
        )
        # Use pre-computed metrics from signal JSON
        fw_bt = {
            "m": {
                "ar": fw.get("oos_return", 0),
                "av": fw.get("oos_vol", 0),
                "sh": fw.get("oos_sharpe", 0),
                "dd": fw.get("max_dd", 0),
                "hr": fw.get("hit_rate", 0),
            }
        }
        render_metrics(fw_bt)
        render_curve(bt_f, key=f"{option}_fixed")
        st.markdown(
            f"<div class='fn'>Pick: {fw.get('pick','—')} &nbsp;·&nbsp; "
            f"OOS Return: {fw.get('oos_return',0)*100:.2f}% &nbsp;·&nbsp; "
            f"Sharpe: {fw.get('oos_sharpe',0):.3f}</div>",
            unsafe_allow_html=True,
        )

    with col_w:
        st.markdown("<div class='label-window'>Shrinking Window</div>",
                    unsafe_allow_html=True)
        if sw.get("winning_window"):
            st.markdown(
                f"<div class='window-badge'>"
                f"Window {sw['winning_window']}: "
                f"{sw.get('winning_train_start','?')} → {sw.get('winning_train_end','?')}"
                f" &nbsp;·&nbsp; OOS: {oos_start} → {oos_end}"
                f"</div>",
                unsafe_allow_html=True,
            )
        # Use pre-computed metrics from signal JSON
        sw_bt = {
            "m": {
                "ar": sw.get("oos_return", 0),
                "av": sw.get("oos_vol", 0),
                "sh": sw.get("oos_sharpe", 0),
                "dd": sw.get("oos_max_dd", 0),
                "hr": sw.get("oos_hit_rate", 0),
            }
        }
        render_metrics(sw_bt)
        render_curve(bt_w, key=f"{option}_window")
        st.markdown(
            f"<div class='fn'>Pick: {sw.get('pick','—')} &nbsp;·&nbsp; "
            f"OOS Return: {sw.get('oos_return',0)*100:.2f}% &nbsp;·&nbsp; "
            f"Sharpe: {sw.get('oos_sharpe',0):.3f}</div>",
            unsafe_allow_html=True,
        )

    # Window comparison table
    all_windows = sw.get("all_windows", [])
    if all_windows:
        st.markdown("<div class='sec-hdr'>Window comparison</div>",
                    unsafe_allow_html=True)
        wdf = pd.DataFrame(all_windows)[
            ["window_id", "train_start", "pick", "oos_return", "oos_sharpe", "method"]
        ].rename(columns={
            "window_id":   "Window",
            "train_start": "Train from",
            "pick":        "Pick",
            "oos_return":  "OOS Return",
            "oos_sharpe":  "Sharpe",
            "method":      "Method",
        })
        wdf["OOS Return"] = wdf["OOS Return"].apply(lambda x: f"{x*100:.2f}%")
        wdf["Sharpe"]     = wdf["Sharpe"].apply(lambda x: f"{x:.3f}")
        st.dataframe(wdf, use_container_width=True, hide_index=True)

    # Signal history
    st.markdown("<div class='sec-hdr'>Signal History</div>",
                unsafe_allow_html=True)
    render_history(hist, master)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#1a1a2e;font-size:34px;'>"
        "PCMCI+ — Causal Discovery ETF Engine</h2>"
        "<p style='color:#6b7280;font-size:16px;margin-top:0;'>"
        "Who drives whom? &nbsp;·&nbsp; Causal centrality + momentum &nbsp;·&nbsp; "
        "8 shrinking windows</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading signals and data..."):
        signals = load_signals()
        master  = load_master()

    tab_a, tab_b = st.tabs([
        "🔗  Option A — Fixed Income / Alts",
        "🔗  Option B — Equity Sectors",
    ])

    with tab_a:
        render_option("A", signals, master)

    with tab_b:
        render_option("B", signals, master)

    st.markdown(
        "<div style='margin-top:40px;padding-top:16px;border-top:1px solid #e5e7eb;"
        "font-size:13px;color:#9ca3af;text-align:center;'>"
        "P2-ETF-PCMCI-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
