"""
backtest.py — Historische Simulation: statisches vs. ML-Grid-Spacing.

Das ist das Kernstueck der Arbeit. Wir simulieren den Grid-Bot auf
Tagesdaten (OHLCV) und vergleichen ein STATISCHES Spacing (0.5%) mit dem
ML-vorhergesagten Spacing. Beide laufen durch exakt dieselbe Grid- und
Filter-Logik, damit der Vergleich fair ist.

FILL-LOGIK (kritisch, Tagesdaten):
  - Low  <= Grid-Level  -> Buy-Order gefuellt
  - High >= Grid-Level  -> Sell-Order gefuellt
  - Beruehrt eine Kerze BEIDE: Annahme "erst Low (Buy), dann High (Sell)".
    Das ist ein worst-case-nahes, konservatives Szenario.
  - Niemals nur der Close fuer Fills (das unterschaetzt Trades + Gebuehren).

BEKANNTE LIMITATION (im Bericht erwaehnen):
  Mit Tagesdaten ist die Reihenfolge von High und Low innerhalb einer
  Kerze unbekannt. Bei sehr volatilen Kerzen (beide Level beruehrt) ist
  "erst Low, dann High" eine Naeherung — ein inhaerentes Limit von
  Tages-Backtests.
"""

import numpy as np
import pandas as pd

import config
import sma_filter
import ml_spacing
import grid_logic


START_CAPITAL = 10000.0   # Startkapital in USD fuer die Simulation

# Die gesamte Grid-Mechanik (Level, Orders, Fills, Rebuild) lebt jetzt in
# grid_logic.py und wird sowohl vom Backtest als auch spaeter vom Live-Bot
# genutzt. So gibt es nur EINE Quelle der Wahrheit fuer die Grid-Logik.


def annualize(value, num_days):
    """Rechnet einen Wert ueber 'num_days' auf ein Jahr (365 Tage) hoch."""
    if num_days <= 0:
        return 0.0
    return value * 365.0 / num_days


def run_backtest(df, mode="static", model=None, feature_df=None):
    """
    Fuehrt den Grid-Backtest ueber den uebergebenen DataFrame aus.

    Input:  df mit OHLCV + 'sma' (per sma_filter.add_sma), mode "static"
            oder "ml", model (fuer ML), feature_df (vorberechnete Features)
    Output: dict mit allen Pflicht-Metriken + Equity-Serie + Regime-Infos.

    Ablauf pro Tag: erst Fills (falls aktiv), dann SMA-Signal am Close
    auswerten (Entry/Exit/Stop), bei ML zusaetzlich taeglicher Rebuild-Check.
    """
    state = grid_logic.new_state(START_CAPITAL)

    stop_events = 0
    rebuilds = 0
    days_active = 0
    equity_curve = []
    dates = []

    n = len(df)
    for i in range(n):
        close = df["close"].iloc[i]
        low = df["low"].iloc[i]
        high = df["high"].iloc[i]
        sma = df["sma"].iloc[i]

        # 1) Falls aktiv: Tages-Fills abarbeiten.
        if state["active"]:
            grid_logic.simulate_day_fills(state, low, high)
            days_active += 1

        # 2) ML-Rebuild-Check (nur wenn aktiv und Modell vorhanden).
        if state["active"] and mode == "ml":
            feats = feature_df.iloc[i].to_dict() if feature_df is not None else None
            if feats is not None and not any(pd.isna(v) for v in feats.values()):
                new_spacing = ml_spacing.predict_spacing(model, feats)
                if ml_spacing.should_rebuild(state["spacing"], new_spacing) \
                        and not pd.isna(sma):
                    grid_logic.rebuild_grid(state, close, sma, new_spacing)
                    rebuilds += 1

        # 3) SMA-Signal am Tagesschluss auswerten.
        status = sma_filter.trend_status(df, i, state["active"])

        if state["active"] and not status["active"]:
            # Exit oder Stop-Loss -> Grid aufloesen.
            grid_logic.close_grid(state, close)
            if status["stop_loss_hit"]:
                stop_events += 1
        elif (not state["active"]) and status["active"]:
            # Entry bestaetigt -> Grid aufbauen.
            if mode == "ml":
                feats = feature_df.iloc[i].to_dict() if feature_df is not None else None
                if feats is not None and not any(pd.isna(v) for v in feats.values()):
                    spacing = ml_spacing.predict_spacing(model, feats)
                else:
                    spacing = config.GRID_SPACING_PCT
            else:
                spacing = config.GRID_SPACING_PCT
            if not pd.isna(sma):
                grid_logic.open_grid(state, close, sma, spacing)

        # 4) Equity am Tagesende (Cash + Inventarwert).
        equity = grid_logic.grid_equity(state, close)
        equity_curve.append(equity)
        dates.append(df.index[i])

    equity = pd.Series(equity_curve, index=dates)
    return summarize(equity, state, stop_events, rebuilds, days_active, n)


def summarize(equity, state, stop_events, rebuilds, days_active, num_days):
    """
    Berechnet aus Equity-Serie + Endzustand alle Pflicht-Metriken.

    Input:  equity (pd.Series), state (Endzustand), Zaehler, num_days
    Output: dict mit Kennzahlen + Equity-Serie.
    """
    start_eq = equity.iloc[0] if len(equity) else START_CAPITAL
    end_eq = equity.iloc[-1] if len(equity) else START_CAPITAL
    total_profit = end_eq - START_CAPITAL

    # Max-Drawdown (absolut + prozentual) aus der Equity-Kurve.
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd_abs = drawdown.min() if len(equity) else 0.0
    dd_pct_series = drawdown / running_max
    max_dd_pct = dd_pct_series.min() if len(equity) else 0.0

    # Sharpe (annualisiert, risikofreier Zins = 0) aus Tagesrenditen.
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    # CAGR (echte, zinseszins-basierte Jahresrendite). Aussagekraeftiger als
    # die lineare Hochrechnung, vor allem ueber mehrere Jahre.
    if num_days > 0 and end_eq > 0:
        cagr = (end_eq / START_CAPITAL) ** (365.0 / num_days) - 1.0
    else:
        cagr = 0.0

    return {
        "grid_profit": state["realized"],
        "grid_profit_annual": annualize(state["realized"], num_days),
        # Profit-Metriken zusaetzlich in % p.a. (bezogen auf Startkapital).
        "grid_profit_pct_annual": annualize(state["realized"], num_days) / START_CAPITAL,
        "total_profit": total_profit,
        "total_profit_annual": annualize(total_profit, num_days),
        "total_profit_pct_annual": annualize(total_profit, num_days) / START_CAPITAL,
        "total_return_pct": total_profit / START_CAPITAL,
        "cagr": cagr,
        "max_dd_abs": max_dd_abs,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "trades": state["trades"],
        "fees": state["fees"],
        "stop_events": stop_events,
        "rebuilds": rebuilds,
        "days_active": days_active,
        "days_total": num_days,
        "end_equity": end_eq,
        "equity": equity,
    }


def run_buy_hold(df):
    """
    Buy & Hold Benchmark: am ersten Tag zum Open kaufen, bis zum letzten
    Tag halten.

    Input:  df mit OHLC und DatetimeIndex
    Output: dict im gleichen Format wie summarize() (zum Tabellen-Vergleich).

    Fairness: wir buchen je eine Taker-Gebuehr beim Kauf (Open Tag 1) und
    beim fiktiven Verkauf (Close letzter Tag), damit der Benchmark
    denselben Gebuehren-Massstab wie der Bot hat.
    """
    n = len(df)
    if n == 0:
        return summarize(pd.Series([START_CAPITAL]), _empty_state(), 0, 0, 0, 0)

    entry_price = df["open"].iloc[0]
    buy_fee = START_CAPITAL * config.TAKER_FEE
    qty = (START_CAPITAL - buy_fee) / entry_price   # Kauf abzgl. Gebuehr

    # Tagesequity = Wert der gehaltenen BTC zum jeweiligen Close.
    equity = (qty * df["close"]).copy()

    # Fiktiver Verkauf am Ende: Schlussgebuehr vom Brutto-Endwert abziehen.
    sell_gross = equity.iloc[-1]
    sell_fee = sell_gross * config.TAKER_FEE
    equity.iloc[-1] = sell_gross - sell_fee

    state = _empty_state()
    state["trades"] = 1                       # genau ein Kauf
    state["fees"] = buy_fee + sell_fee
    return summarize(equity, state, stop_events=0, rebuilds=0,
                     days_active=n, num_days=n)


def _empty_state():
    """Leerer Zustands-dict (Hilfsfunktion fuer den Buy & Hold Benchmark)."""
    return {
        "realized": 0.0, "trades": 0, "fees": 0.0,
    }


def print_metrics(name, m):
    """Druckt die Kennzahlen eines Backtest-Laufs lesbar untereinander."""
    print(f"\n--- {name} ---")
    print(f"  Grid-Profit (Periode):     {m['grid_profit']:>12,.2f} USD")
    print(f"  Grid-Profit p.a.:          {m['grid_profit_annual']:>12,.2f} USD "
          f"({m['grid_profit_pct_annual']:.2%} p.a.)")
    print(f"  Gesamtprofit (Periode):    {m['total_profit']:>12,.2f} USD "
          f"({m['total_return_pct']:.2%})")
    print(f"  Gesamtprofit p.a.:         {m['total_profit_annual']:>12,.2f} USD "
          f"({m['total_profit_pct_annual']:.2%} p.a.)")
    print(f"  Rendite p.a. (CAGR):       {m['cagr']:>11.2%}")
    print(f"  Max-Drawdown:              {m['max_dd_abs']:>12,.2f} USD "
          f"({m['max_dd_pct']:.2%})")
    print(f"  Sharpe (annualisiert):     {m['sharpe']:>12.2f}")
    print(f"  Abgeschlossene Trades:     {m['trades']:>12d}")
    print(f"  Gebuehren gesamt:          {m['fees']:>12,.2f} USD")
    print(f"  Stop-Loss-Events:          {m['stop_events']:>12d}")
    print(f"  Rebuilds (nur ML):         {m['rebuilds']:>12d}")
    print(f"  Tage aktiv / gesamt:       {m['days_active']:>6d} / {m['days_total']}")


def print_table(m_static, m_ml, m_bh, title=""):
    """
    Druckt die Ergebnistabelle mit drei Spalten: Statisch / ML / Buy&Hold.

    Input:  drei Metrik-dicts (aus summarize / run_buy_hold), optionaler Titel
    Output: nichts.
    """
    if title:
        print(f"\n{title}")
    rows = [
        ("Gesamtprofit p.a. (USD)", "total_profit_annual", "{:>12,.2f}"),
        ("Gesamtprofit p.a. (%)",    "total_profit_pct_annual", "{:>11.2%}"),
        ("Rendite p.a. CAGR (%)",    "cagr", "{:>11.2%}"),
        ("Gesamtrendite Periode (%)", "total_return_pct", "{:>11.2%}"),
        ("Grid-Profit p.a. (USD)",  "grid_profit_annual", "{:>12,.2f}"),
        ("Grid-Profit p.a. (%)",     "grid_profit_pct_annual", "{:>11.2%}"),
        ("Max-Drawdown (%)",         "max_dd_pct", "{:>11.2%}"),
        ("Max-Drawdown (USD)",      "max_dd_abs", "{:>12,.2f}"),
        ("Sharpe (annualisiert)",    "sharpe", "{:>12.2f}"),
        ("Abgeschlossene Trades",    "trades", "{:>12d}"),
        ("Gebuehren gesamt (USD)",  "fees", "{:>12,.2f}"),
        ("Stop-Loss-Events",         "stop_events", "{:>12d}"),
    ]

    name_w = 28
    print(f"\n  {'Metrik':<{name_w}} {'Statisch':>13} {'ML':>13} {'Buy&Hold':>13}")
    print(f"  {'-' * (name_w + 3 * 14)}")
    for label, key, fmt in rows:
        s = fmt.format(m_static[key])
        ml = fmt.format(m_ml[key])
        bh = fmt.format(m_bh[key])
        print(f"  {label:<{name_w}} {s:>13} {ml:>13} {bh:>13}")


def compare_period(df_full, model, feature_df, start, end, label):
    """
    Vergleicht statisch vs. ML vs. Buy&Hold fuer einen Zeitraum.

    Input:  df_full (mit sma), model, feature_df, start/end (Datum), label
    Output: (metrics_static, metrics_ml, metrics_buy_hold)
    """
    mask = (df_full.index >= start) & (df_full.index <= end)
    df = df_full[mask]
    feats = feature_df[mask]

    m_static = run_backtest(df, mode="static")
    m_ml = run_backtest(df, mode="ml", model=model, feature_df=feats)
    m_bh = run_buy_hold(df)

    print_table(m_static, m_ml, m_bh,
                title=f"========== {label} ({start} bis {end}) ==========")

    # Direkter Mehrwert-Vergleich (Hauptmetrik: Grid-Profit p.a.).
    diff = m_ml["grid_profit_annual"] - m_static["grid_profit_annual"]
    print(f"\n  >> ML-Mehrwert Grid-Profit p.a.: {diff:+,.2f} USD")

    # Regime-Analyse: Anteil aktiver (Bull) vs. pausierter (Baer) Tage.
    share_static = m_static["days_active"] / max(m_static["days_total"], 1)
    print(f"  >> Bot aktiv (Bull-Regime): {share_static:.1%} der Tage, "
          f"pausiert (Baer): {1 - share_static:.1%}")
    return m_static, m_ml, m_bh


def plot_equity(curves, filename="equity_curve.png"):
    """
    Speichert eine einfache Equity-Kurve (optional, nur wenn matplotlib da).

    Input:  curves = dict {label: pd.Series}, filename
    Output: nichts (Datei wird geschrieben).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib nicht verfuegbar - Plot uebersprungen.")
        return

    plt.figure(figsize=(10, 5))
    for label, series in curves.items():
        plt.plot(series.index, series.values, label=label)
    plt.title("Equity-Kurve: statisch vs. ML-Spacing vs. Buy&Hold")
    plt.xlabel("Datum")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Equity-Kurve gespeichert: {filename}")


if __name__ == "__main__":
    print("=== backtest.py Test ===")
    import data

    df = data.load_data()
    df = sma_filter.add_sma(df)
    feature_df = ml_spacing.compute_features(df)[ml_spacing.FEATURE_COLUMNS]
    model = ml_spacing.load_model()
    if model is None:
        print("WARN: kein Modell geladen -> ML faellt auf statisches Spacing zurueck.")

    # --- Teil 1: Mini-Backtest auf 90 Tagen (Pflicht-Test) ---------------
    print("\n##### MINI-BACKTEST (letzte 90 Tage) #####")
    mini = df.iloc[-90:]
    mini_feats = feature_df.iloc[-90:]
    m_mini_static = run_backtest(mini, mode="static")
    m_mini_ml = run_backtest(mini, mode="ml", model=model, feature_df=mini_feats)
    print_metrics("MINI statisch", m_mini_static)
    print_metrics("MINI ML", m_mini_ml)

    # Sanity-Checks auf dem Mini-Lauf.
    assert m_mini_static["days_total"] == 90, "Mini-Backtest sollte 90 Tage haben"
    assert m_mini_static["fees"] >= 0, "Gebuehren duerfen nicht negativ sein"
    assert m_mini_static["end_equity"] > 0, "Equity muss positiv bleiben"
    print("\nOK  Mini-Backtest Sanity-Checks bestanden.")

    # --- Teil 2: Voller Vergleich In-Sample vs. Out-of-Sample ------------
    m_is_s, m_is_ml, m_is_bh = compare_period(
        df, model, feature_df,
        config.IN_SAMPLE_START, config.IN_SAMPLE_END, "IN-SAMPLE")
    m_os_s, m_os_ml, m_os_bh = compare_period(
        df, model, feature_df,
        config.OUT_SAMPLE_START, config.OUT_SAMPLE_END,
        "OUT-OF-SAMPLE (entscheidend)")

    # --- Teil 3: Equity-Kurve (optional) ---------------------------------
    plot_equity({
        "OOS statisch": m_os_s["equity"],
        "OOS ML": m_os_ml["equity"],
        "OOS Buy&Hold": m_os_bh["equity"],
    })

    print("\nAlle backtest-Laeufe abgeschlossen.")
