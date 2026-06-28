"""
paper_trade.py — Paper-Trading-Validierung des Spot-Grids OHNE Hebel.

Schritt 7: Bevor echtes Geld oder eine Live-API ins Spiel kommt, validieren
wir den kompletten Stack (data -> sma_filter -> ml_spacing -> grid_logic)
im Trockenlauf. Es werden KEINE echten Orders gesendet, es fliesst KEIN
Geld, und es wird KEIN Hebel benutzt (Spot, 1x).

Der taegliche Ablauf hier ist exakt die Orchestrierung, die monitor.py
spaeter live fahren wird:
    1. Daten/Signale fuer den Tag holen
    2. Wenn aktiv: Fills des Tages verarbeiten
    3. ML-Rebuild pruefen
    4. SMA-Signal: Grid oeffnen / schliessen (Entry/Exit/Stop)
    5. Zustand sichern (Restart-Sicherheit)

So sehen wir Tag fuer Tag, was der Bot tun WUERDE — und koennen die Logik
verifizieren, ohne Risiko.
"""

import json
import os

import pandas as pd

import config
import data
import sma_filter
import ml_spacing
import grid_logic


# Spot-Modus: ausdruecklich KEIN Hebel. Wir handeln nur eingesetztes Kapital.
PAPER_LEVERAGE = 1
PAPER_CAPITAL = 10000.0          # fiktives Startkapital (USDT)
PAPER_STATE_PATH = "paper_state.json"
DEFAULT_DAYS = 60                # wie viele letzte Tage durchgespielt werden


def save_paper_state(state):
    """
    Sichert den Laufzustand im state.json-Schema (Restart-Sicherheit).

    Input:  state (grid_logic-state dict)
    Output: nichts (schreibt PAPER_STATE_PATH).

    Wir speichern bewusst das gleiche, schlanke Schema, das spaeter live
    in state.json landet — so ist Schritt 9 nur noch ein kleiner Schritt.
    Doppelorders nach einem Neustart werden so verhindert.
    """
    snapshot = {
        "grid_active": state["active"],
        "grid_start_price": state["start_price"],
        "current_spacing": state["spacing"],
        "open_orders": (
            [{"side": "buy", "level": o["price"]} for o in state["buy_orders"].values()]
            + [{"side": "sell", "level": o["price"]} for o in state["sell_orders"].values()]
        ),
    }
    with open(PAPER_STATE_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)


def show_planned_orders(state, max_each=5):
    """
    Zeigt die Orders, die das Grid aktuell offen haette (nur Anzeige).

    Input:  state (grid_logic-state dict), max_each (wie viele je Seite)
    Output: nichts (print).
    """
    buys = sorted((o for o in state["buy_orders"].values()),
                  key=lambda o: o["price"], reverse=True)
    sells = sorted((o for o in state["sell_orders"].values()),
                   key=lambda o: o["price"])
    print(f"    Offene Buy-Orders:  {len(buys)} "
          f"(naechste: " + ", ".join(f"{o['price']:.0f}" for o in buys[:max_each]) + ")")
    print(f"    Offene Sell-Orders: {len(sells)} "
          f"(naechste: " + ", ".join(f"{o['price']:.0f}" for o in sells[:max_each]) + ")")


def paper_day(state, df, i, feature_df, model, verbose=True):
    """
    Fuehrt EINEN Tag des Live-Loops im Paper-Modus aus.

    Input:  state, df (mit 'sma'), Tagesindex i, feature_df, model
    Output: nichts (state wird veraendert, Ereignisse werden geprintet).

    Diese Funktion ist die wiederverwendbare Tages-Orchestrierung — die
    gleiche Logik nutzt monitor.py spaeter im echten Live-Betrieb.
    """
    day = df.index[i].date()
    close = df["close"].iloc[i]
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]
    sma = df["sma"].iloc[i]

    trades_before = state["trades"]

    # 1) Wenn aktiv: Fills des Tages verarbeiten (Spot-Limit-Orders).
    if state["active"]:
        grid_logic.simulate_day_fills(state, low, high)

    # 2) ML-Rebuild pruefen (nur wenn aktiv).
    rebuilt = False
    if state["active"] and model is not None:
        feats = feature_df.iloc[i].to_dict()
        if not any(pd.isna(v) for v in feats.values()):
            new_spacing = ml_spacing.predict_spacing(model, feats)
            if ml_spacing.should_rebuild(state["spacing"], new_spacing) \
                    and not pd.isna(sma):
                grid_logic.rebuild_grid(state, close, sma, new_spacing)
                rebuilt = True

    # 3) SMA-Signal am Tagesschluss.
    status = sma_filter.trend_status(df, i, state["active"])

    event = "halten"
    if state["active"] and not status["active"]:
        grid_logic.close_grid(state, close)
        event = "STOP-LOSS -> Grid geschlossen" if status["stop_loss_hit"] \
            else "EXIT -> Grid geschlossen"
    elif (not state["active"]) and status["active"]:
        # Entry: Spacing per ML (mit Fallback) bestimmen und Grid oeffnen.
        if model is not None:
            feats = feature_df.iloc[i].to_dict()
            spacing = ml_spacing.predict_spacing(model, feats) \
                if not any(pd.isna(v) for v in feats.values()) \
                else config.GRID_SPACING_PCT
        else:
            spacing = config.GRID_SPACING_PCT
        if not pd.isna(sma):
            grid_logic.open_grid(state, close, sma, spacing)
            event = f"ENTRY -> Grid geoeffnet (Spacing {spacing:.3%})"

    # 4) Zustand sichern (Restart-Sicherheit, state.json-Schema).
    save_paper_state(state)

    if verbose:
        fills = state["trades"] - trades_before
        equity = grid_logic.grid_equity(state, close)
        flags = []
        if fills:
            flags.append(f"{fills} Fills")
        if rebuilt:
            flags.append("Rebuild")
        extra = (" [" + ", ".join(flags) + "]") if flags else ""
        print(f"  {day}  Close {close:>9.0f}  SMA {sma:>9.0f}  "
              f"{'AKTIV ' if state['active'] else 'PAUSE '}  "
              f"Equity {equity:>10.2f}  {event}{extra}")


def run_paper(days=DEFAULT_DAYS, verbose=True):
    """
    Spielt die letzten 'days' Tage als Paper-Trade durch und fasst zusammen.

    Input:  days (Anzahl letzter Tage), verbose (Tagesausgabe an/aus)
    Output: das finale state-dict.
    """
    print(f"=== Paper-Trading (SPOT, Hebel {PAPER_LEVERAGE}x) ueber "
          f"{days} Tage ===")

    df = data.load_data()
    df = sma_filter.add_sma(df)
    feature_df = ml_spacing.compute_features(df)[ml_spacing.FEATURE_COLUMNS]

    model = ml_spacing.load_model()
    if model is None:
        print("WARN: kein ML-Modell -> Spacing faellt auf statischen Wert zurueck.")

    state = grid_logic.new_state(PAPER_CAPITAL)

    # WICHTIG: Der SMA-Filter braucht Historie VOR dem ersten Paper-Tag,
    # damit Entry/Exit bestaetigt werden koennen. Wir starten den Loop also
    # erst nach genug Vorlauf, spielen aber nur die letzten 'days' Tage.
    start = max(config.SMA_PERIOD + config.CONFIRMATION_CANDLES_ENTRY,
                len(df) - days)
    print(f"Zeitraum: {df.index[start].date()} bis {df.index[-1].date()}\n")

    for i in range(start, len(df)):
        paper_day(state, df, i, feature_df, model, verbose=verbose)

    last_close = df["close"].iloc[-1]
    equity = grid_logic.grid_equity(state, last_close)
    print("\n--- Paper-Trade Zusammenfassung ---")
    print(f"  Status am Ende:        {'AKTIV' if state['active'] else 'PAUSE'}")
    print(f"  Abgeschlossene Trades: {state['trades']}")
    print(f"  Realisierter Profit:   {state['realized']:.2f} USDT")
    print(f"  Gebuehren:             {state['fees']:.2f} USDT")
    print(f"  Cash / BTC:            {state['cash']:.2f} USDT / {state['btc']:.6f} BTC")
    print(f"  Equity (Spot):         {equity:.2f} USDT "
          f"({equity / PAPER_CAPITAL - 1:+.2%})")
    return state


if __name__ == "__main__":
    state = run_paper(days=DEFAULT_DAYS, verbose=True)

    # --- Validierung: aktuelle Live-Empfehlung (was wuerde der Bot HEUTE tun) ---
    df = data.load_data()
    df = sma_filter.add_sma(df)
    i = len(df) - 1
    status = sma_filter.trend_status(df, i, grid_active=state["active"])
    print("\n--- Aktuelle Empfehlung (letzter Tagesschluss) ---")
    print(f"  Datum:           {df.index[i].date()}")
    print(f"  Close:           {status['close']:.2f}")
    print(f"  SMA-120:         {status['sma']:.2f}")
    print(f"  Stop-Loss-Preis: {status['stop_loss_price']:.2f}")
    print(f"  Soll-Zustand:    {'GRID AKTIV (Long)' if status['active'] else 'PAUSE'}")

    if state["active"]:
        print("\n  Aktuell geplante Orders (Auszug):")
        show_planned_orders(state)

    # --- Sanity-Checks (kein Hebel, Invarianten, Zustand gesichert) --------
    assert PAPER_LEVERAGE == 1, "Spot-Paper-Trading darf keinen Hebel nutzen"
    assert state["cash"] >= 0, "Cash darf im Spot nie negativ werden"
    btc_sum = sum(o["qty"] for o in state["sell_orders"].values())
    assert abs(state["btc"] - btc_sum) < 1e-6 or not state["active"], \
        "Bei aktivem Grid muss BTC zu den Sell-Orders passen"
    assert os.path.exists(PAPER_STATE_PATH), "Zustand muss gesichert sein"
    print("\nOK  Paper-Trade Sanity-Checks bestanden (Spot, kein Hebel).")
