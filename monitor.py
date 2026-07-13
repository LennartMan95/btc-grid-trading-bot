"""
monitor.py — Taegliche Orchestrierung (EIN Durchlauf pro Cronjob-Aufruf).

Wird einmal pro Tag kurz nach UTC-Tagesabschluss aufgerufen (kein while-True).
Ablauf:
  1. Tageskerzen von Alpaca laden
  2. SMA-Filter (LONG / PAUSE)
  3. ML-Spacing + Rebuild pruefen
  4. Fills von Alpaca erkennen, Gegenorders setzen (execution.py)
  5. Grid oeffnen / schliessen / neu aufbauen (grid_logic.py)
  6. state.json nach jeder Aenderung sichern

Paper-Modus only (ALPACA_PAPER=True), Spot-only, kein Hebel.
"""

import argparse

import pandas as pd
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderStatus

import config
import data
import sma_filter
import ml_spacing
import grid_logic
import execution


def _sma_status(active):
    """Gibt 'LONG' oder 'PAUSE' fuer state.json zurueck."""
    return "LONG" if active else "PAUSE"


def _persist(grid_state, sma_status, bar_date, logger):
    """Speichert grid_state sofort in state.json."""
    snap = data.snapshot_from_grid(grid_state, sma_status, bar_date)
    data.save_state(snap)
    logger.info("state.json gesichert (Status=%s, Spacing=%.3f%%).",
                sma_status, grid_state["spacing"] * 100)


def sync_account(grid_state, client, logger):
    """
    Holt Cash, BTC und Equity vom Alpaca-Paper-Konto.

    Input:  grid_state (wird aktualisiert), client, logger
    Output: nichts.
    """
    acc = client.get_account()
    grid_state["capital"] = float(acc.equity)
    grid_state["cash"] = float(acc.cash)
    positions = client.get_all_positions()
    btc_qty = 0.0
    for pos in positions:
        sym = str(pos.symbol).replace("/", "")
        if sym in ("BTCUSD", "BTC"):
            btc_qty = float(pos.qty)
    grid_state["btc"] = btc_qty
    logger.info("Konto: Equity=%.2f USD, Cash=%.2f, BTC=%.6f",
                grid_state["capital"], grid_state["cash"], grid_state["btc"])


def _find_order_by_id(grid_state, order_id):
    """
    Findet eine Order anhand der Alpaca-order_id.

    Output: ('buy'/'sell', level_idx) oder (None, None).
    """
    for idx, order in grid_state["buy_orders"].items():
        if order.get("order_id") == order_id:
            return "buy", idx
    for idx, order in grid_state["sell_orders"].items():
        if order.get("order_id") == order_id:
            return "sell", idx
    return None, None


def _order_notional(qty, price):
    """Schaetzt den USD-Wert einer Order (Alpaca-Minimum ~10 USD)."""
    return float(qty) * float(price)


def place_limit_for_level(grid_state, side, idx, logger):
    """
    Platziert eine Limit-Order fuer ein Grid-Level, falls noch keine order_id.

    Input:  grid_state, side ('buy'/'sell'), level_idx, logger
    Output: order_id oder None.
    """
    book = grid_state["buy_orders"] if side == "buy" else grid_state["sell_orders"]
    order = book.get(idx)
    if order is None:
        return None
    if order.get("order_id"):
        return order["order_id"]

    qty = order["qty"]
    price = order["price"]
    if _order_notional(qty, price) < 10.0:
        logger.warning("Level %d %s zu klein (%.2f USD) — Alpaca-Minimum 10 USD.",
                       idx, side.upper(), _order_notional(qty, price))
        return None

    oid = execution.place_order(config.SYMBOL, side, qty, price, logger=logger)
    if oid:
        order["order_id"] = oid
    return oid


def cancel_all_orders(grid_state, client, logger):
    """
    Storniert alle offenen Grid-Orders bei Alpaca und raeumt order_ids auf.

    Input:  grid_state, client, logger
    Output: nichts.
    """
    seen = set()
    for book in (grid_state["buy_orders"], grid_state["sell_orders"]):
        for order in book.values():
            oid = order.get("order_id")
            if oid and oid not in seen:
                execution.cancel_order(oid, logger=logger)
                seen.add(oid)
            order.pop("order_id", None)

    # Sicherheit: auch verwaiste offene Orders auf dem Konto canceln.
    try:
        open_orders = client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for o in open_orders:
            if str(o.symbol).replace("/", "") in ("BTCUSD", "BTC/USD", "BTC"):
                execution.cancel_order(str(o.id), logger=logger)
    except Exception as err:
        logger.warning("Offene Orders abfragen fehlgeschlagen: %s", err)


def deploy_all_limits(grid_state, logger):
    """
    Platziert Limit-Orders fuer alle Level ohne order_id.

    Input:  grid_state, logger
    Output: Anzahl neu platzierter Orders.
    """
    placed = 0
    for idx in sorted(grid_state["buy_orders"].keys()):
        if place_limit_for_level(grid_state, "buy", idx, logger):
            placed += 1
    for idx in sorted(grid_state["sell_orders"].keys()):
        if place_limit_for_level(grid_state, "sell", idx, logger):
            placed += 1
    logger.info("%d Limit-Orders platziert.", placed)
    return placed


def process_fills(grid_state, client, logger):
    """
    Prueft gefuellte Alpaca-Orders und setzt Gegenorders (Grid-Mechanik).

    Input:  grid_state, client, logger
    Output: Anzahl verarbeiteter Fills.
    """
    try:
        closed = client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=200))
    except Exception as err:
        logger.error("Gefuellte Orders nicht abrufbar: %s", err)
        return 0

    fills = 0
    for alpaca_order in closed:
        if alpaca_order.status != OrderStatus.FILLED:
            continue
        oid = str(alpaca_order.id)
        side, idx = _find_order_by_id(grid_state, oid)
        if side is None:
            continue

        if side == "buy":
            grid_logic.handle_buy_fill(grid_state, idx)
            fills += 1
            logger.info("Buy-Fill Level %d @ %.2f — Gegen-Sell wird gesetzt.", idx,
                        float(alpaca_order.filled_avg_price or 0))
            up = idx + 1
            if up in grid_state["sell_orders"]:
                place_limit_for_level(grid_state, "sell", up, logger)
        else:
            grid_logic.handle_sell_fill(grid_state, idx)
            fills += 1
            logger.info("Sell-Fill Level %d @ %.2f — Gegen-Buy wird gesetzt.", idx,
                        float(alpaca_order.filled_avg_price or 0))
            down = idx - 1
            if down in grid_state["buy_orders"]:
                place_limit_for_level(grid_state, "buy", down, logger)

    if fills:
        logger.info("%d Fills verarbeitet.", fills)
    return fills


def close_grid_live(grid_state, client, logger):
    """
    Grid live aufloesen: Orders canceln, BTC per Market verkaufen.

    Input:  grid_state, client, logger
    Output: nichts.
    """
    cancel_all_orders(grid_state, client, logger)
    sync_account(grid_state, client, logger)
    if grid_state["btc"] > 0:
        execution.place_order(config.SYMBOL, "sell", grid_state["btc"],
                              limit_price=None, logger=logger)
    grid_state["buy_orders"] = {}
    grid_state["sell_orders"] = {}
    grid_state["active"] = False
    sync_account(grid_state, client, logger)
    logger.info("Grid geschlossen (Spot, alle Orders gecancelt).")


def open_grid_live(grid_state, close, sma, spacing, client, logger):
    """
    Grid live eroeffnen: grid_logic plant, execution setzt Orders.

    Input:  grid_state, close, sma, spacing, client, logger
    Output: nichts.
    """
    sync_account(grid_state, client, logger)
    grid_logic.open_grid(grid_state, close, sma, spacing)

    # Seed-Inventar fuer Sell-Levels: Market-Buy auf Alpaca (Spot-only).
    seed_qty = sum(o["qty"] for o in grid_state["sell_orders"].values())
    if seed_qty > 0 and _order_notional(seed_qty, close) >= 10.0:
        execution.place_order(config.SYMBOL, "buy", seed_qty,
                              limit_price=None, logger=logger)
        logger.info("Seed-Market-Buy: %.6f BTC fuer obere Sell-Levels.", seed_qty)

    deploy_all_limits(grid_state, logger)
    sync_account(grid_state, client, logger)
    logger.info("Grid geoeffnet (Spacing=%.3f%%, %d Buy / %d Sell-Levels).",
                spacing * 100, len(grid_state["buy_orders"]),
                len(grid_state["sell_orders"]))


def rebuild_grid_live(grid_state, close, sma, new_spacing, client, logger):
    """
    Grid live umbauen: Orders canceln, rebuild_grid, Limits neu setzen.

    Inventar bleibt auf dem Konto — kein Market-Liquidieren (wie Backtest).
    """
    cancel_all_orders(grid_state, client, logger)
    grid_logic.rebuild_grid(grid_state, close, sma, new_spacing)
    deploy_all_limits(grid_state, logger)
    sync_account(grid_state, client, logger)
    logger.info("Grid-Rebuild auf Spacing %.3f%% abgeschlossen.", new_spacing * 100)


def run_once(force=False):
    """
    Fuehrt EINEN taeglichen Monitor-Durchlauf aus (Cronjob-tauglich).

    Input:  force=True ueberspringt Duplikat-Schutz fuer manuelle Tests
    Output: nichts.
    """
    logger = config.setup_logging()
    logger.info("=== monitor.py Start (Alpaca Paper, Spot-only) ===")

    if not config.ALPACA_PAPER:
        logger.error("ALPACA_PAPER ist False — Abbruch (kein Live-Trading).")
        return
    client = execution.get_trading_client()
    if client is None:
        logger.error("TradingClient nicht verfuegbar — Keys in .env pruefen.")
        return

    # 1) Daten laden (frisch von Alpaca fuer aktuellen Tagesschluss).
    df = data.load_data(force_refresh=True)
    df = sma_filter.add_sma(df)
    feature_df = ml_spacing.compute_features(df)[ml_spacing.FEATURE_COLUMNS]
    model = ml_spacing.load_model()
    if model is None:
        logger.warning("Kein ML-Modell — Fallback auf statisches Spacing.")

    i = len(df) - 1
    bar_date = df.index[i].date()
    close = df["close"].iloc[i]
    sma = df["sma"].iloc[i]
    logger.info("Letzte Kerze: %s  Close=%.2f  SMA-120=%.2f", bar_date, close, sma)

    # 2) state.json laden oder neu anlegen.
    snap = data.load_state()
    if snap is None:
        snap = data.empty_state_snapshot()
    grid_state = data.grid_state_from_snapshot(snap)
    sync_account(grid_state, client, logger)

    # Duplikat-Schutz: nicht zweimal am selben Tag laufen (Cronjob-Sicherheit).
    if snap.get("last_bar_date") == str(bar_date) and not force:
        logger.info("Bereits verarbeitet fuer %s — ueberspringe (force=True zum Erzwingen).",
                    bar_date)
        return

    was_active = grid_state["active"]

    # 3) Fills verarbeiten (wenn Grid aktiv).
    if grid_state["active"]:
        process_fills(grid_state, client, logger)
        _persist(grid_state, _sma_status(True), bar_date, logger)

    # 4) ML-Rebuild pruefen.
    if grid_state["active"] and model is not None:
        feats = feature_df.iloc[i].to_dict()
        if not any(pd.isna(v) for v in feats.values()) and not pd.isna(sma):
            new_spacing = ml_spacing.predict_spacing(model, feats)
            if ml_spacing.should_rebuild(grid_state["spacing"], new_spacing):
                rebuild_grid_live(grid_state, close, sma, new_spacing, client, logger)
                _persist(grid_state, _sma_status(True), bar_date, logger)

    # 5) SMA-Signal am Tagesschluss.
    status = sma_filter.trend_status(df, i, grid_state["active"])
    target_active = status["active"]
    sma_status = _sma_status(target_active)

    if grid_state["active"] and not target_active:
        reason = "Stop-Loss" if status["stop_loss_hit"] else "Exit"
        logger.info("%s ausgeloest — Grid wird geschlossen.", reason)
        close_grid_live(grid_state, client, logger)
        _persist(grid_state, "PAUSE", bar_date, logger)

    elif (not grid_state["active"]) and target_active:
        if model is not None:
            feats = feature_df.iloc[i].to_dict()
            spacing = ml_spacing.predict_spacing(model, feats) \
                if not any(pd.isna(v) for v in feats.values()) \
                else config.GRID_SPACING_PCT
        else:
            spacing = config.GRID_SPACING_PCT
        if not pd.isna(sma):
            logger.info("Entry bestaetigt — Grid wird geoeffnet (Spacing=%.3f%%).",
                        spacing * 100)
            open_grid_live(grid_state, close, sma, spacing, client, logger)
            _persist(grid_state, "LONG", bar_date, logger)

    elif grid_state["active"] and target_active:
        # Grid laeuft weiter — fehlende Limits nachziehen.
        deploy_all_limits(grid_state, logger)
        _persist(grid_state, "LONG", bar_date, logger)
        logger.info("Grid aktiv — gehalten.")

    else:
        _persist(grid_state, "PAUSE", bar_date, logger)
        logger.info("Bot pausiert (SMA-Filter: kein Long-Signal).")

    logger.info("=== monitor.py Ende (Status=%s) ===", sma_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Grid Bot — taeglicher Monitor-Lauf")
    parser.add_argument("--force", action="store_true",
                        help="Duplikat-Schutz fuer diesen Tag ignorieren (manueller Test)")
    args = parser.parse_args()
    run_once(force=args.force)
