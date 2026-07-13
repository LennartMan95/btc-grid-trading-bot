"""
execution.py — Minimalistische Alpaca-Order-Anbindung (Paper, Spot-only).

Nur zwei oeffentliche Funktionen: place_order und cancel_order.
Echtes Geld wird NICHT bewegt — ALPACA_PAPER muss True sein.

Bei API-Fehlern (Timeout, Rate-Limit): 3 Retries mit kurzer Pause,
danach Fehler loggen und None zurueckgeben statt zu crashen.
"""

import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config


RETRY_PAUSE_SEC = 2
MAX_RETRIES = 3


def get_trading_client():
    """
    Erzeugt den Alpaca TradingClient im Paper-Modus.

    Output: TradingClient oder None bei fehlenden Keys / Live-Modus.
    """
    if not config.ALPACA_PAPER:
        return None
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        return None
    return TradingClient(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        paper=True,
    )


def _retry(callable_fn, logger):
    """
    Fuehrt callable_fn bis zu 3x aus, mit Pause zwischen Versuchen.

    Input:  callable_fn (Funktion ohne Argumente), logger
    Output: Ergebnis oder None bei endgueltigem Fehler.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return callable_fn()
        except Exception as err:
            last_err = err
            if logger:
                logger.warning("API-Versuch %d/%d fehlgeschlagen: %s",
                               attempt, MAX_RETRIES, err)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_PAUSE_SEC)
    if logger:
        logger.error("API nach %d Versuchen abgebrochen: %s",
                     MAX_RETRIES, last_err)
    return None


def place_order(symbol, side, qty, limit_price=None, logger=None):
    """
    Platziert eine Limit- oder Market-Order bei Alpaca (Paper).

    Input:  symbol (z.B. 'BTC/USD'), side ('buy'/'sell'), qty (BTC),
            limit_price (None = Market-Order), logger
    Output: order_id (str) oder None bei Fehler.

    Limit-Orders sind Maker (Grid-Standard). Market-Orders nur fuer
    Grid-Start (Seed-Inventar) und Grid-Close (alles verkaufen).
    """
    client = get_trading_client()
    if client is None:
        if logger:
            logger.error("Kein TradingClient — Keys fehlen oder nicht im Paper-Modus.")
        return None

    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    qty = round(float(qty), 8)

    def _submit():
        if limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                limit_price=round(float(limit_price), 2),
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )
        order = client.submit_order(req)
        return str(order.id)

    order_id = _retry(_submit, logger)
    if order_id and logger:
        kind = "Limit" if limit_price else "Market"
        logger.info("%s-Order platziert: %s %s qty=%s @ %s",
                    kind, order_id, side.upper(), qty,
                    limit_price if limit_price else "MARKET")
    return order_id


def cancel_order(order_id, logger=None):
    """
    Storniert eine offene Order bei Alpaca.

    Input:  order_id (str), logger
    Output: True bei Erfolg, False bei Fehler.
    """
    client = get_trading_client()
    if client is None:
        if logger:
            logger.error("Kein TradingClient — Cancel nicht moeglich.")
        return False

    oid = str(order_id)

    def _cancel():
        client.cancel_order_by_id(oid)
        return True

    ok = _retry(_cancel, logger)
    if ok and logger:
        logger.info("Order storniert: %s", oid)
    return bool(ok)


if __name__ == "__main__":
    log = config.setup_logging("execution_test")
    log.info("=== execution.py Test (Paper) ===")
    assert config.ALPACA_PAPER is True, "Nur Paper-Modus erlaubt"

    # Limit-Order weit unter Markt -> sofort wieder canceln.
    oid = place_order(config.SYMBOL, "buy", 0.001, limit_price=10000.0, logger=log)
    if oid:
        assert cancel_order(oid, logger=log), "Cancel muss funktionieren"
        log.info("OK  place_order + cancel_order erfolgreich.")
    else:
        log.error("Test fehlgeschlagen — keine Order platziert.")
