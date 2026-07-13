"""
data.py — Laedt historische BTC/USD OHLCV-Tagesdaten von Alpaca.

Hauptaufgabe: Tageskerzen ab dem fruehesten Alpaca-Datum (2021-01-01) ueber
alpaca-py (CryptoHistoricalDataClient) holen, lokal als CSV cachen und als
pandas DataFrame zurueckgeben. Der Cache macht Backtests reproduzierbar und
schont die API.

WICHTIG — Struktur & Timezone:
  - Alpaca liefert einen MultiIndex (symbol, timestamp) mit tz-AWARE
    UTC-Zeitstempeln. Wir flachen den DataFrame ab (reset_index), behalten
    nur open/high/low/close/volume und wandeln den Zeitstempel projektweit
    einheitlich in ein tz-NAIVE Tagesdatum um (.tz_localize(None)).
  - Damit hat der DataFrame exakt dieselbe flache, tz-naive Struktur wie
    frueher unter ccxt — der Rest der Architektur bleibt unveraendert.

Zusaetzlich (fuer Schritt 9, Live-Betrieb): kleine Helfer zum Lesen und
Schreiben der state.json.
"""

import os
import json
from datetime import datetime, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config


# Lokaler Cache fuer die heruntergeladenen Alpaca-Tagesdaten.
CACHE_PATH = "data/btc_usd_1d.csv"


def _get_client():
    """
    Erzeugt den Alpaca-Krypto-Datenclient.

    Fuer historische Krypto-Daten sind keine Keys zwingend noetig; wenn sie
    in .env vorhanden sind, nutzen wir sie (hoehere Rate-Limits).
    """
    if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
        return CryptoHistoricalDataClient(config.ALPACA_API_KEY,
                                          config.ALPACA_SECRET_KEY)
    return CryptoHistoricalDataClient()


def fetch_ohlcv_all(symbol=config.SYMBOL, start=config.DATA_START):
    """
    Holt ALLE Tageskerzen ab 'start' fuer 'symbol' von Alpaca.

    Input:  symbol (z.B. "BTC/USD"), start ("2021-01-01")
    Output: pandas DataFrame mit Spalten open/high/low/close/volume,
            tz-naivem DatetimeIndex (UTC-Tag).

    alpaca-py paginiert intern automatisch — wir bekommen die volle Historie
    in einem Aufruf zurueck und muessen nur noch umformatieren.
    """
    client = _get_client()
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
    )
    bars = client.get_crypto_bars(req)

    # Alpaca-DataFrame: MultiIndex (symbol, timestamp) -> flach machen.
    df = bars.df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    # Nur die klassischen OHLCV-Spalten behalten (wie frueher unter ccxt).
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Zeitstempel ist tz-AWARE (UTC) -> projektweit einheitlich tz-NAIVE
    # Tagesdatum. Sonst crasht Pandas bei Vergleichen mit naiven Datums.
    df["date"] = pd.to_datetime(df["timestamp"], utc=True) \
        .dt.tz_localize(None).dt.normalize()
    df = df.drop(columns=["timestamp"]).set_index("date")

    # Duplikate entfernen und sortieren.
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_data(force_refresh=False):
    """
    Liefert die BTC/USD-Tagesdaten als DataFrame — aus dem Cache oder frisch.

    Input:  force_refresh=True erzwingt einen Neu-Download trotz Cache.
    Output: DataFrame open/high/low/close/volume mit tz-naivem Tagesindex,
            OHNE die heutige (noch unfertige) Kerze.

    Die laufende Kerze des aktuellen UTC-Tages ist noch nicht abgeschlossen.
    Wir entfernen sie, damit kein Look-ahead durch halbfertige Daten entsteht.
    """
    if os.path.exists(CACHE_PATH) and not force_refresh:
        print(f"Lade Daten aus Cache: {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"], index_col="date")
    else:
        print("Lade Daten frisch von Alpaca (alpaca-py) ...")
        df = fetch_ohlcv_all()
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_csv(CACHE_PATH)
        print(f"Cache gespeichert: {CACHE_PATH}")

    # Heutige, noch nicht geschlossene Tageskerze (UTC) entfernen.
    today_utc = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    df = df[df.index < today_utc]
    return df


# ---------------------------------------------------------------------------
# STATE MANAGEMENT (Live-Betrieb, Schritt 9)
# ---------------------------------------------------------------------------

def empty_state_snapshot():
    """
    Leerer Laufzustand fuer den allerersten Start oder nach Reset.

    Output: dict im state.json-Schema (restart-sicher).
    """
    return {
        "grid_active": False,
        "sma_status": "PAUSE",
        "grid_start_price": 0.0,
        "current_spacing": config.GRID_SPACING_PCT,
        "last_run_at": None,
        "last_bar_date": None,
        "capital": 0.0,
        "levels": [],
        "per_level": 0.0,
        "buy_orders": {},
        "sell_orders": {},
        "trades": 0,
        "realized": 0.0,
        "fees": 0.0,
    }


def _orders_to_json(orders):
    """Wandelt Order-Buecher (int-Keys) in JSON-taugliche str-Keys um."""
    out = {}
    for idx, order in orders.items():
        entry = dict(order)
        out[str(idx)] = entry
    return out


def _orders_from_json(orders):
    """Laedt Order-Buecher aus state.json zurueck (int-Keys)."""
    out = {}
    for idx, order in orders.items():
        out[int(idx)] = dict(order)
    return out


def snapshot_from_grid(grid_state, sma_status, last_bar_date=None):
    """
    Baut das state.json-Snapshot aus dem grid_logic-Zustand.

    Input:  grid_state (dict), sma_status ('LONG'/'PAUSE'), last_bar_date
    Output: dict zum Speichern in state.json.
    """
    return {
        "grid_active": grid_state["active"],
        "sma_status": sma_status,
        "grid_start_price": grid_state["start_price"],
        "current_spacing": grid_state["spacing"],
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "last_bar_date": str(last_bar_date) if last_bar_date else None,
        "capital": grid_state["capital"],
        "levels": grid_state["levels"],
        "per_level": grid_state["per_level"],
        "buy_orders": _orders_to_json(grid_state["buy_orders"]),
        "sell_orders": _orders_to_json(grid_state["sell_orders"]),
        "trades": grid_state["trades"],
        "realized": grid_state["realized"],
        "fees": grid_state["fees"],
    }


def grid_state_from_snapshot(snap, cash=None, btc=None):
    """
    Stellt den grid_logic-Zustand aus state.json wieder her.

    Input:  snap (dict aus load_state), optional cash/btc von Alpaca
    Output: grid_state-dict fuer grid_logic/monitor.
    """
    state = {
        "capital": snap.get("capital", 0.0),
        "cash": cash if cash is not None else snap.get("capital", 0.0),
        "btc": btc if btc is not None else 0.0,
        "fees": snap.get("fees", 0.0),
        "realized": snap.get("realized", 0.0),
        "trades": snap.get("trades", 0),
        "active": snap.get("grid_active", False),
        "levels": snap.get("levels", []),
        "buy_orders": _orders_from_json(snap.get("buy_orders", {})),
        "sell_orders": _orders_from_json(snap.get("sell_orders", {})),
        "per_level": snap.get("per_level", 0.0),
        "spacing": snap.get("current_spacing", config.GRID_SPACING_PCT),
        "start_price": snap.get("grid_start_price", 0.0),
    }
    return state


def load_state():
    """
    Liest state.json. Gibt None zurueck, wenn noch keine Datei existiert.

    Output: dict im state.json-Schema oder None.
    """
    if not os.path.exists(config.STATE_PATH):
        return None
    with open(config.STATE_PATH, "r") as f:
        return json.load(f)


def save_state(snapshot):
    """
    Schreibt den Laufzustand sofort in state.json (restart-sicher).

    Input:  snapshot-dict (siehe snapshot_from_grid)
    Output: nichts.
    """
    snapshot["last_run_at"] = datetime.now(timezone.utc).isoformat()
    with open(config.STATE_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)


if __name__ == "__main__":
    print("=== data.py Test (Alpaca BTC/USD) ===")
    df = load_data()

    print(f"\nGeladene Tageskerzen: {len(df)}")
    print(f"Zeitraum: {df.index.min().date()} bis {df.index.max().date()}")

    print("\nErste 3 Zeilen:")
    print(df.head(3))

    print("\nLetzte 3 Zeilen:")
    print(df.tail(3))

    # Mini-Plausibilitaet: High >= Low, keine NaNs im close, tz-naiv.
    assert (df["high"] >= df["low"]).all(), "Datenfehler: High < Low gefunden"
    assert df["close"].notna().all(), "Datenfehler: NaN im close"
    assert df.index.tz is None, "Index muss tz-naiv sein (Projekt-Konvention)"
    print("\nOK  Plausibilitaet (High>=Low, kein NaN, tz-naiv) bestanden.")
