"""
data.py — Laedt historische BTC/USDT OHLCV-Tagesdaten (UTC-Schluss).

Hauptaufgabe (Schritt 2): Tageskerzen ab 2017 ueber ccxt von Binance
holen, lokal als CSV cachen und als pandas DataFrame zurueckgeben.
Der Cache sorgt dafuer, dass Backtests reproduzierbar sind und nicht
bei jedem Lauf erneut die API belasten.

Zusaetzlich (fuer Schritt 9, Live-Betrieb): kleine Helfer zum Lesen und
Schreiben der state.json. Im Backtest werden diese nicht gebraucht.
"""

import os
import json
import time

import pandas as pd
import ccxt

import config


# Lokaler Cache fuer die heruntergeladenen Tagesdaten.
CACHE_PATH = "data/btc_usdt_1d.csv"

# Ein Tag in Millisekunden — ccxt arbeitet mit ms-Zeitstempeln.
ONE_DAY_MS = 24 * 60 * 60 * 1000


def fetch_ohlcv_all(symbol=config.SYMBOL, timeframe=config.TIMEFRAME,
                    start=config.DATA_START):
    """
    Holt ALLE Tageskerzen ab 'start' von Binance via ccxt.

    Input:  symbol (z.B. "BTC/USDT"), timeframe ("1d"), start ("2017-01-01")
    Output: pandas DataFrame mit Spalten open/high/low/close/volume,
            DatetimeIndex (UTC-Tag).

    Binance liefert pro Anfrage max. ~1000 Kerzen, daher paginieren wir
    mit dem 'since'-Zeitstempel, bis keine neuen Kerzen mehr kommen.
    """
    exchange = ccxt.binance()
    since = exchange.parse8601(start + "T00:00:00Z")
    limit = 1000

    all_rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows += batch
        print(f"  ... {len(batch)} Kerzen geladen (gesamt {len(all_rows)})")

        # Naechster Start = letzte Kerze + 1 Tag, sonst Endlosschleife.
        since = batch[-1][0] + ONE_DAY_MS

        # Weniger als 'limit' Kerzen => wir sind am aktuellen Rand angekommen.
        if len(batch) < limit:
            break

        # ccxt-Ratelimit respektieren (freundlich zur API bleiben).
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])

    # ms-Zeitstempel (UTC) in ein Tagesdatum umwandeln und als Index setzen.
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df = df.drop(columns=["timestamp"]).set_index("date")

    # Sicherheitshalber Duplikate entfernen und sortieren.
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_data(force_refresh=False):
    """
    Liefert die BTC-Tagesdaten als DataFrame — aus dem Cache oder frisch.

    Input:  force_refresh=True erzwingt einen Neu-Download trotz Cache.
    Output: DataFrame open/high/low/close/volume mit UTC-Tagesindex,
            OHNE die heutige (noch unfertige) Kerze.

    Die laufende Kerze des aktuellen UTC-Tages ist noch nicht
    abgeschlossen. Wir entfernen sie, damit kein Look-ahead durch
    halbfertige Daten entsteht.
    """
    if os.path.exists(CACHE_PATH) and not force_refresh:
        print(f"Lade Daten aus Cache: {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"], index_col="date")
    else:
        print("Lade Daten frisch von Binance (ccxt) ...")
        df = fetch_ohlcv_all()
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_csv(CACHE_PATH)
        print(f"Cache gespeichert: {CACHE_PATH}")

    # Heutige, noch nicht geschlossene Tageskerze (UTC) entfernen.
    today_utc = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    df = df[df.index < today_utc]
    return df


# ---------------------------------------------------------------------------
# STATE MANAGEMENT (nur Live-Betrieb, Schritt 9) — hier nur die Helfer.
# ---------------------------------------------------------------------------

def load_state():
    """
    Liest die state.json (Laufzustand) als dict. Gibt None zurueck,
    wenn noch keine Datei existiert (z.B. allererster Start).
    """
    if not os.path.exists(config.STATE_PATH):
        return None
    with open(config.STATE_PATH, "r") as f:
        return json.load(f)


def save_state(state):
    """
    Schreibt den Laufzustand (dict) sofort in state.json.
    Wird live nach jedem Fill/Rebuild aufgerufen, um Doppelorders
    nach einem Neustart zu verhindern.
    """
    with open(config.STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    print("=== data.py Test ===")
    df = load_data()

    print(f"\nGeladene Tageskerzen: {len(df)}")
    print(f"Zeitraum: {df.index.min().date()} bis {df.index.max().date()}")

    print("\nErste 3 Zeilen:")
    print(df.head(3))

    print("\nLetzte 3 Zeilen:")
    print(df.tail(3))

    # Mini-Plausibilitaet: High muss >= Low sein, keine NaNs in close.
    assert (df["high"] >= df["low"]).all(), "Datenfehler: High < Low gefunden"
    assert df["close"].notna().all(), "Datenfehler: NaN im close"
    print("\nOK  Plausibilitaet (High>=Low, kein NaN im close) bestanden.")
