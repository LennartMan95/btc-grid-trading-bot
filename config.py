"""
config.py — Zentrale Konfiguration fuer den BTC Grid-Trading-Bot.

Hier stehen ALLE Parameter und Konstanten an einem Ort. Jede andere
Datei importiert ihre Einstellungen aus diesem Modul. So gibt es nur
eine einzige Quelle der Wahrheit fuer die Strategie.

Das Logging-Setup (Standard-Library logging mit File + Console) liegt
ab Schritt 9 ebenfalls hier — ein Aufruf von setup_logging() genuegt.
"""

import logging
import os

from dotenv import load_dotenv

# .env laden (API-Keys etc.). Secrets stehen AUSSCHLIESSLICH in .env,
# niemals hartkodiert im Code. .env bleibt in .gitignore.
load_dotenv()

# ---------------------------------------------------------------------------
# MARKT / DATEN
# ---------------------------------------------------------------------------

SYMBOL = "BTC/USD"           # Alpaca-Krypto-Spot-Paar (USD, nicht USDT)
TIMEFRAME = "1d"             # Tageskerzen, UTC-Schluss
# Alpacas Krypto-Historie beginnt erst 2021-01-01 (frueher keine Daten).
DATA_START = "2021-01-01"

# ---------------------------------------------------------------------------
# ALPACA (Paper-Trading, Spot-only, KEIN Hebel)
# ---------------------------------------------------------------------------
# Keys kommen aus .env (python-dotenv). Fehlen sie, bleiben sie None —
# fuer den Abruf historischer Krypto-Daten sind keine Keys noetig.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Sicherheitsschalter: fuer diese Abgabe IMMER Paper-Trading (nie live).
# Wird spaeter explizit an jeden Alpaca-TradingClient uebergeben
# (TradingClient(..., paper=ALPACA_PAPER)).
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# SMA-RICHTUNGSFILTER
# ---------------------------------------------------------------------------
# Der 120-Tage-SMA entscheidet die Richtung: Long-Grid oder Pause.
# KEIN Short-Grid. Bei Schwaeche pausiert der Bot komplett.

SMA_PERIOD = 120             # Laenge des Trend-SMA in Tagen

# Buffer um den SMA: Der Schlusskurs muss DEUTLICH ueber/unter dem SMA
# liegen, sonst zittert der Bot bei jeder kleinen SMA-Beruehrung hin und
# her. +1% Abstand filtert dieses Rauschen und verlangt klares Momentum.
SMA_ENTRY_BUFFER = 0.01      # +1% ueber SMA fuer Einstieg
SMA_EXIT_BUFFER = 0.01       # -1% unter SMA fuer Ausstieg

# Wie viele Tageskerzen in Folge die Bedingung erfuellen muessen, bevor
# der Bot reagiert. Verhindert Fehlsignale durch eine einzelne Kerze.
CONFIRMATION_CANDLES_ENTRY = 2
CONFIRMATION_CANDLES_EXIT = 2

# ---------------------------------------------------------------------------
# DYNAMISCHER STOP-LOSS
# ---------------------------------------------------------------------------
# Untere Schutzgrenze = SMA_120 * 0.99. Wird taeglich neu berechnet und
# zieht im Bullenmarkt automatisch mit nach oben. Der Faktor 0.99 laesst
# 1% Puffer unter dem SMA, damit ein kurzes Unterschreiten nicht sofort
# zum Stop fuehrt (Bestaetigung ueber CONFIRMATION_CANDLES_EXIT zusaetzlich).
STOP_LOSS_SMA_FACTOR = 0.99

# ---------------------------------------------------------------------------
# GRID-PARAMETER
# ---------------------------------------------------------------------------
# LOWER_PRICE und UPPER_PRICE werden zur Laufzeit aus SMA und aktuellem
# Preis berechnet (siehe grid_logic.py). Hier nur die Faktoren/Regeln.

LOWER_PRICE_SMA_FACTOR = 0.99    # LOWER_PRICE = SMA_120 * 0.99 (beim Start)
UPPER_PRICE_FACTOR = 10          # UPPER_PRICE = aktueller_preis * 10 (theoret. Limit)

# Statischer Fallback fuer das Grid-Spacing (0.5%). Sichert Profit nach
# ca. 0.2% Maker/Taker-Gebuehren (Round-Trip, Annahme). Im Normalbetrieb
# wird dieser Wert taeglich vom ML-Modell ueberschrieben.
GRID_SPACING_PCT = 0.005

MAX_GRID_COUNT = 200             # Harte Obergrenze fuer die Anzahl Grid-Level

# ---------------------------------------------------------------------------
# ML-SPACING GRENZEN
# ---------------------------------------------------------------------------
# Das ML-Modell sagt direkt ein Spacing in % voraus. Diese Grenzen
# begrenzen die Vorhersage auf einen sinnvollen Bereich (clip).

# 0.5% harte Untergrenze — bei Alpaca kostet ein Round-Trip 0.4%
# (0.15% Maker + 0.25% Taker). Ein Spacing unter 0.4% waere strukturell
# ein Verlustgeschaeft; 0.5% laesst ~0.1% Nettomarge pro Trade. Nie
# darunter. Das Modell darf nach oben frei entscheiden.
ML_SPACING_MIN = 0.005

# 5% Soft-Limit nur gegen Ausreisser bei Datenfehlern. Kein harter Clip.
# Normalbereich: 0.5-2%.
ML_SPACING_MAX = 0.050

# Faustregel: natuerlichen Preisbereich in ca. 10 Grid-Stufen aufteilen.
# Wird im Target von ml_spacing.py verwendet.
GRID_LEVELS_PER_RANGE = 10

# Rebuild des Grids erst, wenn das neue Spacing > 20% vom aktuellen
# abweicht. Verhindert staendiges Neuaufbauen bei kleinen Schwankungen.
ML_REBUILD_THRESHOLD = 0.20

# Tage in die Zukunft fuer das ML-Target (OPTIMAL_SPACING_FORWARD).
ML_TARGET_FORWARD_DAYS = 3

# ---------------------------------------------------------------------------
# KAPITAL (Spot-only, KEIN Hebel)
# ---------------------------------------------------------------------------

CAPITAL_INVESTED = 0.80          # 80% des Kapitals werden im Grid eingesetzt
CAPITAL_RESERVE = 0.20           # 20% Reserve (Puffer fuer Nachkaeufe/Fees)
# Kein Hebel/Leverage mehr: reines Spot-Trading ueber Alpaca.

# ---------------------------------------------------------------------------
# STOP-LOSS / TAKE-PROFIT (Strategie-Ebene)
# ---------------------------------------------------------------------------
# Der eigentliche Stop-Loss-Preis wird taeglich dynamisch als
# SMA_120 * STOP_LOSS_SMA_FACTOR berechnet (siehe oben). Hier nur Schalter.

STOP_LOSS_ENABLED = True
TAKE_PROFIT_ENABLED = False      # Optional, bei Bedarf konfigurierbar
TAKE_PROFIT_PCT = None           # z.B. 0.50 fuer +50% — None = deaktiviert

# ---------------------------------------------------------------------------
# GEBUEHREN
# ---------------------------------------------------------------------------
# Reale Alpaca-Krypto-Gebuehren (Einstiegsstufe, Volumen < 100k USD).
# Ein Grid-Round-Trip = Buy (Maker) + Sell (Taker) = 0.15% + 0.25% = 0.4%.
# Diese 0.4% muss jedes Spacing erst verdienen, bevor Profit entsteht —
# deshalb liegt ML_SPACING_MIN darueber (siehe oben).
MAKER_FEE = 0.0015               # 0.15% pro Maker-Order (Alpaca)
TAKER_FEE = 0.0025               # 0.25% pro Taker-Order (Alpaca)
FEE_PER_TRADE = MAKER_FEE + TAKER_FEE   # 0.4% Round-Trip pro Grid-Trade

# ---------------------------------------------------------------------------
# DATEIPFADE
# ---------------------------------------------------------------------------

MODEL_PATH = "models/spacing_model.pkl"   # Trainiertes ML-Modell
STATE_PATH = "state.json"                 # Laufzustand (nur Live-Betrieb)
LOG_PATH = "grid_bot.log"                 # Logdatei (ab Schritt 9)

# ---------------------------------------------------------------------------
# BACKTEST-ZEITRAEUME
# ---------------------------------------------------------------------------

# Alpaca-Daten ab 2021 -> neue Splits (mit Prof abgestimmt).
IN_SAMPLE_START = "2021-01-01"   # Trainings-/In-Sample-Periode (3 Jahre)
IN_SAMPLE_END = "2023-12-31"
OUT_SAMPLE_START = "2024-01-01"  # Out-of-Sample — der entscheidende Vergleich
OUT_SAMPLE_END = "2026-07-13"    # bis zum aktuell letzten verfuegbaren Tag


# ---------------------------------------------------------------------------
# LOGGING (Live-Betrieb, Schritt 9)
# ---------------------------------------------------------------------------

def setup_logging(name="grid_bot"):
    """
    Richtet Logging auf Datei (LOG_PATH) und Konsole ein.

    Input:  name (Logger-Name)
    Output: konfigurierter Logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# SANITY-CHECKS
# ---------------------------------------------------------------------------
# Faengt grobe Konfigurationsfehler sofort beim Start ab, bevor echtes
# Geld bewegt wird. Lieber hier hart abbrechen als spaeter im Live-Betrieb.

if __name__ == "__main__":
    print("=== config.py Sanity-Checks ===")

    # Kapitalaufteilung muss exakt 100% ergeben.
    assert abs(CAPITAL_INVESTED + CAPITAL_RESERVE - 1.0) < 1e-9, \
        "CAPITAL_INVESTED + CAPITAL_RESERVE muss 1.0 ergeben"
    print(f"OK  Kapital: {CAPITAL_INVESTED:.0%} investiert + "
          f"{CAPITAL_RESERVE:.0%} Reserve = 100%")

    # Spacing-Untergrenze muss klar ueber den Alpaca-Gebuehren liegen.
    assert ML_SPACING_MIN > FEE_PER_TRADE, \
        "ML_SPACING_MIN muss groesser als Round-Trip-Gebuehren (0.4%) sein"
    print(f"OK  ML_SPACING_MIN = {ML_SPACING_MIN:.3%} > "
          f"{FEE_PER_TRADE:.1%} Alpaca Round-Trip")

    # Sicherheit: Diese Abgabe laeuft ausschliesslich im Paper-Modus.
    assert ALPACA_PAPER is True, "ALPACA_PAPER muss True sein (kein Live-Trading)"
    print(f"OK  ALPACA_PAPER = {ALPACA_PAPER} (Paper-Trading, Spot, kein Hebel)")

    # Zusaetzliche Plausibilitaet: Spacing-Grenzen sinnvoll geordnet.
    assert ML_SPACING_MIN < ML_SPACING_MAX, \
        "ML_SPACING_MIN muss kleiner als ML_SPACING_MAX sein"
    print(f"OK  Spacing-Range: {ML_SPACING_MIN:.3%} bis {ML_SPACING_MAX:.3%}")

    print("Alle Sanity-Checks bestanden.")
