"""
config.py — Zentrale Konfiguration fuer den BTC Grid-Trading-Bot.

Hier stehen ALLE Parameter und Konstanten an einem Ort. Jede andere
Datei importiert ihre Einstellungen aus diesem Modul. So gibt es nur
eine einzige Quelle der Wahrheit fuer die Strategie.

Das Logging-Setup (Standard-Library logging mit File + Console) wird
erst ab Schritt 9 (Live-Betrieb) hier ergaenzt. Bis dahin nutzen die
Backtest-Module einfaches print().
"""

# ---------------------------------------------------------------------------
# MARKT / DATEN
# ---------------------------------------------------------------------------

SYMBOL = "BTC/USDT"          # Handelspaar (erst Spot, spaeter Futures)
TIMEFRAME = "1d"             # Tageskerzen, UTC-Schluss
DATA_START = "2017-01-01"    # Historie ab 2017 fuer Backtest + ML-Training

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
# ca. 0.2% Maker/Taker-Gebuehren (Binance/Bybit Round-Trip). Im Normal-
# betrieb wird dieser Wert taeglich vom ML-Modell ueberschrieben.
GRID_SPACING_PCT = 0.005

MAX_GRID_COUNT = 200             # Harte Obergrenze fuer die Anzahl Grid-Level

# ---------------------------------------------------------------------------
# ML-SPACING GRENZEN
# ---------------------------------------------------------------------------
# Das ML-Modell sagt direkt ein Spacing in % voraus. Diese Grenzen
# begrenzen die Vorhersage auf einen sinnvollen Bereich (clip).

# 0.3% reiner Gebuehrenschutz — Binance/Bybit Round-Trip ist 0.2%,
# minimale Nettomarge. Das Modell entscheidet selbst, ob es hoeher geht.
ML_SPACING_MIN = 0.003

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
# KAPITAL & HEBEL
# ---------------------------------------------------------------------------

CAPITAL_INVESTED = 0.80          # 80% des Kapitals werden im Grid eingesetzt
CAPITAL_RESERVE = 0.20           # 20% Reserve (Puffer fuer Nachkaeufe/Fees)

# Harte Obergrenze fuer den Hebel. Wird im Spot-Test mit 1 gefahren,
# spaeter Futures mit MAXIMAL 3x. risk.py muss diesen Wert pruefen und
# darf ihn NIEMALS ueberschreiten.
LEVERAGE = 3

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
# Realistische Gebuehren fuer Profitberechnung im Backtest und Live.
# Binance/Bybit: ca. 0.1% Maker + 0.1% Taker. Ein vollstaendiger
# Grid-Trade (Buy + Sell) kostet damit rund 0.2% Round-Trip.
MAKER_FEE = 0.001                # 0.1% pro Maker-Order
TAKER_FEE = 0.001                # 0.1% pro Taker-Order
FEE_PER_TRADE = MAKER_FEE + TAKER_FEE   # 0.2% Round-Trip pro Grid-Trade

# ---------------------------------------------------------------------------
# DATEIPFADE
# ---------------------------------------------------------------------------

MODEL_PATH = "models/spacing_model.pkl"   # Trainiertes ML-Modell
STATE_PATH = "state.json"                 # Laufzustand (nur Live-Betrieb)
LOG_PATH = "grid_bot.log"                 # Logdatei (ab Schritt 9)

# ---------------------------------------------------------------------------
# BACKTEST-ZEITRAEUME
# ---------------------------------------------------------------------------

IN_SAMPLE_START = "2017-01-01"   # Trainings-/In-Sample-Periode
IN_SAMPLE_END = "2021-12-31"
OUT_SAMPLE_START = "2022-01-01"  # Out-of-Sample — der entscheidende Vergleich
OUT_SAMPLE_END = "2024-12-31"


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

    # Spacing-Untergrenze muss klar ueber den Gebuehren liegen.
    assert ML_SPACING_MIN > 0.002, \
        "ML_SPACING_MIN muss groesser als 0.2% (Round-Trip-Gebuehren) sein"
    print(f"OK  ML_SPACING_MIN = {ML_SPACING_MIN:.3%} > 0.2% Gebuehren")

    # Hebel-Obergrenze niemals ueberschreiten.
    assert LEVERAGE <= 3, "LEVERAGE darf 3x niemals ueberschreiten"
    print(f"OK  LEVERAGE = {LEVERAGE}x (<= 3x)")

    # Zusaetzliche Plausibilitaet: Spacing-Grenzen sinnvoll geordnet.
    assert ML_SPACING_MIN < ML_SPACING_MAX, \
        "ML_SPACING_MIN muss kleiner als ML_SPACING_MAX sein"
    print(f"OK  Spacing-Range: {ML_SPACING_MIN:.3%} bis {ML_SPACING_MAX:.3%}")

    print("Alle Sanity-Checks bestanden.")
