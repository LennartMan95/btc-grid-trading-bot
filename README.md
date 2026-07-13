# BTC Grid Trading Bot

Automatisierter Bitcoin Grid-Trading-Bot mit **ML-optimiertem Grid-Spacing**
(Decision Tree Regressor), **120-Tage-SMA-Richtungsfilter** und **dynamischem
Stop-Loss**. Die Strategie kauft Dips und verkauft Rallyes innerhalb eines
Grids, fährt aber nur Long, wenn der SMA-Filter einen klaren Aufwärtstrend
bestätigt — sonst pausiert der Bot.

Daten und Ausführung laufen über **Alpaca** (Krypto-Spot, Paper-Trading).
Kein Hebel, kein Futures — reines Spot-Trading auf **BTC/USD**.

> Hinweis: Das ML-Modell ist ein **Decision Tree Regressor** (scikit-learn)
> mit Linear-Regression-Fallback — bewusst erklärbar und schnell trainierbar,
> keine GPU nötig.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Alpaca Paper-Keys eintragen
```

Jedes Modul ist eigenständig ausführbar (Selbsttest im `__main__`-Block), z.B.:

```bash
python config.py        # Sanity-Checks der Parameter
python data.py          # OHLCV-Daten von Alpaca laden
python sma_filter.py    # SMA + Entry/Exit/Stop testen
python ml_spacing.py    # Modell trainieren + Vorhersage
python backtest.py      # Vollständiger Backtest-Vergleich
python paper_trade.py   # Paper-Trading-Trockenlauf (Spot, kein Hebel)
python monitor.py       # Täglicher Live-Loop (Alpaca Paper, ein Durchlauf)
```

## Live-Betrieb (Cronjob)

Der Bot wird **einmal täglich** nach UTC-Tagesabschluss per Cronjob gestartet:

```bash
python monitor.py
```

Manueller Test (Duplikat-Schutz überspringen): `python monitor.py --force`

**Hinweis:** Der tägliche Cronjob erfordert, dass der Mac zum Ausführungszeitpunkt
nicht im Schlafmodus ist (Energieeinstellungen anpassen oder Mac angeschlossen lassen).

Beispiel-Cron (00:05 UTC):

```cron
CRON_TZ=UTC
5 0 * * * cd /Users/lennartmanske/Desktop/grid_bot_btc && /Users/lennartmanske/Desktop/grid_bot_btc/.venv/bin/python monitor.py >> /Users/lennartmanske/Desktop/grid_bot_btc/cron_monitor.log 2>&1
```

## Module

| Modul | Beschreibung |
|---|---|
| `config.py` | Zentrale Konfiguration: Parameter, Alpaca-Keys (`.env`), Gebühren, Splits. |
| `data.py` | Lädt BTC/USD-Tagesdaten (UTC-Schluss) via alpaca-py, cacht sie lokal als CSV. |
| `sma_filter.py` | 120-Tage-SMA, Entry/Exit-Confirmation und dynamischer Stop-Loss (`SMA*0.99`). |
| `ml_spacing.py` | Feature-Berechnung, Training und Vorhersage des Grid-Spacings in % (`.pkl`). |
| `grid_logic.py` | Grid-Mechanik: Level, Orders, Fill-Handling, Rebuild (Backtest + Live). |
| `backtest.py` | Historische Simulation mit OHLC-Fill-Logik: statisch vs. ML vs. Buy&Hold. |
| `paper_trade.py` | Täglicher Trockenlauf im Paper-Modus (Spot, kein Hebel). |
| `execution.py` | Minimalistische Alpaca-Order-Anbindung (`place_order`, `cancel_order`). |
| `monitor.py` | Tägliche Orchestrierung: SMA, ML, Grid, Fills, `state.json` (Cronjob). |

## Gebühren & Spacing

Alpaca-Krypto-Gebühren (Einstiegsstufe, in `config.py`):

| Order-Typ | Gebühr |
|---|---|
| Maker (Limit) | 0,15 % |
| Taker (Market) | 0,25 % |
| **Round-Trip** (Buy + Sell) | **0,40 %** |

Damit jedes Grid-Level nach Gebühren profitabel bleibt, liegt die harte
Untergrenze für das ML-Spacing bei **`ML_SPACING_MIN = 0,5 %`**
(≈ 0,1 % Nettomarge pro Round-Trip).

## Ergebnisse — Out-of-Sample (2024–2026)

Der entscheidende Vergleich auf ungesehenen Daten (Modell trainiert nur auf
2021–2023, Alpaca BTC/USD, realistische Gebühren):

| Strategie | CAGR p.a. | Sharpe | Max-Drawdown | Gebühren |
|---|---|---|---|---|
| Statisch (0,5 %) | 9,21 % | 0,63 | −19,31 % | 756 USD |
| **ML-Spacing** | **13,17 %** | **0,73** | **−17,76 %** | **519 USD** |
| Buy & Hold | 17,38 % | 0,54 | −53,07 % | 63 USD |

**Kernaussage:** Buy & Hold erzielt die höchste absolute Rendite, aber um den
Preis eines Drawdowns von −53 %. Der Bot liefert einen Großteil der Rendite bei
deutlich geringerem Risiko. Das **ML-Spacing schlägt das statische Spacing**
risikoadjustiert (höherer Sharpe, geringerer Drawdown) bei gleichzeitig ~31 %
weniger Gebühren und weniger Trades.

![Equity-Kurve](equity_curve.png)

## Disclaimer

Dieses Projekt dient ausschließlich Forschungs- und Ausbildungszwecken und ist
keine Anlageberatung. Handel mit Kryptowährungen ist hochriskant. Für diese
Abgabe läuft der Bot ausschließlich im **Alpaca Paper-Modus** (`ALPACA_PAPER=True`).
