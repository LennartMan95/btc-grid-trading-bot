"""
sma_filter.py — Richtungsfilter und dynamischer Stop-Loss ueber den 120-SMA.

Der 120-Tage-SMA (UTC-Tagesschluss) entscheidet, ob der Bot ein Long-Grid
faehrt oder pausiert. KEIN Short-Grid. Zusaetzlich liefert dieses Modul den
taeglich nachgezogenen Stop-Loss-Preis.

Kernideen:
  - Entry nur bei klarem Momentum: mehrere Kerzen deutlich UEBER dem SMA.
  - Exit/Pause bei klarer Schwaeche: mehrere Kerzen deutlich UNTER dem SMA.
  - Der Buffer (+/-1%) filtert das "Zittern" direkt am SMA heraus.
"""

import pandas as pd

import config


def add_sma(df, period=config.SMA_PERIOD):
    """
    Fuegt dem DataFrame eine 'sma'-Spalte hinzu (gleitender Mittelwert
    des Close ueber 'period' Tage).

    Input:  df mit Spalte 'close', period (Default 120)
    Output: derselbe df mit zusaetzlicher Spalte 'sma' (erste period-1
            Werte sind NaN, da das Fenster noch nicht voll ist).
    """
    df = df.copy()
    df["sma"] = df["close"].rolling(window=period).mean()
    return df


def is_entry_confirmed(df, i, n=config.CONFIRMATION_CANDLES_ENTRY,
                       buffer=config.SMA_ENTRY_BUFFER):
    """
    Prueft, ob an Position i ein Long-Einstieg bestaetigt ist.

    Input:  df mit 'close' und 'sma', Integer-Position i, n Kerzen, buffer
    Output: True, wenn die letzten n Kerzen JEWEILS mindestens (1+buffer)
            ueber ihrem SMA geschlossen haben.

    WARUM der Buffer: Nicht der erste Hauch ueber dem SMA zaehlt, sondern
    erst ein Schluss >= SMA*1.01. So startet der Bot nur bei echtem
    Momentum und nicht beim Rauschen direkt an der SMA-Linie.
    """
    if i < n - 1:
        return False
    for j in range(i - n + 1, i + 1):
        close = df["close"].iloc[j]
        sma = df["sma"].iloc[j]
        if pd.isna(sma):
            return False
        if close < sma * (1 + buffer):
            return False
    return True


def is_exit_confirmed(df, i, n=config.CONFIRMATION_CANDLES_EXIT,
                      buffer=config.SMA_EXIT_BUFFER):
    """
    Prueft, ob an Position i ein Ausstieg / eine Pause bestaetigt ist.

    Input:  df mit 'close' und 'sma', Integer-Position i, n Kerzen, buffer
    Output: True, wenn die letzten n Kerzen JEWEILS mindestens (1-buffer)
            unter ihrem SMA geschlossen haben.

    Symmetrisch zum Entry: erst ein klarer Schluss <= SMA*0.99 ueber
    mehrere Kerzen loest die Pause aus, nicht ein kurzes Antippen.
    """
    if i < n - 1:
        return False
    for j in range(i - n + 1, i + 1):
        close = df["close"].iloc[j]
        sma = df["sma"].iloc[j]
        if pd.isna(sma):
            return False
        if close > sma * (1 - buffer):
            return False
    return True


def stop_loss_price(sma_value):
    """
    Berechnet den dynamischen Stop-Loss-Preis aus dem aktuellen SMA.

    Input:  sma_value (aktueller 120-SMA)
    Output: SMA * STOP_LOSS_SMA_FACTOR (z.B. SMA * 0.99)

    WARUM SMA-basiert: Der Stop zieht im Bullenmarkt automatisch mit dem
    SMA nach oben und schuetzt so aufgelaufene Gewinne, ohne dass wir ihn
    manuell nachfuehren muessen. Der Faktor 0.99 laesst 1% Puffer.
    """
    return sma_value * config.STOP_LOSS_SMA_FACTOR


def is_stop_loss_hit(df, i):
    """
    Prueft, ob an Position i der Stop-Loss ausgeloest wurde.

    Input:  df mit 'close' und 'sma', Integer-Position i
    Output: True, wenn der Close UNTER dem Stop-Loss-Preis liegt UND der
            Ausstieg ueber CONFIRMATION_CANDLES_EXIT Kerzen bestaetigt ist.

    Doppelte Bedingung: nur ein bestaetigter Bruch (nicht ein einzelner
    Docht) unter SMA*0.99 stoppt den Bot — vermeidet Fehl-Stops.
    """
    sma = df["sma"].iloc[i]
    if pd.isna(sma):
        return False
    close = df["close"].iloc[i]
    below_stop = close < stop_loss_price(sma)
    return below_stop and is_exit_confirmed(df, i)


def trend_status(df, i, grid_active):
    """
    Bestimmt fuer Position i den naechsten Soll-Zustand des Bots.

    Input:  df mit 'close'/'sma', Position i, grid_active (aktueller Zustand)
    Output: dict mit Signalen und dem neuen Soll-Zustand 'active'.

    Diese Funktion buendelt die Logik fuer backtest.py und monitor.py:
      - Ist das Grid inaktiv -> aktiv werden, wenn Entry bestaetigt.
      - Ist das Grid aktiv   -> pausieren, wenn Exit ODER Stop-Loss greift.
    """
    sma = df["sma"].iloc[i]
    close = df["close"].iloc[i]
    entry = is_entry_confirmed(df, i)
    exit_ = is_exit_confirmed(df, i)
    stop = is_stop_loss_hit(df, i)

    if grid_active:
        active = not (exit_ or stop)
    else:
        active = entry

    return {
        "date": df.index[i],
        "close": close,
        "sma": sma,
        "stop_loss_price": stop_loss_price(sma) if pd.notna(sma) else None,
        "entry_confirmed": entry,
        "exit_confirmed": exit_,
        "stop_loss_hit": stop,
        "active": active,
    }


if __name__ == "__main__":
    print("=== sma_filter.py Test ===")

    # --- Teil 1: Synthetische Asserts mit bekanntem Ergebnis -------------

    # Szenario A: 120 Tage flach bei 100 -> close == sma -> KEIN Entry
    # (close 100 < sma*1.01 = 101), und KEIN Exit (close nicht unter SMA).
    flat = pd.DataFrame({"close": [100.0] * 122})
    flat = add_sma(flat)
    assert is_entry_confirmed(flat, 121) is False, "Flat darf keinen Entry geben"
    assert is_exit_confirmed(flat, 121) is False, "Flat darf keinen Exit geben"
    print("OK  Flacher Markt: weder Entry noch Exit (kein Zittern).")

    # Szenario B: flach bei 100, dann 2 Tage bei 110 -> Entry bestaetigt.
    up = pd.DataFrame({"close": [100.0] * 120 + [110.0, 110.0]})
    up = add_sma(up)
    assert is_entry_confirmed(up, 121) is True, "Klarer Anstieg muss Entry geben"
    print("OK  +10% ueber 2 Kerzen: Entry bestaetigt.")

    # Szenario C: flach bei 100, dann 2 Tage bei 88 -> Exit + Stop-Loss.
    down = pd.DataFrame({"close": [100.0] * 120 + [88.0, 88.0]})
    down = add_sma(down)
    assert is_exit_confirmed(down, 121) is True, "Klarer Abfall muss Exit geben"
    assert is_stop_loss_hit(down, 121) is True, "Bruch unter SMA*0.99 = Stop"
    print("OK  -12% ueber 2 Kerzen: Exit + Stop-Loss ausgeloest.")

    # Szenario D: nur 1 Tag ueber +1% reicht NICHT (Confirmation = 2).
    one = pd.DataFrame({"close": [100.0] * 121 + [110.0]})
    one = add_sma(one)
    assert is_entry_confirmed(one, 121) is False, "Eine Kerze darf nicht reichen"
    print("OK  Nur 1 Kerze ueber Buffer: kein Entry (Confirmation greift).")

    # --- Teil 2: Status auf echten Daten ---------------------------------
    import data

    df = data.load_data()
    df = add_sma(df)

    i = len(df) - 1
    status = trend_status(df, i, grid_active=False)
    print("\nAktueller Trendstatus (letzter Tag):")
    print(f"  Datum:            {status['date'].date()}")
    print(f"  Close:            {status['close']:.2f}")
    print(f"  SMA-120:          {status['sma']:.2f}")
    print(f"  Abstand zu SMA:   {(status['close'] / status['sma'] - 1) * 100:+.2f}%")
    print(f"  Stop-Loss-Preis:  {status['stop_loss_price']:.2f}")
    print(f"  Entry bestaetigt: {status['entry_confirmed']}")
    print(f"  Exit bestaetigt:  {status['exit_confirmed']}")
    print(f"  Stop-Loss-Hit:    {status['stop_loss_hit']}")
    print(f"  -> Grid aktiv?    {status['active']}")

    print("\nAlle sma_filter-Tests bestanden.")
