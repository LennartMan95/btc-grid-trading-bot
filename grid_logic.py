"""
grid_logic.py — Die reine Grid-Mechanik: Level, Orders, Fills, Rebuild.

Dieses Modul ist die zentrale, wiederverwendbare Logik des Grids. Es kennt
weder SMA-Filter noch ML — es verwaltet nur:
  - die geometrischen Grid-Level,
  - die offenen Buy-/Sell-Orders (inkl. Menge und Kostenbasis),
  - das Fill-Handling (nach jedem Fill sofort die Gegenorder setzen),
  - das Aufloesen (close) und Umsetzen (rebuild) des Grids.

backtest.py nutzt es fuer die Simulation, monitor.py/execution.py spaeter
fuer den Live-Betrieb. Der gesamte Zustand liegt in EINEM dict ('state'),
das auch in state.json gesichert werden kann.

Order-Format (einheitlich fuer Buy und Sell):
  {"price": Level-Preis, "qty": Menge BTC, "cost": Einstandskosten in USDT}
"""

import config


def new_state(capital):
    """
    Erzeugt einen frischen, leeren Grid-Zustand.

    Input:  capital (Startkapital in USDT)
    Output: state-dict mit Cash, leeren Order-Buechern und Zaehlern.
    """
    return {
        "capital": capital,
        "cash": capital,
        "btc": 0.0,
        "fees": 0.0,
        "realized": 0.0,
        "trades": 0,
        "active": False,
        "levels": [],
        "buy_orders": {},     # level_index -> {price, qty, cost}
        "sell_orders": {},    # level_index -> {price, qty, cost}
        "per_level": 0.0,
        "spacing": config.GRID_SPACING_PCT,
        "start_price": 0.0,
    }


def build_grid_levels(lower, upper, spacing):
    """
    Erzeugt geometrisch gestaffelte Grid-Level von 'lower' aufwaerts.

    Input:  lower (Untergrenze), upper (Obergrenze), spacing (z.B. 0.005)
    Output: Liste der Grid-Preise.

    Geometrisch (jeder Level = vorheriger * (1+spacing)), weil das Spacing
    ein Prozentwert ist — so liefert jeder Round-Trip denselben %-Profit.
    Gedeckelt durch MAX_GRID_COUNT.
    """
    levels = []
    p = lower
    while p <= upper and len(levels) < config.MAX_GRID_COUNT:
        levels.append(p)
        p *= (1 + spacing)
    return levels


def _make_buy(price, per_level):
    """Baut ein Buy-Order-dict fuer einen Level-Preis (Menge aus per_level)."""
    return {"price": price, "qty": per_level / price, "cost": per_level}


def open_grid(state, close, sma, spacing):
    """
    Baut ein frisches Grid am aktuellen Preis und setzt die Startorders.

    Input:  state (dict, wird veraendert), close, sma, spacing
    Output: nichts (state wird in-place aktualisiert).

    Seeding gemaess Vorgabe:
      - Buy-Orders auf allen Leveln UNTER dem Close (Dips kaufen).
      - Sell-Orders auf allen Leveln UEBER dem Close — dafuer kaufen wir
        eine Anfangs-Position zum Close (sonst gaebe es nichts zu
        verkaufen). Das macht den Bot bewusst long-orientiert.
    """
    lower = sma * config.LOWER_PRICE_SMA_FACTOR     # SMA * 0.99
    upper = close * config.UPPER_PRICE_FACTOR       # close * 10 (theoret. Limit)
    levels = build_grid_levels(lower, upper, spacing)

    state["levels"] = levels
    state["spacing"] = spacing
    state["start_price"] = close
    state["buy_orders"] = {}
    state["sell_orders"] = {}

    invested = state["capital"] * config.CAPITAL_INVESTED
    n_levels = max(len(levels), 1)
    per_level = invested / n_levels   # gleicher Quote-Betrag pro Level
    state["per_level"] = per_level

    for idx, price in enumerate(levels):
        if price < close:
            # Buy-Order: Cash bleibt liegen, wird erst beim Fill ausgegeben.
            state["buy_orders"][idx] = _make_buy(price, per_level)
        elif price > close:
            # Sell-Order: Anfangs-Inventar jetzt zum Close kaufen.
            if state["cash"] >= per_level:
                qty = per_level / close
                fee = per_level * config.MAKER_FEE
                state["cash"] -= (per_level + fee)
                state["btc"] += qty
                state["fees"] += fee
                state["sell_orders"][idx] = {
                    "price": price,
                    "qty": qty,
                    "cost": per_level + fee,   # Einstandskosten inkl. Gebuehr
                }

    state["active"] = True


def close_grid(state, close):
    """
    Loest das Grid auf: alle Orders canceln, gesamtes BTC zum Close
    verkaufen (Taker-Gebuehr). Wird bei Exit und Stop-Loss genutzt.

    Input:  state (dict, wird veraendert), close
    Output: nichts.
    """
    if state["btc"] > 0:
        revenue = state["btc"] * close
        fee = revenue * config.TAKER_FEE
        state["cash"] += (revenue - fee)
        state["fees"] += fee
        state["btc"] = 0.0
    state["buy_orders"] = {}
    state["sell_orders"] = {}
    state["active"] = False


def rebuild_grid(state, close, sma, new_spacing):
    """
    Setzt das Grid auf ein neues Spacing um, OHNE das Inventar zu verkaufen.

    Input:  state (dict, wird veraendert), close, sma, new_spacing
    Output: nichts.

    WARUM kein Liquidieren: Ein Rebuild ist in der Praxis nur das Canceln
    und Neu-Platzieren von Limit-Orders — das kostet KEINE Gebuehren
    (Gebuehren fallen nur bei echten Fills an). Das gehaltene BTC-Inventar
    inkl. seiner Kostenbasis bleibt erhalten und wird gleichmaessig auf die
    neuen oberen Sell-Level verteilt. So entstehen beim Rebuild weder
    kuenstliche Liquidations-Gebuehren noch verloren gegangene Grid-Profite.
    """
    btc_total = state["btc"]
    cost_total = sum(o["cost"] for o in state["sell_orders"].values())

    lower = sma * config.LOWER_PRICE_SMA_FACTOR
    upper = close * config.UPPER_PRICE_FACTOR
    levels = build_grid_levels(lower, upper, new_spacing)

    state["levels"] = levels
    state["spacing"] = new_spacing
    state["buy_orders"] = {}
    state["sell_orders"] = {}

    invested = state["capital"] * config.CAPITAL_INVESTED
    n_levels = max(len(levels), 1)
    state["per_level"] = invested / n_levels

    lower_idx = [idx for idx, p in enumerate(levels) if p < close]
    upper_idx = [idx for idx, p in enumerate(levels) if p > close]

    # Buy-Orders unter dem Preis (Cash-gedeckt, fuellen erst spaeter).
    for idx in lower_idx:
        state["buy_orders"][idx] = _make_buy(levels[idx], state["per_level"])

    # Bestehendes Inventar gleichmaessig auf die oberen Sell-Level verteilen.
    # Kostenbasis bleibt in Summe erhalten -> keine kuenstliche P&L.
    if btc_total > 0 and upper_idx:
        qty_each = btc_total / len(upper_idx)
        cost_each = cost_total / len(upper_idx)
        for idx in upper_idx:
            state["sell_orders"][idx] = {
                "price": levels[idx],
                "qty": qty_each,
                "cost": cost_each,
            }
    # state["btc"] und state["cash"] bleiben unveraendert (keine Fills).


def handle_buy_fill(state, idx):
    """
    Verarbeitet einen gefuellten Buy auf Level 'idx' und setzt die Gegenorder.

    Input:  state (dict, wird veraendert), idx (Level-Index der Buy-Order)
    Output: True, wenn gefuellt; False, wenn Cash nicht reichte.

    Nach dem Kauf wird sofort eine Sell-Order eine Stufe hoeher gesetzt
    (klassische Grid-Mechanik: tief kaufen, eine Stufe hoeher verkaufen).
    """
    order = state["buy_orders"].get(idx)
    if order is None:
        return False
    cost = order["cost"]
    if state["cash"] < cost:
        return False   # nicht genug Cash -> Order kann nicht fuellen
    qty = order["qty"]
    fee = cost * config.MAKER_FEE
    state["cash"] -= (cost + fee)
    state["btc"] += qty
    state["fees"] += fee
    del state["buy_orders"][idx]

    up = idx + 1
    if up < len(state["levels"]):
        state["sell_orders"][up] = {
            "price": state["levels"][up],
            "qty": qty,
            "cost": cost + fee,
        }
    return True


def handle_sell_fill(state, idx):
    """
    Verarbeitet einen gefuellten Sell auf Level 'idx' und setzt die Gegenorder.

    Input:  state (dict, wird veraendert), idx (Level-Index der Sell-Order)
    Output: True, wenn gefuellt; False, wenn keine Order existierte.

    Realisiert den Grid-Profit dieses Round-Trips (nach Gebuehren) und setzt
    sofort wieder eine Buy-Order eine Stufe tiefer.
    """
    order = state["sell_orders"].get(idx)
    if order is None:
        return False
    revenue = order["qty"] * order["price"]
    fee = revenue * config.TAKER_FEE
    proceeds = revenue - fee
    state["cash"] += proceeds
    state["btc"] -= order["qty"]
    state["fees"] += fee
    state["realized"] += (proceeds - order["cost"])
    state["trades"] += 1
    del state["sell_orders"][idx]

    down = idx - 1
    if down >= 0:
        state["buy_orders"][down] = _make_buy(state["levels"][down],
                                              state["per_level"])
    return True


def simulate_day_fills(state, low, high):
    """
    Backtest-Helfer: spielt die Fills eines Tages aus Low/High durch.

    Input:  state (dict), low, high der Tageskerze
    Output: nichts (state wird in-place aktualisiert).

    Reihenfolge "erst Low, dann High": zuerst fuellen alle erreichten
    Buy-Orders (Preis dropt bis 'low', oben zuerst), danach alle erreichten
    Sell-Orders (Preis steigt bis 'high', unten zuerst). Eine in der
    Low-Phase frisch gesetzte Sell-Order kann so am selben Tag fuellen.

    Hinweis: Im Live-Betrieb werden stattdessen die echten Fills der Boerse
    ueber handle_buy_fill / handle_sell_fill verarbeitet.
    """
    # --- Phase 1: Low -> Buy-Orders mit price >= low (oben zuerst).
    hit_buys = [idx for idx, o in state["buy_orders"].items() if low <= o["price"]]
    for idx in sorted(hit_buys, key=lambda k: state["buy_orders"][k]["price"],
                      reverse=True):
        handle_buy_fill(state, idx)

    # --- Phase 2: High -> Sell-Orders mit price <= high (unten zuerst).
    hit_sells = [idx for idx, o in state["sell_orders"].items() if high >= o["price"]]
    for idx in sorted(hit_sells, key=lambda k: state["sell_orders"][k]["price"]):
        handle_sell_fill(state, idx)


def grid_equity(state, price):
    """Aktueller Gesamtwert: Cash + Inventarwert (BTC * Preis)."""
    return state["cash"] + state["btc"] * price


if __name__ == "__main__":
    print("=== grid_logic.py Test ===")

    # Synthetisches Szenario: SMA=100, Start-Close=120, Spacing 2%.
    state = new_state(10000.0)
    open_grid(state, close=120.0, sma=100.0, spacing=0.02)
    n_buy = len(state["buy_orders"])
    n_sell = len(state["sell_orders"])
    print(f"Grid gebaut: {len(state['levels'])} Level, "
          f"{n_buy} Buy-Orders, {n_sell} Sell-Orders")
    assert n_buy > 0 and n_sell > 0, "Seeding muss Buy- und Sell-Orders erzeugen"

    # Invariante: gehaltenes BTC == Summe der Mengen aller Sell-Orders.
    btc_sum = sum(o["qty"] for o in state["sell_orders"].values())
    assert abs(state["btc"] - btc_sum) < 1e-9, "BTC muss zu Sell-Orders passen"
    print("OK  Invariante BTC == Summe Sell-Mengen.")

    # Ein Tag mit kraeftigem Ausschlag nach unten und oben -> Round-Trips.
    trades_before = state["trades"]
    simulate_day_fills(state, low=100.0, high=140.0)
    print(f"Trades nach volatilem Tag: {state['trades']} "
          f"(+{state['trades'] - trades_before})")
    assert state["trades"] > trades_before, "Volatiler Tag muss Trades ausloesen"
    assert state["realized"] > 0, "Round-Trips muessen Profit realisieren"
    print(f"OK  Realisierter Grid-Profit: {state['realized']:.2f} USDT, "
          f"Gebuehren: {state['fees']:.2f} USDT")

    # Rebuild auf engeres Spacing: Inventar bleibt, keine Liquidation.
    btc_pre = state["btc"]
    fees_pre = state["fees"]
    rebuild_grid(state, close=130.0, sma=100.0, new_spacing=0.01)
    assert abs(state["btc"] - btc_pre) < 1e-9, "Rebuild darf Inventar nicht verkaufen"
    assert abs(state["fees"] - fees_pre) < 1e-9, "Rebuild darf keine Gebuehren kosten"
    print("OK  Rebuild ohne Liquidation und ohne Gebuehren.")

    # Close: alles zu Cash, keine offenen Orders mehr.
    close_grid(state, close=130.0)
    assert state["btc"] == 0.0, "Nach close darf kein BTC mehr gehalten werden"
    assert not state["active"], "Nach close ist das Grid inaktiv"
    print(f"OK  Grid geschlossen, Endkapital: {state['cash']:.2f} USDT")

    print("\nAlle grid_logic-Tests bestanden.")
