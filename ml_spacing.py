"""
ml_spacing.py — ML-Modul zur taeglichen Vorhersage des Grid-Spacings in %.

Idee: Statt fuer jeden Tag dutzende Backtests zu fahren, lernt ein Modell
DIREKT ein sinnvolles Grid-Spacing in Prozent. Die Features beschreiben
nur die Vergangenheit, das Target beschreibt direkt das gewuenschte
Spacing der nahen Zukunft (naechste 3 Tage).

Diese Datei darf laut Vorgabe etwas komplexer sein als der Rest. Zwei
Fallback-Ebenen sichern den Live-Betrieb ab:
  1. Modell-Fallback: Linear Regression, falls der Decision Tree nicht
     trainiert werden kann.
  2. Spacing-Fallback: GRID_SPACING_PCT (0.5%), falls gar kein Modell
     geladen werden kann.
"""

import os

import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import config
import sma_filter


# Reihenfolge der Features ist fix — Training und Prediction MUESSEN
# exakt dieselbe Spaltenreihenfolge verwenden.
FEATURE_COLUMNS = [
    "atr_7",
    "atr_14",
    "atr_30",
    "volatility_14",
    "sma_distance",
    "volume_ratio",
]


def compute_features(df):
    """
    Berechnet alle ML-Input-Features (nur aus vergangenen Daten).

    Input:  df mit open/high/low/close/volume und DatetimeIndex
    Output: df mit zusaetzlichen Feature-Spalten (FEATURE_COLUMNS) + 'sma'

    WARUM ATR in Prozent: BTC ist von ~4k (2017) auf ~70k (2024)
    gestiegen. Ein absoluter ATR waere ueber die Jahre voellig
    unterschiedlich skaliert und das auf 2017-2021 trainierte Modell
    waere 2022-2024 unbrauchbar. Wir teilen die True Range durch den
    Vortages-Close -> stationaere, regime-uebergreifend vergleichbare %.
    """
    df = sma_filter.add_sma(df)

    # True Range = groesste der drei Spannen (klassische ATR-Definition),
    # hier relativ zum Vortages-Close ausgedrueckt (Prozent).
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr_pct = true_range / prev_close

    df["atr_7"] = tr_pct.rolling(7).mean()
    df["atr_14"] = tr_pct.rolling(14).mean()
    df["atr_30"] = tr_pct.rolling(30).mean()

    # Standardabweichung der taeglichen Returns (14 Tage) — schon relativ.
    daily_return = df["close"].pct_change()
    df["volatility_14"] = daily_return.rolling(14).std()

    # Abstand des Preises zum 120-SMA in % (Trendlage).
    df["sma_distance"] = df["close"] / df["sma"] - 1.0

    # Aktuelles Volumen relativ zum 30-Tage-Schnitt (Aktivitaet/Interesse).
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(30).mean()

    return df


def compute_target(df, forward=config.ML_TARGET_FORWARD_DAYS,
                   levels=config.GRID_LEVELS_PER_RANGE):
    """
    Berechnet das Label OPTIMAL_SPACING_FORWARD (direkt in % Spacing).

    Input:  df mit high/low/close, forward (Tage), levels (Grid-Stufen)
    Output: df mit Spalte 'target'

    Formel: (max High - min Low der NAECHSTEN 3 Tage) / Close / Levels.
    Das ist der natuerliche kommende Preisbereich, in ~10 Grid-Stufen
    aufgeteilt, als Prozentsatz. Der Blick in die Zukunft (shift) ist
    NUR im Target erlaubt — die Features bleiben rein vergangenheits-
    basiert (kein Look-ahead-Bias). Die letzten 'forward' Zeilen werden
    dadurch NaN und spaeter per dropna() entfernt.
    """
    df = df.copy()
    fwd_high = df["high"].rolling(forward).max().shift(-forward)
    fwd_low = df["low"].rolling(forward).min().shift(-forward)
    df["target"] = (fwd_high - fwd_low) / df["close"] / levels
    return df


def build_dataset(df):
    """
    Baut die saubere Trainings-/Auswertungsmatrix.

    Input:  df mit OHLCV
    Output: (X, y, clean_df) — X = Features, y = Target, beide ohne NaN.

    Die fuehrenden NaNs (SMA/ATR-Fenster noch nicht voll) und die
    abschliessenden NaNs (Target-Zukunft fehlt) werden mit dropna()
    entfernt — so trainiert das Modell nur auf vollstaendigen Zeilen.
    """
    df = compute_features(df)
    df = compute_target(df)
    clean = df[FEATURE_COLUMNS + ["target"]].dropna()
    return clean[FEATURE_COLUMNS], clean["target"], clean


def train_model(df, save_path=config.MODEL_PATH):
    """
    Trainiert das Spacing-Modell und speichert es als .pkl.

    Input:  df mit OHLCV (volle Historie), save_path fuer das Modell
    Output: das trainierte Modell-Objekt (Decision Tree oder Linear)

    Sauberer Split: Training NUR auf In-Sample (2017-2021), Bewertung
    auf Out-of-Sample (2022-2024). So sehen wir sofort, ob das Modell
    generalisiert oder nur auswendig lernt.
    """
    X, y, clean = build_dataset(df)

    train_mask = (clean.index >= config.IN_SAMPLE_START) & \
                 (clean.index <= config.IN_SAMPLE_END)
    test_mask = (clean.index >= config.OUT_SAMPLE_START) & \
                (clean.index <= config.OUT_SAMPLE_END)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Trainingszeilen (In-Sample {config.IN_SAMPLE_START[:4]}-"
          f"{config.IN_SAMPLE_END[:4]}):   {len(X_train)}")
    print(f"Testzeilen     (Out-of-Sample {config.OUT_SAMPLE_START[:4]}-"
          f"{config.OUT_SAMPLE_END[:4]}): {len(X_test)}")

    # Primaer: Decision Tree. WARUM diese Hyperparameter:
    #   max_depth=4        -> flach, erklaerbar, kaum Overfitting
    #   min_samples_leaf=20 -> jedes Blatt stuetzt sich auf >=20 Tage,
    #                          kein Anpassen an einzelne Ausreisser-Tage
    #   random_state=42    -> reproduzierbares Ergebnis fuer den Bericht
    try:
        model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20,
                                      random_state=42)
        model.fit(X_train, y_train)
        model_name = "DecisionTreeRegressor"
    except Exception as e:
        # Fallback-Modell, falls der Baum aus irgendeinem Grund scheitert.
        print(f"Decision Tree fehlgeschlagen ({e}) -> Fallback LinearRegression")
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_name = "LinearRegression"

    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test)) \
        if len(X_test) else float("nan")

    print(f"\nModell: {model_name}")
    print(f"MAE In-Sample:     {train_mae:.4%} Spacing-Fehler")
    print(f"MAE Out-of-Sample: {test_mae:.4%} Spacing-Fehler")

    # Erklaerbarkeit: welche Features treiben die Vorhersage (nur Tree).
    if hasattr(model, "feature_importances_"):
        print("\nFeature-Wichtigkeit:")
        for name, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
            print(f"  {name:15s} {imp:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, save_path)
    print(f"\nModell gespeichert: {save_path}")
    return model


def load_model(path=config.MODEL_PATH):
    """
    Laedt das gespeicherte Modell. Gibt None zurueck, wenn keine Datei
    existiert oder das Laden scheitert — der Aufrufer faellt dann auf
    GRID_SPACING_PCT zurueck.
    """
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        return bundle["model"]
    except Exception:
        return None


def predict_spacing(model, features):
    """
    Sagt das Grid-Spacing in % voraus und begrenzt es auf den sicheren
    Bereich.

    Input:  model (oder None), features = dict mit FEATURE_COLUMNS
    Output: Spacing als float (z.B. 0.012 = 1.2%)

    Clip-Logik: ML_SPACING_MIN ist eine HARTE Untergrenze (Gebuehren-
    schutz). ML_SPACING_MAX (5%) ist nur ein weites Soft-Limit gegen
    Ausreisser durch Datenfehler — im Normalfall (0.5-2%) greift es nie.
    Ohne Modell: Fallback auf den statischen GRID_SPACING_PCT.
    """
    if model is None:
        return config.GRID_SPACING_PCT

    X = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]],
                     columns=FEATURE_COLUMNS)
    raw = float(model.predict(X)[0])
    return float(np.clip(raw, config.ML_SPACING_MIN, config.ML_SPACING_MAX))


def latest_features(df):
    """
    Liefert die Feature-Werte der letzten vollstaendigen Zeile als dict.

    Input:  df mit OHLCV
    Output: dict {feature_name: wert} fuer die juengste Kerze, oder None,
            wenn noch nicht genug Historie fuer alle Features da ist.
    """
    feat = compute_features(df)
    feat = feat[FEATURE_COLUMNS].dropna()
    if feat.empty:
        return None
    return feat.iloc[-1].to_dict()


def should_rebuild(current_spacing, new_spacing,
                   threshold=config.ML_REBUILD_THRESHOLD):
    """
    Entscheidet, ob das Grid wegen geaendertem Spacing neu gebaut wird.

    Input:  current_spacing (aktuell aktiv), new_spacing (ML-Vorhersage)
    Output: True, wenn die relative Abweichung > threshold (z.B. 20%).

    Verhindert staendiges Neuaufbauen bei kleinen Schwankungen — nur bei
    deutlicher Aenderung lohnt der Aufwand (und die Gebuehren) des Rebuilds.
    """
    if not current_spacing:
        return True
    return abs(new_spacing - current_spacing) / current_spacing > threshold


if __name__ == "__main__":
    print("=== ml_spacing.py Test ===")
    import data

    df = data.load_data()

    # 1) Modell trainieren (sauberer In-/Out-of-Sample-Split) und speichern.
    model = train_model(df)

    # 2) Modell zurueckladen (wie spaeter im Live-Betrieb).
    loaded = load_model()
    assert loaded is not None, "Modell sollte ladbar sein"

    # 3) Vorhersage fuer die juengste Kerze + Begruendung anzeigen.
    feats = latest_features(df)
    raw = float(loaded.predict(
        pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS]],
                     columns=FEATURE_COLUMNS))[0])
    spacing = predict_spacing(loaded, feats)

    print("\n--- Vorhersage fuer juengste Kerze ---")
    print("Features:")
    for name in FEATURE_COLUMNS:
        print(f"  {name:15s} {feats[name]:+.4f}")
    print(f"\nRoh-Vorhersage Spacing: {raw:.4%}")
    print(f"Geclippt [{config.ML_SPACING_MIN:.1%}, "
          f"{config.ML_SPACING_MAX:.1%}]: {spacing:.4%}")

    # Begruendung: woher kommt das Spacing.
    if raw < config.ML_SPACING_MIN:
        grund = "Roh-Wert unter Gebuehren-Untergrenze -> auf MIN angehoben"
    elif raw > config.ML_SPACING_MAX:
        grund = "Roh-Wert ueber Soft-Limit (Ausreisser?) -> auf MAX gekappt"
    else:
        grund = "Roh-Wert im Normalbereich -> unveraendert uebernommen"
    print(f"Begruendung: {grund}")

    # 4) Fallback ohne Modell -> statischer Wert.
    assert predict_spacing(None, feats) == config.GRID_SPACING_PCT, \
        "Ohne Modell muss der statische Fallback greifen"
    print(f"\nOK  Fallback ohne Modell = {config.GRID_SPACING_PCT:.3%}")

    # 5) Rebuild-Schwelle pruefen.
    assert should_rebuild(0.005, 0.007) is True, ">40% Aenderung = Rebuild"
    assert should_rebuild(0.010, 0.0105) is False, "5% Aenderung = kein Rebuild"
    print("OK  Rebuild-Schwelle (20%) funktioniert.")

    print("\nAlle ml_spacing-Tests bestanden.")
