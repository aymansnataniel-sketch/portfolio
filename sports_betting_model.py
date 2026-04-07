#!/usr/bin/env python3
"""Modelo predictivo para apuestas deportivas.

Entrena un clasificador para estimar la probabilidad de victoria local y
calcula si existe valor esperado positivo (EV+) comparando contra cuotas.

Uso:
    python sports_betting_model.py --data datos.csv --target home_win
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class Metrics:
    auc: float
    logloss: float
    brier: float


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    base_model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.03,
        max_iter=350,
        l2_regularization=0.2,
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", calibrated),
        ]
    )


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> Metrics:
    return Metrics(
        auc=roc_auc_score(y_true, y_prob),
        logloss=log_loss(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
    )


def add_value_edges(df: pd.DataFrame, prob_col: str = "prob_home_win") -> pd.DataFrame:
    required = {"odds_home", "odds_away", prob_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para EV: {sorted(missing)}")

    out = df.copy()
    out["ev_home"] = (out[prob_col] * out["odds_home"]) - 1.0
    out["ev_away"] = ((1.0 - out[prob_col]) * out["odds_away"]) - 1.0
    out["best_side"] = np.where(out["ev_home"] >= out["ev_away"], "home", "away")
    out["best_ev"] = out[["ev_home", "ev_away"]].max(axis=1)
    out["edge_flag"] = out["best_ev"] > 0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelo predictivo para apuestas deportivas")
    parser.add_argument("--data", type=Path, required=True, help="CSV ordenado por fecha")
    parser.add_argument("--target", default="home_win", help="Variable objetivo binaria (0/1)")
    parser.add_argument("--pred-out", type=Path, default=Path("predicciones.csv"))
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=["date", "match_id"],
        help="Columnas a excluir del entrenamiento",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {args.target}")

    y = df[args.target].astype(int).values
    feature_df = df.drop(columns=[c for c in [args.target, *args.drop_cols] if c in df.columns])

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    model = build_pipeline(numeric_cols, categorical_cols)

    tscv = TimeSeriesSplit(n_splits=5)
    probs = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in tscv.split(feature_df):
        X_train, X_test = feature_df.iloc[train_idx], feature_df.iloc[test_idx]
        y_train = y[train_idx]
        model.fit(X_train, y_train)
        probs[test_idx] = model.predict_proba(X_test)[:, 1]

    valid_mask = probs > 0
    fold_metrics = evaluate(y[valid_mask], probs[valid_mask])

    print("=== Métricas out-of-sample (TimeSeriesSplit) ===")
    print(f"AUC:      {fold_metrics.auc:.4f}")
    print(f"LogLoss:  {fold_metrics.logloss:.4f}")
    print(f"Brier:    {fold_metrics.brier:.4f}")

    output = df.copy()
    output["prob_home_win"] = probs

    if {"odds_home", "odds_away"}.issubset(df.columns):
        output = add_value_edges(output, prob_col="prob_home_win")
        n_edges = int(output["edge_flag"].sum())
        print(f"Partidos con EV+ detectados: {n_edges}")

    output.to_csv(args.pred_out, index=False)
    print(f"Archivo de predicciones guardado en: {args.pred_out}")


if __name__ == "__main__":
    main()
