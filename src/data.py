"""Carga e limpeza da base de Pokémon."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "is_legendary"

FEATURE_LABELS: dict[str, str] = {
    "hp": "HP",
    "attack": "Ataque",
    "defense": "Defesa",
    "sp_attack": "Ataque Especial",
    "sp_defense": "Defesa Especial",
    "speed": "Velocidade",
    "height_m": "Altura (m)",
    "weight_kg": "Peso (kg)",
    "base_total": "Total Base",
}

ALL_FEATURES: list[str] = list(FEATURE_LABELS.keys())

COMBAT_STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]

PRESETS: dict[str, list[str]] = {
    "aula": COMBAT_STATS.copy(),
    "fisicos": ["height_m", "weight_kg"],
    "agregado": ["base_total"],
    "completo": ALL_FEATURES.copy(),
}

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "base_pokemon.csv"


def _ensure_base_total(df: pd.DataFrame) -> pd.DataFrame:
    if "base_total" not in df.columns:
        missing = [c for c in COMBAT_STATS if c not in df.columns]
        if missing:
            raise ValueError(
                "A coluna base_total não existe e não foi possível calculá-la. "
                f"Faltam os atributos: {', '.join(missing)}."
            )
        df = df.copy()
        df["base_total"] = df[COMBAT_STATS].sum(axis=1)
    return df


def load_pokemon_data(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Carrega o CSV, calcula base_total se ausente e elimina linhas com NaN.

    O dropna é aplicado sobre as colunas usadas na modelagem (atributos + alvo),
    para que o treino ocorra apenas com registros 100% preenchidos.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo '{path}' não encontrado. Coloque base_pokemon.csv na raiz do projeto."
        )

    df = pd.read_csv(path)
    rows_raw = len(df)
    df = _ensure_base_total(df)

    if TARGET not in df.columns:
        raise ValueError("A coluna alvo 'is_legendary' não está presente no CSV.")

    missing_features = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            "O CSV não contém todas as colunas utilizadas na modelagem: "
            + ", ".join(missing_features)
        )

    used_cols = ALL_FEATURES + [TARGET]
    for col in used_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Elimina qualquer Pokémon com valor ausente nas colunas utilizadas.
    df = df.dropna(subset=used_cols).copy()
    df[TARGET] = df[TARGET].astype(int)
    dropped = rows_raw - len(df)

    meta = {
        "rows_raw": rows_raw,
        "rows_clean": len(df),
        "dropped": dropped,
        "legendaries": int(df[TARGET].sum()),
        "path": str(path),
    }
    return df.reset_index(drop=True), meta
