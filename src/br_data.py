"""Carga da base do Brasileirão e regras anti-vazamento."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "brasileirao.csv"
GUIDE_PDF = Path(__file__).resolve().parent.parent / "data" / "guia-ml-brasileirao.pdf"

COL_CLUB = "Time"

TARGETS: dict[str, str] = {
    "campeao": "Ja foi Campeao",
    "rebaixado": "Ja foi Rebaixado",
}

TARGET_LABELS: dict[str, str] = {
    "campeao": "Já foi campeão?",
    "rebaixado": "Já foi rebaixado?",
}

TARGET_HELP: dict[str, str] = {
    "campeao": "9 de 45 clubes com pelo menos 1 título (2003–2025). No banco: cliente de alto valor.",
    "rebaixado": "40 de 45 clubes caíram ao menos uma vez. No banco: risco de inadimplência.",
}

# Atributos selecionáveis pedidos — Títulos e Rebaixamentos ficam de fora (vazamento).
FEATURE_LABELS: dict[str, str] = {
    "Jogos": "Jogos",
    "Vitorias": "Vitórias",
    "Empates": "Empates",
    "Derrotas": "Derrotas",
    "Aproveitamento (%)": "Aproveitamento (%)",
    "Gols Feitos": "Gols feitos",
    "Gols Sofridos": "Gols sofridos",
    "Saldo de Gols": "Saldo de gols",
    "Vezes com Artilheiro": "Vezes com artilheiro",
    "Cartoes Vermelhos (2014-25)": "Cartões vermelhos",
}

ALL_FEATURES: list[str] = list(FEATURE_LABELS.keys())

LEAKAGE_NOTE: dict[str, str] = {
    "campeao": "A coluna Títulos não entra no treino: ela define o alvo e o modelo colaria.",
    "rebaixado": "A coluna Rebaixamentos não entra no treino: ela define o alvo e o modelo colaria.",
}

PRESET_ORDER = ["aula", "ataque", "solidez", "completo", "personalizado"]

PRESET_LABELS = {
    "aula": "Aula",
    "ataque": "Ataque",
    "solidez": "Solidez",
    "completo": "Completo",
    "personalizado": "Personalizado",
}

PRESET_BASE: dict[str, list[str]] = {
    "aula": ["Jogos", "Vitorias", "Empates", "Derrotas", "Saldo de Gols"],
    "ataque": ["Gols Feitos", "Vezes com Artilheiro", "Saldo de Gols", "Aproveitamento (%)"],
    "solidez": ["Gols Sofridos", "Derrotas", "Cartoes Vermelhos (2014-25)", "Jogos"],
    "completo": ALL_FEATURES.copy(),
    "personalizado": ALL_FEATURES.copy(),
}

PRESET_CAPTIONS = {
    "aula": "Jogos, V/E/D e saldo — recorte didático",
    "ataque": "Gols, saldo, artilheiros e aproveitamento",
    "solidez": "Gols sofridos, derrotas, cartões e jogos",
    "completo": "Os 10 atributos selecionáveis",
    "personalizado": "Você marca cada coluna",
}

BANK_MAP = {
    "campeao": "Identificar cliente de alto valor (private)",
    "rebaixado": "Prever inadimplência",
}


def allowed_features(_target_key: str | None = None) -> list[str]:
    return ALL_FEATURES.copy()


def preset_features(preset: str, _target_key: str | None = None) -> list[str]:
    return [f for f in PRESET_BASE.get(preset, ALL_FEATURES) if f in FEATURE_LABELS]


def load_brasileirao(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"Base '{path}' não encontrada.")
    df = pd.read_csv(path)
    rows_raw = len(df)
    needed = [COL_CLUB, *ALL_FEATURES, *TARGETS.values()]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Colunas ausentes na base: " + ", ".join(missing))
    for col in ALL_FEATURES + list(TARGETS.values()):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed).copy()
    for col in TARGETS.values():
        df[col] = df[col].astype(int)
    meta = {
        "rows_raw": rows_raw,
        "rows_clean": len(df),
        "dropped": rows_raw - len(df),
        "n_campeoes": int(df[TARGETS["campeao"]].sum()),
        "n_rebaixados": int(df[TARGETS["rebaixado"]].sum()),
        "path": str(path),
    }
    return df.reset_index(drop=True), meta
