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
    "recorrente": "Rebaixado Recorrente",
    "artilheiro": "Teve Artilheiro",
    "multicampeao": "Multicampeao",
}

TARGET_ORDER: list[str] = [
    "campeao",
    "rebaixado",
    "recorrente",
    "artilheiro",
    "multicampeao",
]

GUIDE_TARGETS = frozenset({"campeao", "rebaixado"})
CHALLENGE_TARGETS = frozenset({"recorrente", "artilheiro", "multicampeao"})

TARGET_LABELS: dict[str, str] = {
    "campeao": "Já foi campeão?",
    "rebaixado": "Já foi rebaixado?",
    "recorrente": "Caiu duas vezes ou mais?",
    "artilheiro": "Já teve artilheiro da Série A?",
    "multicampeao": "Foi multicampeão? (2+ títulos)",
}

TARGET_HELP: dict[str, str] = {
    "campeao": "Guia · 9/45. Quase se separa por volume de jogos — vários modelos batem 100% no teste. Use para ver a armadilha.",
    "rebaixado": "Guia · 40/45. Chutar SIM já acerta ~89%. Compare acurácia com F1.",
    "recorrente": "Desafio · 24/45, quase equilibrado. No teste os F1 ficam ~67–77% e os métodos discordam.",
    "artilheiro": "Desafio · 15/45. A coluna ‘vezes com artilheiro’ sai do treino (ela define o alvo).",
    "multicampeao": "Desafio · 6/45, classe rara. Árvore e boosting erram; a logística às vezes acerta. Compare F1, não acurácia.",
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
    "recorrente": "Rebaixamentos (contagem) não entra no treino: o alvo é ‘caiu 2+ vezes’.",
    "artilheiro": "Vezes com artilheiro não entra no treino: ela é a definição do alvo.",
    "multicampeao": "Títulos não entra no treino: o alvo é ‘dois ou mais títulos’.",
}

# Colunas dos 10 atributos que colariam neste alvo.
BLOCKED_FEATURES: dict[str, list[str]] = {
    "campeao": [],
    "rebaixado": [],
    "recorrente": [],
    "artilheiro": ["Vezes com Artilheiro"],
    "multicampeao": [],
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
    "rebaixado": "Prever inadimplência (primeira ocorrência)",
    "recorrente": "Prever inadimplência reincidente",
    "artilheiro": "Cliente que já contratou um produto premium",
    "multicampeao": "Cliente private recorrente (mais de um ciclo de alto valor)",
}


def target_choice_label(key: str) -> str:
    prefix = "Guia · " if key in GUIDE_TARGETS else "Desafio · "
    return prefix + TARGET_LABELS[key]


def allowed_features(target_key: str | None = None) -> list[str]:
    blocked = set(BLOCKED_FEATURES.get(target_key or "", []))
    return [f for f in ALL_FEATURES if f not in blocked]


def preset_features(preset: str, target_key: str | None = None) -> list[str]:
    allowed = set(allowed_features(target_key))
    return [f for f in PRESET_BASE.get(preset, ALL_FEATURES) if f in allowed]


def load_brasileirao(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"Base '{path}' não encontrada.")
    df = pd.read_csv(path)
    rows_raw = len(df)
    needed = [COL_CLUB, *ALL_FEATURES, "Titulos", "Rebaixamentos"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Colunas ausentes na base: " + ", ".join(missing))
    for col in [*ALL_FEATURES, "Titulos", "Rebaixamentos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed).copy()
    df["Titulos"] = df["Titulos"].astype(int)
    df["Rebaixamentos"] = df["Rebaixamentos"].astype(int)
    df["Ja foi Campeao"] = (df["Titulos"] >= 1).astype(int)
    df["Ja foi Rebaixado"] = (df["Rebaixamentos"] >= 1).astype(int)
    df["Rebaixado Recorrente"] = (df["Rebaixamentos"] >= 2).astype(int)
    df["Teve Artilheiro"] = (df["Vezes com Artilheiro"] > 0).astype(int)
    df["Multicampeao"] = (df["Titulos"] >= 2).astype(int)
    meta = {
        "rows_raw": rows_raw,
        "rows_clean": len(df),
        "dropped": rows_raw - len(df),
        "n_campeoes": int(df["Ja foi Campeao"].sum()),
        "n_rebaixados": int(df["Ja foi Rebaixado"].sum()),
        "n_recorrentes": int(df["Rebaixado Recorrente"].sum()),
        "n_artilheiros": int(df["Teve Artilheiro"].sum()),
        "n_multicampeoes": int(df["Multicampeao"].sum()),
        "path": str(path),
    }
    return df.reset_index(drop=True), meta
