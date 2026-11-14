"""Pastas do dossiê, relativas à raiz do repositório."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ARTEFATOS = RAIZ / "data" / "credito"
GRAFICOS = ARTEFATOS / "graficos"
CSV_CLIENTES = ARTEFATOS / "clientes_credito_50k.csv"
