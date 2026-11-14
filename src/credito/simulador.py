"""Carrega o que o pipeline gravou e devolve a chance de um cliente."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .caminhos import ARTEFATOS
from .dataset import FEATURES, gerar_cliente, ler_parametros


class ModelosIncompativeis(RuntimeError):
    """O binário gravado não abriu nesta instalação."""


def pronto() -> bool:
    return (ARTEFATOS / "relatorio.json").exists() and (ARTEFATOS / "modelos.joblib").exists()


def ler_relatorio() -> dict:
    return json.loads((ARTEFATOS / "relatorio.json").read_text(encoding="utf-8"))


def ler_perfil() -> dict:
    return json.loads((ARTEFATOS / "perfil.json").read_text(encoding="utf-8"))


def _conferir(modelos: dict) -> None:
    perfil = ler_perfil()
    linha = {}
    for coluna in FEATURES:
        info = perfil[coluna]
        linha[coluna] = info["moda"] if info["tipo"] == "categoria" else info["mediana"]
    frame = linha_para_frame(linha)
    for modelo in modelos.values():
        saida = modelo.predict_proba(frame)
        if getattr(saida, "shape", None) != (1, 2):
            raise ValueError("um modelo gravado não devolveu chance para duas classes")


def carregar_modelos() -> dict:
    try:
        modelos = joblib.load(ARTEFATOS / "modelos.joblib")
        _conferir(modelos)
    except Exception as exc:
        versoes = {}
        caminho = ARTEFATOS / "versoes.json"
        if caminho.exists():
            versoes = json.loads(caminho.read_text(encoding="utf-8"))
        esperado = ", ".join(f"{nome} {versao}" for nome, versao in versoes.items())
        raise ModelosIncompativeis(
            "Não foi possível usar os modelos gravados"
            + (f" ({esperado})" if esperado else "")
            + f": {exc}"
        ) from exc
    return modelos


def parametros_gerador() -> dict:
    return ler_parametros(ARTEFATOS / "gerador.json")


def linha_para_frame(linha: dict) -> pd.DataFrame:
    return pd.DataFrame([{coluna: linha[coluna] for coluna in FEATURES}])


def chance(modelo, linha: dict) -> float:
    return float(modelo.predict_proba(linha_para_frame(linha))[0, 1])


def cliente_novo(seed: int) -> dict:
    serie = gerar_cliente(seed, parametros_gerador())
    dados = {coluna: serie[coluna] for coluna in FEATURES}
    dados["id_cliente"] = str(serie["id_cliente"])
    dados["inadimplente_90d"] = int(serie["inadimplente_90d"])
    return dados


def figura(nome: str) -> Path:
    return ARTEFATOS / "graficos" / nome
