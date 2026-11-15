"""Carrega o que o pipeline gravou e devolve a chance de um cliente."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .caminhos import ARTEFATOS, CSV_CLIENTES
from .dataset import FEATURES, gerar_cliente, ler_parametros
from .politica import FRACAO_TESTE, SEMENTE


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


def assinatura_modelos() -> str:
    """Chave do cache: muda quando o joblib, o CSV ou o esquema de colunas muda."""
    partes = []
    for arquivo in (ARTEFATOS / "modelos.joblib", CSV_CLIENTES):
        if arquivo.exists():
            stat = arquivo.stat()
            partes.append(f"{arquivo.name}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            partes.append(f"{arquivo.name}:ausente")
    esquema = hashlib.sha256(repr(tuple(FEATURES)).encode()).hexdigest()[:12]
    partes.append(esquema)
    return "|".join(partes)


def colunas_do_modelo(modelo) -> set[str] | None:
    prep = modelo.named_steps.get("prep") if hasattr(modelo, "named_steps") else None
    nomes = getattr(prep, "feature_names_in_", None)
    if nomes is None:
        nomes = getattr(modelo, "feature_names_in_", None)
    if nomes is None:
        return None
    return {str(nome) for nome in nomes}


def compativel(modelos: dict) -> bool:
    if not isinstance(modelos, dict) or not modelos:
        return False
    esperadas = set(FEATURES)
    for modelo in modelos.values():
        vistas = colunas_do_modelo(modelo)
        if vistas is None or vistas != esperadas:
            return False
    return True


def teste_compativel(df: pd.DataFrame) -> bool:
    colunas = set(df.columns)
    return set(FEATURES) <= colunas and {"id_cliente", "inadimplente_90d"} <= colunas


def resolver_modelos(obter, limpar, recarregar) -> dict:
    """Se o objeto cacheado divergir de FEATURES, joga fora e lê o joblib de novo."""
    try:
        modelos = obter()
        if not compativel(modelos):
            raise ValueError("columns are missing")
        return modelos
    except Exception:
        limpar()
        return recarregar()


def resolver_teste(obter, limpar, recarregar) -> pd.DataFrame:
    try:
        df = obter()
        if not teste_compativel(df):
            raise ValueError("columns are missing")
        return df
    except Exception:
        limpar()
        return recarregar()


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


def amostra_teste() -> pd.DataFrame:
    df = pd.read_csv(CSV_CLIENTES)
    y = df["inadimplente_90d"].to_numpy(dtype=int)
    indice = np.arange(len(df))
    _, indice_teste = train_test_split(
        indice,
        test_size=FRACAO_TESTE,
        stratify=y,
        random_state=SEMENTE,
    )
    return df.iloc[indice_teste].reset_index(drop=True)
