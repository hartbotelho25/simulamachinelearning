"""Carrega o que o pipeline gravou e devolve a chance de um cliente."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .caminhos import ARTEFATOS, CSV_CLIENTES
from .dataset import ALVOS, FEATURES, ler_parametros
from .importancia import pesos_do_modelo, ranking_itens
from .treino import fatiar, treinar_todos


class ModelosIncompativeis(RuntimeError):
    """O binário gravado não abriu nesta instalação."""


def pronto() -> bool:
    return (ARTEFATOS / "relatorio.json").exists() and (ARTEFATOS / "modelos.joblib").exists()


def ler_relatorio() -> dict:
    return json.loads((ARTEFATOS / "relatorio.json").read_text(encoding="utf-8"))


def _pipelines(modelos: dict) -> list:
    if not isinstance(modelos, dict):
        return []
    if "inadimplente" in modelos and isinstance(modelos.get("inadimplente"), dict):
        saida = []
        for bloco in modelos.values():
            if isinstance(bloco, dict):
                saida.extend(bloco.values())
        return saida
    return list(modelos.values())


def _conferir(modelos: dict) -> None:
    if not compativel(modelos):
        raise ValueError("columns are missing")
    frame = linha_para_frame({coluna: 0 for coluna in FEATURES})
    for modelo in _pipelines(modelos):
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
    blocos = [modelos.get(alvo) for alvo in ALVOS]
    if any(not isinstance(bloco, dict) or not bloco for bloco in blocos):
        return False
    esperadas = set(FEATURES)
    vistos = 0
    for bloco in blocos:
        for modelo in bloco.values():
            vistas = colunas_do_modelo(modelo)
            if vistas is None or vistas != esperadas:
                return False
            vistos += 1
    return vistos >= 8


def teste_compativel(df: pd.DataFrame) -> bool:
    colunas = set(df.columns)
    return set(FEATURES) <= colunas and {"id_cliente", *ALVOS} <= colunas


def treinar_em_memoria() -> dict:
    """Treina os quatro modelos nos dois alvos a partir do CSV, sem abrir o joblib."""
    df = pd.read_csv(CSV_CLIENTES)
    if not teste_compativel(df):
        raise ModelosIncompativeis("A base gravada não tem as colunas do simulador.")
    X_tr, _X_te, i_tr, _i_te = fatiar(df)
    treino = df.iloc[i_tr]
    modelos = {
        alvo: treinar_todos(X_tr, treino[alvo].to_numpy(dtype=int))
        for alvo in ALVOS
    }
    _conferir(modelos)
    return modelos


def resolver_modelos(obter, limpar, recarregar) -> dict:
    """Se o cache ou o joblib falhar, treina de novo a partir do CSV."""
    try:
        modelos = obter()
        if not compativel(modelos):
            raise ValueError("columns are missing")
        return modelos
    except Exception:
        limpar()
        try:
            modelos = recarregar()
            if not compativel(modelos):
                raise ValueError("columns are missing")
            return modelos
        except Exception:
            return treinar_em_memoria()


def resolver_teste(obter, limpar, recarregar) -> pd.DataFrame:
    try:
        df = obter()
        if not teste_compativel(df):
            raise ValueError("columns are missing")
        return df
    except Exception:
        limpar()
        return recarregar()


def _ler_joblib() -> dict:
    try:
        modelos = joblib.load(ARTEFATOS / "modelos.joblib")
        _conferir(modelos)
        return modelos
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


def carregar_modelos() -> dict:
    """Abre o joblib. Se a versão do Python ou do sklearn não bater, treina do CSV."""
    try:
        return _ler_joblib()
    except ModelosIncompativeis:
        return treinar_em_memoria()


def parametros_gerador() -> dict:
    return ler_parametros(ARTEFATOS / "gerador.json")


def linha_para_frame(linha: dict) -> pd.DataFrame:
    return pd.DataFrame([{coluna: int(linha.get(coluna, 0)) for coluna in FEATURES}])


def chance(modelo, linha: dict) -> float:
    return float(modelo.predict_proba(linha_para_frame(linha))[0, 1])


def chances(modelos: dict, slug: str, linha: dict) -> tuple[float, float]:
    """As duas chances são o predict_proba de cada modelo, sem ajuste posterior."""
    p_inad = chance(modelos["inadimplente"][slug], linha)
    p_90 = chance(modelos["inadimplente_90d"][slug], linha)
    return p_inad, p_90


def ler_importancias() -> dict:
    caminho = ARTEFATOS / "importancias.json"
    if not caminho.exists():
        return {}
    return json.loads(caminho.read_text(encoding="utf-8"))


def pesos_do_slug(modelos: dict, slug: str) -> dict[str, float]:
    """Percentual aprendido pelo modelo carregado. A mesma conta vai para importancias.json."""
    tabela = ranking_itens(pesos_do_modelo(modelos["inadimplente"][slug]))
    return {item["variavel"]: float(item["pct"]) for item in tabela}


def figura(nome: str) -> Path:
    return ARTEFATOS / "graficos" / nome


def amostra_teste() -> pd.DataFrame:
    df = pd.read_csv(CSV_CLIENTES)
    _x_tr, _x_te, _i_tr, i_te = fatiar(df)
    return df.iloc[i_te].reset_index(drop=True)
