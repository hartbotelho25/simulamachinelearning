"""Gera a base, treina os quatro modelos e grava o dossiê.

Na raiz do repositório:

    python -m src.credito.pipeline

O CSV completo fica em data/credito/dataset_sintetico.csv e não entra no Git.
O app publicado usa o relatório, as figuras, a amostra e os modelos já gravados.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .caminhos import ARTEFATOS, CSV_CLIENTES, GRAFICOS
from .combinacoes import medir_combinacoes, sinais_das_colunas
from .dataset import COLUNAS, conferir_esquema, gerar_base, perfil_controles, salvar_parametros
from .graficos import combinacoes_barras, gerar_todos
from .importancia import explicar_logistica, pacote_importancia, ranking_itens, spearman_blocos
from .metricas import avaliar, faixas_alerta, leituras, modelo_referencia
from .politica import AMARELO_MAX, CORTE_MATRIZ, N_CLIENTES, SEMENTE, VERDE_MAX
from .treino import CATALOGO, fatiar, treinar_todos


def _gravar_versoes(pasta: Path) -> None:
    import sklearn
    import xgboost

    pasta.joinpath("versoes.json").write_text(
        json.dumps(
            {
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _pct(valor: float) -> str:
    return f"{100 * valor:.1f}%".replace(".", ",")


def _num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _com_artigo(nome: str) -> str:
    chave = nome.lower()
    if chave.startswith("regress"):
        return "a regressão logística"
    if chave.startswith("rede"):
        return "a rede neural"
    if chave.startswith("random"):
        return "a Random Forest"
    if chave.startswith("gradient"):
        return "o Gradient Boosting"
    return nome


def _textos(modelos: list[dict], faixas: list[dict], blocos_ref: list[dict], plantado: list[dict], rho: float, nome_ref: str) -> dict:
    por_auc = max(modelos, key=lambda m: m["roc_auc"])
    por_recall = max(modelos, key=lambda m: m["matriz"]["recall"])
    por_precisao = max(modelos, key=lambda m: m["matriz"]["precisao"])
    taxas = [f["taxa_observada"] for f in faixas]
    if all(t is not None for t in taxas) and taxas[0] < taxas[1] < taxas[2]:
        frase_faixa = (
            f"No teste, a inadimplência realizada sobe com a faixa d{_com_artigo(nome_ref)}: "
            f"verde {_pct(taxas[0])}, amarelo {_pct(taxas[1])}, vermelho {_pct(taxas[2])}."
        )
    else:
        pedacos = ", ".join(
            f"{f['faixa']} {'sem cliente' if f['taxa_observada'] is None else _pct(f['taxa_observada'])}"
            for f in faixas
        )
        frase_faixa = f"No teste, as taxas realizadas por faixa do {nome_ref} foram: {pedacos}."
    return {
        "auc": (
            f"{por_auc['nome']} teve o maior ROC AUC no teste: {_num(por_auc['roc_auc'])}."
        ),
        "corte": (
            f"No corte 0,50, {por_recall['nome']} captura {_pct(por_recall['matriz']['recall'])} "
            f"dos inadimplentes do teste. {por_precisao['nome']} acerta "
            f"{_pct(por_precisao['matriz']['precisao'])} dos alarmes que dispara."
        ),
        "faixas": frase_faixa,
        "blocos": (
            f"N{_com_artigo(nome_ref)}, o bloco de maior peso medido foi {blocos_ref[0]['nome']} "
            f"({_pct(blocos_ref[0]['fatia'])} da importância positiva). "
            f"No processo gerador, a maior fatia de sinal plantado está em {plantado[0]['nome']} "
            f"({_pct(plantado[0]['fatia'])}). "
            f"Spearman entre as duas ordens de bloco, neste modelo: {_num(rho, 2)}."
        ),
    }


def main() -> None:
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    print("gerando base", flush=True)
    df, parametros = gerar_base(N_CLIENTES, SEMENTE)
    conferir_esquema(df)
    if len(df) != N_CLIENTES:
        raise SystemExit(f"a base saiu com {len(df)} linhas")
    livro = df.copy()
    for coluna in livro.columns:
        if coluna != "id_cliente" and pd.api.types.is_float_dtype(livro[coluna]):
            livro[coluna] = livro[coluna].round(4)
    livro.to_csv(CSV_CLIENTES, index=False)
    livro.head(200).to_csv(ARTEFATOS / "amostra_clientes.csv", index=False)
    salvar_parametros(ARTEFATOS / "gerador.json", parametros)
    (ARTEFATOS / "perfil.json").write_text(
        json.dumps(perfil_controles(df), ensure_ascii=False),
        encoding="utf-8",
    )

    print("treinando", flush=True)
    X_tr, X_te, y_tr, y_te = fatiar(df)
    modelos = treinar_todos(X_tr, y_tr)
    probas = {slug: pipe.predict_proba(X_te)[:, 1] for slug, pipe in modelos.items()}
    avaliados = [avaliar(slug, y_te, probas[slug]) for slug in (item["slug"] for item in CATALOGO)]
    referencia = modelo_referencia(avaliados)
    faixas = faixas_alerta(y_te, probas[referencia])

    print("importância", flush=True)
    importancia = pacote_importancia(modelos, X_te, y_te)
    importancia["ordem_modelos"] = [{"slug": item["slug"], "nome": item["nome"]} for item in CATALOGO]
    explicacao = explicar_logistica(modelos["logistica"])
    rho = spearman_blocos(importancia["plantado"], importancia["medido"][referencia]["blocos"])
    textos = _textos(
        avaliados,
        faixas,
        importancia["medido"][referencia]["blocos"],
        importancia["plantado"],
        rho,
        next(m["nome"] for m in avaliados if m["slug"] == referencia),
    )

    print("combinações", flush=True)
    combinacoes = medir_combinacoes(X_te, y_te, probas, referencia)
    for combo in combinacoes:
        print(
            combo["id"],
            combo["n"],
            f"obs {combo['taxa_observada']:.3f}",
            f"prev {combo['prevista_referencia']:.3f}",
            "critica" if combo["critica"] else "abaixo",
            flush=True,
        )
    sinais = sinais_das_colunas(X_te, y_te)

    print("gráficos", flush=True)
    figuras = gerar_todos(
        avaliados,
        probas,
        y_te,
        faixas,
        importancia,
        explicacao,
        referencia,
        GRAFICOS,
    )
    nome_ref = next(m["nome"] for m in avaliados if m["slug"] == referencia)
    figuras["combinacoes"] = Path(combinacoes_barras(combinacoes, GRAFICOS, nome_ref)).name
    for item in avaliados:
        item.pop("roc", None)

    mlp = modelos["mlp"].named_steps["clf"]
    epocas = int(np.max(mlp.n_iter_)) if getattr(mlp, "n_iter_", None) is not None else None

    relatorio = {
        "sintetico": True,
        "seed": SEMENTE,
        "n_colunas": len(COLUNAS),
        "corte_matriz": CORTE_MATRIZ,
        "verde_max": VERDE_MAX,
        "amarelo_max": AMARELO_MAX,
        "referencia": referencia,
        "mlp_epocas": epocas,
        "base": {
            "n": int(len(df)),
            "n_colunas": int(df.shape[1]),
            "taxa": float(df["inadimplente_90d"].mean()),
            "positivos": int(df["inadimplente_90d"].sum()),
            "idade_mediana": float(df["idade"].median()),
            "renda_mediana": float(df["renda_declarada"].median()),
            "fluxo_mediano": float(df["fluxo_medio_entradas_3m"].median()),
            "score_mediano": float(df["score_bureau_atual"].median()),
            "n_treino": int(len(y_tr)),
            "n_teste": int(len(y_te)),
            "taxa_teste": float(y_te.mean()),
            "positivos_teste": int(y_te.sum()),
        },
        "modelos": avaliados,
        "faixas": faixas,
        "importancia": {
            "amostra": importancia["amostra"],
            "repeticoes": importancia["repeticoes"],
            "por_modelo": {slug: pacote["blocos"] for slug, pacote in importancia["medido"].items()},
            "itens": {
                slug: ranking_itens(pacote["colunas"])
                for slug, pacote in importancia["medido"].items()
            },
            "sinais": sinais,
            "plantado": importancia["plantado"],
            "spearman_referencia": rho,
        },
        "combinacoes": combinacoes,
        "explicacao": explicacao,
        "textos": textos,
        "figuras": figuras,
        "leituras": leituras(avaliados, faixas, referencia),
    }
    (ARTEFATOS / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(modelos, ARTEFATOS / "modelos.joblib", compress=3)
    _gravar_versoes(ARTEFATOS)
    print("referência", referencia)
    for item in avaliados:
        m = item["matriz"]
        print(
            f"{item['slug']:10} auc {item['roc_auc']:.3f}  prec {m['precisao']:.3f}  "
            f"rec {m['recall']:.3f}  f1 {m['f1']:.3f}  acc {m['acuracia']:.3f}"
        )
    print("taxa", f"{relatorio['base']['taxa']:.3f}", "spearman", f"{rho:.2f}")
    print("gravado em", ARTEFATOS)


if __name__ == "__main__":
    main()
