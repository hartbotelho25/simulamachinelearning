"""Gera a base, treina os quatro modelos nos dois alvos e grava o dossiê.

Na raiz do repositório:

    python -m src.credito.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .caminhos import ARTEFATOS, CSV_CLIENTES, GRAFICOS
from .chaves import montar_chaves
from .combinacoes import medir_combinacoes
from .dataset import COLUNAS, conferir_esquema, gerar_base, salvar_parametros
from .graficos import combinacoes_barras, matrizes
from .importancia import pesos_do_modelo, ranking_blocos, ranking_itens
from .metricas import avaliar, modelo_referencia
from .politica import AMARELO_MAX, CORTE_MATRIZ, N_CLIENTES, SEMENTE, VERDE_MAX
from .treino import CATALOGO, fatiar, treinar_todos

ALVOS = (
    ("inadimplente", "Inadimplência geral", "Vai ficar inadimplente?"),
    ("inadimplente_90d", "Acima de 90 dias", "Vai passar de 90 dias?"),
)


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


def _importancia_aprendida(modelos: dict) -> dict:
    itens = {}
    blocos = {}
    for slug, modelo in modelos.items():
        pesos = pesos_do_modelo(modelo)
        itens[slug] = ranking_itens(pesos)
        blocos[slug] = ranking_blocos(pesos)
    return {"itens": itens, "blocos": blocos}


def _pacote_alvo(slug_alvo: str, nome: str, pergunta: str, y_te, probas, modelos) -> dict:
    avaliados = [avaliar(slug, y_te, probas[slug]) for slug in (item["slug"] for item in CATALOGO)]
    for item in avaliados:
        item.pop("roc", None)
    importancia = _importancia_aprendida(modelos)
    return {
        "id": slug_alvo,
        "nome": nome,
        "pergunta": pergunta,
        "referencia": modelo_referencia(avaliados),
        "modelos": avaliados,
        "importancia": importancia,
    }


def main() -> None:
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    print("gerando base", flush=True)
    df, parametros = gerar_base(N_CLIENTES, SEMENTE)
    conferir_esquema(df, checar_taxas=True)
    df.to_csv(CSV_CLIENTES, index=False)
    df.head(200).to_csv(ARTEFATOS / "amostra_clientes.csv", index=False)
    salvar_parametros(ARTEFATOS / "gerador.json", parametros)

    print("treinando", flush=True)
    X_tr, X_te, i_tr, i_te = fatiar(df)
    treino = df.iloc[i_tr]
    teste = df.iloc[i_te]
    gravados = {}
    pacotes = {}
    probas = {}
    for slug_alvo, nome, pergunta in ALVOS:
        print(" ", nome, flush=True)
        y_tr = treino[slug_alvo].to_numpy(dtype=int)
        y_te = teste[slug_alvo].to_numpy(dtype=int)
        modelos = treinar_todos(X_tr, y_tr)
        gravados[slug_alvo] = modelos
        probas[slug_alvo] = {
            slug: modelo.predict_proba(X_te)[:, 1] for slug, modelo in modelos.items()
        }
        print("  importância", nome, flush=True)
        pacotes[slug_alvo] = _pacote_alvo(slug_alvo, nome, pergunta, y_te, probas[slug_alvo], modelos)

    referencia = pacotes["inadimplente"]["referencia"]
    nome_ref = next(item["nome"] for item in CATALOGO if item["slug"] == referencia)
    print("combinações", flush=True)
    combinacoes = medir_combinacoes(
        X_te,
        teste["inadimplente"].to_numpy(dtype=int),
        teste["inadimplente_90d"].to_numpy(dtype=int),
        probas["inadimplente"][referencia],
        np.minimum(probas["inadimplente_90d"][referencia], probas["inadimplente"][referencia]),
    )
    for combo in combinacoes:
        print(
            f"  {combo['id']} n={combo['n']} inad {combo['taxa_inad']:.3f} "
            f"90 {combo['taxa_90']:.3f} {'critica' if combo['critica'] else 'abaixo'}",
            flush=True,
        )

    print("gráficos", flush=True)
    figuras = {}
    for slug_alvo, _nome, _pergunta in ALVOS:
        arquivo = f"matrizes_{slug_alvo}.png"
        matrizes(pacotes[slug_alvo]["modelos"], GRAFICOS, arquivo)
        figuras[slug_alvo] = arquivo
    figuras["combinacoes"] = Path(combinacoes_barras(combinacoes, GRAFICOS, nome_ref)).name

    relatorio = {
        "sintetico": True,
        "seed": SEMENTE,
        "n_colunas": len(COLUNAS),
        "corte_matriz": CORTE_MATRIZ,
        "verde_max": VERDE_MAX,
        "amarelo_max": AMARELO_MAX,
        "referencia": referencia,
        "base": {
            "n": int(len(df)),
            "n_colunas": int(df.shape[1]),
            "n_gatilhos": int(len(COLUNAS) - 3),
            "taxa_inadimplente": float(df["inadimplente"].mean()),
            "taxa_90d": float(df["inadimplente_90d"].mean()),
            "n_treino": int(len(i_tr)),
            "n_teste": int(len(i_te)),
            "taxa_teste_inad": float(teste["inadimplente"].mean()),
            "taxa_teste_90": float(teste["inadimplente_90d"].mean()),
        },
        "alvos": {slug: pacotes[slug] for slug, _nome, _pergunta in ALVOS},
        "combinacoes": combinacoes,
        "chaves": montar_chaves(),
        "figuras": figuras,
    }
    (ARTEFATOS / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(gravados, ARTEFATOS / "modelos.joblib", compress=3)
    importancias = {
        slug: pacotes[slug]["importancia"]
        for slug, _nome, _pergunta in ALVOS
    }
    (ARTEFATOS / "importancias.json").write_text(
        json.dumps(importancias, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _gravar_versoes(ARTEFATOS)
    base = relatorio["base"]
    print(
        f"taxas inad {base['taxa_inadimplente']:.3%}  90d {base['taxa_90d']:.3%}",
        flush=True,
    )
    for slug_alvo, nome, _pergunta in ALVOS:
        print(nome, flush=True)
        for item in pacotes[slug_alvo]["modelos"]:
            corte = item["matriz"]
            print(
                f"  {item['slug']:10} auc {item['roc_auc']:.3f}  prec {corte['precisao']:.3f}  "
                f"rec {corte['recall']:.3f}  f1 {corte['f1']:.3f}  acc {corte['acuracia']:.3f}"
            )
    print("gravado em", ARTEFATOS)


if __name__ == "__main__":
    main()
