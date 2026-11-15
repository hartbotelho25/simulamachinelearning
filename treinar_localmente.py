"""Treina os quatro modelos na sua máquina e imprime as métricas.

Lê a base que já está no repositório. Não regrava data/credito/modelos.joblib,
que é o arquivo aberto pelo site.

Hiperparâmetros: edite a função construir() em src/credito/treino.py.
Cada ramo (logistica, floresta, boosting, mlp) é um modelo.
Para experimentar outro algoritmo, acrescente um item em CATALOGO e um ramo em construir().

Na raiz do repositório:

    pip install -r requirements.txt
    python treinar_localmente.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.credito.caminhos import CSV_CLIENTES
from src.credito.dataset import ALVOS
from src.credito.treino import CATALOGO, fatiar, treinar_todos

CORTE = 0.5
NOMES = {
    "inadimplente": "Inadimplência",
    "inadimplente_90d": "Acima de 90 dias",
}


def main() -> None:
    if not CSV_CLIENTES.exists():
        raise SystemExit(f"Não encontrei a base em {CSV_CLIENTES}")
    df = pd.read_csv(CSV_CLIENTES)
    treino_x, teste_x, i_tr, i_te = fatiar(df)
    treino = df.iloc[i_tr]
    teste = df.iloc[i_te]
    print(f"clientes {len(df)}  treino {len(i_tr)}  teste {len(i_te)}")
    print("modelos:", ", ".join(item["nome"] for item in CATALOGO))
    for alvo in ALVOS:
        print(NOMES.get(alvo, alvo))
        modelos = treinar_todos(treino_x, treino[alvo].to_numpy(dtype=int))
        y = teste[alvo].to_numpy(dtype=int)
        for item in CATALOGO:
            proba = modelos[item["slug"]].predict_proba(teste_x)[:, 1]
            auc = roc_auc_score(y, proba)
            acc = accuracy_score(y, (proba >= CORTE).astype(int))
            print(f"  {item['nome']:28} AUC {auc:.3f}  acurácia {acc:.3f}")


if __name__ == "__main__":
    main()
