# Simulador de Machine Learning

Portal **Streamlit** com dois estudos, desenvolvido por **Hart Botelho**.

1. **Simula Brasileirão** — Série A 2003–2025, 45 clubes. Treino ao vivo e PDF.
2. **Comitê de Crédito** — base sintética corporativa de 50.000 clientes e 38 colunas, quatro modelos gravados e simulador ao lado. Não é carteira de banco.

## O que o Simula Brasileirão faz

1. Escolhe o **alvo**: já foi rebaixado, caiu duas vezes ou mais, já teve artilheiro, ou foi multicampeão.
2. Escolhe os **atributos** (completo ou personalizado).
3. Testa algoritmos (Naive Bayes, KNN, Regressão Logística, Árvore, Random Forest, Gradient Boosting), o split e o **limiar**.
4. Compara ROC AUC, F1, acurácia, precisão, recall, FP e FN. Em cada método, vê a matriz e o impacto das variáveis.
5. Baixa o relatório do experimento em PDF.

## O que o Comitê de Crédito faz

O menu abre um relato de cima a baixo: livro sintético, quatro modelos (regressão logística, Random Forest, XGBoost e rede neural), ROC, faixas de alerta, peso dos blocos e o scorecard. A barra ao lado calcula a chance e sorteia um cliente novo no mesmo processo da base.

A base completa não vai no Git. Para gerá-la de novo e retreinar, na raiz do repositório:

```bash
python -m src.credito.pipeline
```

Os modelos em `data/credito/modelos.joblib` foram salvos com as versões fixadas em `requirements.txt`. Se a nuvem trouxer outra versão de scikit-learn ou xgboost, o relatório continua e o simulador avisa em vez de derrubar a página.

## Rodar na sua máquina

```bash
pip install -r requirements.txt
streamlit run app.py
```

Arquivo principal: `app.py`.

## Publicar no Streamlit Community Cloud (GitHub)

1. O repositório já está no GitHub. O deploy usa a branch `main`.
2. Na raiz precisam estar: `app.py`, `requirements.txt`, `runtime.txt`, `packages.txt`, `data/`, `fonts/` e `src/`.
3. Abra [https://share.streamlit.io](https://share.streamlit.io) e entre com a **mesma conta GitHub**.
4. **New app** → escolha o repositório → branch `main` → Main file path: `app.py`.
5. **Deploy**. O primeiro build instala as dependências.
6. Se o build falhar, em **Manage app** veja o log. Quase sempre é `requirements.txt` ou o caminho do `app.py`.

Não há senha nem banco neste projeto. Não precisa preencher **Secrets**.

Para atualizar o ar: faça `git push` no `main`. O Streamlit Cloud reconstrói sozinho.

## Estrutura

```
app.py                      # menu e navegação
runtime.txt                 # Python 3.12 na nuvem
requirements.txt
packages.txt
data/brasileirao.csv
data/credito/               # relatório, figuras, amostra e modelos
fonts/                      # DejaVu para o PDF na nuvem
src/br_app.py               # Simula Brasileirão
src/credito/                # pipeline modular do Comitê de Crédito
```
