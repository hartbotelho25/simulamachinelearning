# Simulador de Machine Learning — Brasileirão

Portal **Streamlit** com a base do Brasileirão (Série A 2003–2025, 45 clubes).

Desenvolvido por **Hart Botelho**.

## O que o portal faz

1. Escolhe o **alvo**: já foi rebaixado, caiu duas vezes ou mais, já teve artilheiro, ou foi multicampeão.
2. Escolhe os **atributos** (completo ou personalizado).
3. Testa algoritmos (Naive Bayes, KNN, Regressão Logística, Árvore, Random Forest, Gradient Boosting), o split e o **limiar**.
4. Compara ROC AUC, F1, acurácia, precisão, recall, FP e FN. Em cada método, vê a matriz e o impacto das variáveis.
5. Baixa o relatório do experimento em PDF.

## Rodar na sua máquina

```bash
pip install -r requirements.txt
streamlit run app.py
```

Arquivo principal: `app.py`.

## Publicar no Streamlit Community Cloud (GitHub)

1. **Crie o repositório no GitHub** (se ainda estiver só neste ambiente, use o botão **Create repo** no Cursor para gerar o repo e enviar o `main`).
2. Confirme que o GitHub tem na raiz: `app.py`, `requirements.txt`, `packages.txt`, pasta `data/` e pasta `src/`.
3. Abra [https://share.streamlit.io](https://share.streamlit.io) e entre com a **mesma conta GitHub**.
4. **New app** → escolha o repositório → branch `main` → Main file path: `app.py`.
5. **Deploy**. O primeiro build instala as dependências; em 1–3 minutos o app ganha um endereço do tipo `https://<nome>-<usuario>.streamlit.app`.
6. Se o build falhar, em **Manage app** veja o log. Quase sempre é `requirements.txt` ou o caminho do `app.py`.

Não há senha nem banco neste projeto. Não precisa preencher **Secrets**.

Para atualizar o ar: faça `git push` no `main`. O Streamlit Cloud reconstrói sozinho.

## Estrutura

```
app.py
requirements.txt
packages.txt
data/brasileirao.csv
fonts/                  # DejaVu para o PDF na nuvem
src/
```
