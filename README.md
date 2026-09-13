# Simulador de Machine Learning — Brasileirão

Portal **Streamlit** com a base do Brasileirão (Série A 2003–2025, 45 clubes).

Desenvolvido por **Hart Botelho**.

## O que o portal faz

1. Escolhe o **alvo**: já foi rebaixado, caiu duas vezes ou mais, já teve artilheiro, ou foi multicampeão.
2. Escolhe os **atributos** (completo ou personalizado): Jogos, Vitórias, Empates, Derrotas, Aproveitamento, Gols feitos, Gols sofridos, Saldo de gols, Vezes com artilheiro, Cartões vermelhos.
3. Testa algoritmos (Naive Bayes, KNN, Regressão Logística, Árvore, Random Forest, Gradient Boosting), o split treino/teste e o **limiar**.
4. Compara VP, VN, FP, FN, precisão, recall, F1 e, em cada método, a variável de maior e menor impacto.

**Anti-leakage:** `Títulos` não entra para multicampeão; `Rebaixamentos` não entra para rebaixado/recorrente; `Vezes com artilheiro` não entra no alvo artilheiro.

## Instalação e execução

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dependências: `streamlit pandas scikit-learn numpy fpdf2`.

## Estrutura

```
app.py                      # interface
data/brasileirao.csv        # 45 clubes
data/brasileirao.xlsx       # planilha original
src/br_data.py
src/br_modeling.py
src/br_diagnostics.py
src/br_ui.py
src/br_report.py
```
