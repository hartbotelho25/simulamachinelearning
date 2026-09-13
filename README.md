# Simulador de Machine Learning — Brasileirão

Portal **Streamlit** no mesmo molde do piloto dos Pokémon lendários, agora com a base `brasileirao-simulador-DEFINITO` (Série A 2003–2025, 45 clubes).

Desenvolvido por **Hart Botelho**.

## O que o portal faz

1. Você escolhe o **alvo**:
   - **Guia (fáceis):** já foi campeão? / já foi rebaixado? — vários modelos batem 100% no teste.
   - **Desafio:** caiu duas vezes ou mais? (24/45) · já teve artilheiro? (15/45) · foi multicampeão? (6/45).
2. Escolhe os **atributos**: Jogos, Vitórias, Empates, Derrotas, Aproveitamento, Gols feitos, Gols sofridos, Saldo de gols, Vezes com artilheiro, Cartões vermelhos.
3. Testa algoritmos (Naive Bayes, KNN, Regressão Logística, Árvore, Random Forest, Gradient Boosting), o split treino/teste e o **limiar**.
4. Compara VP, VN, FP, FN, precisão, recall, F1 e o impacto de cada variável.
5. Lê o **guia de estudo** (PDF) e os exercícios com gabarito.

**Anti-leakage:** `Títulos` não entra para campeão/multicampeão; `Rebaixamentos` não entra para rebaixado/recorrente; `Vezes com artilheiro` não entra no alvo artilheiro.

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
data/guia-ml-brasileirao.pdf
src/br_data.py
src/br_modeling.py
src/br_diagnostics.py
src/br_ui.py
src/br_report.py
```
