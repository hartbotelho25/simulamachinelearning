# Portal Interativo de Machine Learning — Pokémon Lendários

Aplicação **Streamlit** para analisar o dataset `base_pokemon.csv` e descobrir a forma mais assertiva de prever **quantos Pokémon lendários** existem na base (`is_legendary`).

O portal permite testar algoritmos, combinações flexíveis de atributos, a proporção treino/teste e o limiar de probabilidade da classificação.

## O que o portal faz

1. **Tratamento da base** — ao carregar o CSV, linhas com valores ausentes nas colunas utilizadas são eliminadas (`dropna`). Se `base_total` não existir, ele é calculado como `hp + attack + defense + sp_attack + sp_defense + speed`.
2. **Experimentos interativos** na barra lateral: split 50/50 até 90/10 (padrão 70/30, com `stratify=y`), seleção de algoritmos, presets de atributos e limiar de 0,10 a 0,90.
3. **Painel de resultados** com KPIs da amostra de teste, tabela comparativa (TP, FP, FN, precisão, recall, F1, ROC AUC e margem do real) e diagnóstico automático do sistema.

## Instalação

```bash
pip install streamlit pandas scikit-learn
```

Ou, a partir do repositório:

```bash
pip install -r requirements.txt
```

## Execução

Na raiz do projeto (onde estão `app.py` e `base_pokemon.csv`):

```bash
streamlit run app.py
```

A interface abre no navegador. Coloque o CSV na raiz se for substituir a base inclusa.

## Algoritmos

| Nome no portal        | Implementação scikit-learn        |
|-----------------------|-----------------------------------|
| Naive Bayes           | `GaussianNB`                      |
| Regressão Logística   | `LogisticRegression`              |
| Árvore de Decisão     | `DecisionTreeClassifier`          |
| Random Forest         | `RandomForestClassifier`          |
| Gradient Boosting     | `GradientBoostingClassifier`      |

Naive Bayes e Regressão Logística passam por `StandardScaler`. A semente é fixa (`random_state=42`) para reproduzir o mesmo split.

## Presets de atributos

- **Preset Aula:** `hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`
- **Preset Físicos:** `height_m`, `weight_kg`
- **Preset Agregado:** `base_total`
- **Preset Completo:** todos os atributos acima

## Estrutura

```
app.py                 # interface Streamlit
base_pokemon.csv       # dataset (Gens I–VII)
src/data.py            # carga, base_total e dropna
src/modeling.py        # split, treino, limiar e métricas
src/diagnostics.py     # parecer automático
src/ui.py              # layout e cartões
```

## Base

O arquivo `base_pokemon.csv` reúne 801 Pokémon (até a 7ª geração), com stats de combate, altura, peso, `base_total` e o alvo `is_legendary`. Cerca de 20 registros não têm altura/peso e são removidos na limpeza.
