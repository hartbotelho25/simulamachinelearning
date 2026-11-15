"""Random Forest para inadimplência.

Script completo: abre no VS Code ou, como notebook, no Google Colab.
A base vem do GitHub. A floresta recebe o 0/1 de cada gatilho, sem escala.
Não grava modelo nenhum no disco.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/hartbotelho25/simulamachinelearning/main/data/credito/clientes_credito_50k.csv"
# Troque para "inadimplente_90d" se a pergunta for atraso acima de 90 dias.
ALVO = "inadimplente"

df = pd.read_csv(URL)
print("Shape dos dados:", df.shape)
print("Valores nulos:", int(df.isnull().sum().sum()))
print("Distribuição do alvo:")
print(df[ALVO].value_counts())
print(df.head())

X = df.drop(columns=["id_cliente", "inadimplente", "inadimplente_90d"])
y = df[ALVO]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Treino:", X_train.shape, "Teste:", X_test.shape)

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=16,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
)
modelo.fit(X_train, y_train)

proba = modelo.predict_proba(X_test)[:, 1]
previsto = (proba >= 0.5).astype(int)

print("AUC:", round(float(roc_auc_score(y_test, proba)), 3))
print("Acurácia:", round(float(accuracy_score(y_test, previsto)), 3))
print("Precisão:", round(float(precision_score(y_test, previsto, zero_division=0)), 3))
print("Matriz de confusão (linha = real, coluna = previsto):")
print(confusion_matrix(y_test, previsto))
print(classification_report(y_test, previsto, digits=3, zero_division=0))

peso = np.asarray(modelo.feature_importances_, dtype=float)
total = float(peso.sum())
ordem = np.argsort(peso)[::-1]
print("Importância de cada gatilho (feature_importances_, soma 100%):")
for indice in ordem:
    fatia = 100.0 * float(peso[indice]) / total
    print(f"  {X.columns[indice]}: {fatia:.1f}%")
