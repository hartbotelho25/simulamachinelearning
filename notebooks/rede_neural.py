"""Rede neural para inadimplência.

Script completo: abre no VS Code ou, como notebook, no Google Colab.
Duas camadas ReLU e uma saída sigmoide. A classe rara pesa mais no treino.
A parada antecipada devolve os coeficientes da melhor validação.
A chance final passa por regressão isotônica numa fatia que o ajuste não viu.
O laço de épocas imprime a perda e a acurácia a cada passo.
A base vem do GitHub. Não grava modelo no disco.

No Colab o TensorFlow já vem instalado. No VS Code: pip install tensorflow
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

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
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print("Ajuste:", X_fit.shape, "Calibração:", X_cal.shape, "Teste:", X_test.shape)

escala = StandardScaler()
X_fit_scaled = escala.fit_transform(X_fit)
X_cal_scaled = escala.transform(X_cal)
X_test_scaled = escala.transform(X_test)

modelo = Sequential(
    [
        Input(shape=(X_fit_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo.summary()

classes = np.array([0, 1])
bruto = compute_class_weight(class_weight="balanced", classes=classes, y=y_fit)
peso_classe = {int(classe): float(valor) for classe, valor in zip(classes, bruto)}
print("Peso de cada classe:", peso_classe)

parada = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
historico = modelo.fit(
    X_fit_scaled,
    y_fit,
    validation_split=0.2,
    epochs=40,
    batch_size=256,
    class_weight=peso_classe,
    callbacks=[parada],
    verbose=1,
)
melhor = int(np.argmin(historico.history["val_loss"])) + 1
print(f"Épocas rodadas: {len(historico.history['loss'])}. Melhor validação: época {melhor}.")

print("Perda e acurácia a cada época:")
perda = historico.history["loss"]
acerto = historico.history["accuracy"]
val_perda = historico.history["val_loss"]
val_acerto = historico.history["val_accuracy"]
for epoca, (loss, acc, vloss, vacc) in enumerate(
    zip(perda, acerto, val_perda, val_acerto), start=1
):
    print(
        f"época {epoca:02d}  loss {loss:.4f}  acurácia {acc:.4f}  "
        f"val_loss {vloss:.4f}  val_acurácia {vacc:.4f}"
    )

bruta = modelo.predict(X_test_scaled, verbose=0).ravel()
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(modelo.predict(X_cal_scaled, verbose=0).ravel(), y_cal)
proba = iso.predict(bruta)
print(
    "Brier antes/depois:",
    round(float(brier_score_loss(y_test, bruta)), 3),
    round(float(brier_score_loss(y_test, proba)), 3),
)
print(
    "Média prevista antes/depois:",
    round(float(bruta.mean()), 3),
    round(float(proba.mean()), 3),
    "observada",
    round(float(y_test.mean()), 3),
)
print(
    "Recall no corte 0,5 antes/depois:",
    round(float(recall_score(y_test, bruta >= 0.5, zero_division=0)), 3),
    round(float(recall_score(y_test, proba >= 0.5, zero_division=0)), 3),
)
previsto = (proba >= 0.5).astype(int)

print("AUC:", round(float(roc_auc_score(y_test, proba)), 3))
print("Acurácia:", round(float(accuracy_score(y_test, previsto)), 3))
print("Precisão:", round(float(precision_score(y_test, previsto, zero_division=0)), 3))
print("Matriz de confusão (linha = real, coluna = previsto):")
print(confusion_matrix(y_test, previsto))
print(classification_report(y_test, previsto, digits=3, zero_division=0))

entrada = next(
    camada.get_weights()[0]
    for camada in modelo.layers
    if camada.get_weights() and camada.get_weights()[0].shape[0] == X.shape[1]
)
primeira = np.abs(entrada).sum(axis=1)
total = float(primeira.sum())
ordem = np.argsort(primeira)[::-1]
print("Importância de cada gatilho (peso absoluto da primeira camada, soma 100%):")
for indice in ordem:
    fatia = 100.0 * float(primeira[indice]) / total
    print(f"  {X.columns[indice]}: {fatia:.1f}%")
