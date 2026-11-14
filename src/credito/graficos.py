"""Figuras do dossiê, no verde e no dourado do portal."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .politica import CORTE_MATRIZ

BG = "#0B1020"
PAINEL = "#10261c"
CREAM = "#F4F1E8"
MUDO = "#c9c3b4"
OURO = "#E8B923"
VERDE = "#3dd68c"
CORAL = "#ff8a65"
AZUL = "#6ea8fe"
ROXO = "#c084fc"
GRADE = "#1c3a2c"

CORES = {
    "logistica": OURO,
    "floresta": VERDE,
    "boosting": AZUL,
    "mlp": ROXO,
}
FAIXA_COR = {"verde": VERDE, "amarelo": OURO, "vermelho": CORAL}


def _fig(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h), dpi=140)
    fig.patch.set_facecolor(BG)
    return fig, ax


def _eixo(ax, titulo: str | None = None) -> None:
    ax.set_facecolor(PAINEL)
    ax.tick_params(colors=MUDO, labelsize=8)
    if titulo:
        ax.set_title(titulo, color=CREAM, loc="left", fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_color("#2a4a38")
    ax.grid(color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


def _salvar(fig, pasta: Path, nome: str) -> str:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.tight_layout()
    fig.savefig(caminho, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(caminho)


def _virgula(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def metricas_barras(modelos: list[dict], pasta: Path) -> str:
    fig, ax = _fig(11.2, 5.2)
    _eixo(ax, "Índices no teste")
    chaves = [
        ("roc_auc", "ROC AUC", None),
        ("precisao", "Precisão", "matriz"),
        ("recall", "Recall", "matriz"),
        ("f1", "F1", "matriz"),
        ("acuracia", "Acurácia", "matriz"),
    ]
    x = np.arange(len(chaves))
    largura = 0.18
    for i, modelo in enumerate(modelos):
        valores = []
        for chave, _rotulo, origem in chaves:
            if origem:
                valores.append(modelo[origem][chave])
            else:
                valores.append(modelo[chave])
        desloca = (i - (len(modelos) - 1) / 2) * largura
        barras = ax.bar(
            x + desloca,
            valores,
            largura,
            color=CORES[modelo["slug"]],
            label=modelo["nome"],
        )
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.015,
                _virgula(valor, 2),
                ha="center",
                va="bottom",
                color=MUDO,
                fontsize=6.5,
            )
    ax.set_xticks(x, [rotulo for _c, rotulo, _o in chaves], color=CREAM)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor, de 0 a 1", color=CREAM)
    ax.legend(facecolor=PAINEL, edgecolor="#2a4a38", labelcolor=CREAM, fontsize=8)
    ax.axhline(CORTE_MATRIZ, color="#ffffff", lw=0.4, alpha=0.25)
    return _salvar(fig, pasta, "metricas_modelos.png")


def matrizes(modelos: list[dict], pasta: Path) -> str:
    fig, eixos = plt.subplots(1, 4, figsize=(12.4, 3.6), dpi=140)
    fig.patch.set_facecolor(BG)
    for ax, modelo in zip(eixos, modelos):
        m = modelo["matriz"]
        grade = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
        ax.imshow(grade, cmap="Greens", vmin=0, vmax=grade.max() or 1)
        ax.set_facecolor(PAINEL)
        ax.set_xticks([0, 1], ["Previu em dia", "Previu calote"], color=MUDO, fontsize=7)
        ax.set_yticks([0, 1], ["Estava em dia", "Estava em atraso"], color=MUDO, fontsize=7)
        ax.set_title(modelo["nome"], color=CREAM, fontsize=10, loc="left")
        for (i, j), valor in np.ndenumerate(grade):
            ax.text(j, i, f"{int(valor)}", ha="center", va="center", color=BG, fontsize=12, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color("#2a4a38")
        ax.tick_params(length=0)
        ax.grid(False)
    fig.suptitle("Matrizes no corte 0,50", color=CREAM, fontsize=12, x=0.02, ha="left")
    return _salvar(fig, pasta, "matrizes_confusao.png")


def roc(modelos: list[dict], pasta: Path) -> str:
    fig, ax = _fig(8.2, 5.6)
    _eixo(ax, "Curvas ROC no teste")
    ax.plot([0, 1], [0, 1], color="#5c6b62", lw=1, ls="--")
    for modelo in modelos:
        ax.plot(
            modelo["roc"]["fpr"],
            modelo["roc"]["tpr"],
            color=CORES[modelo["slug"]],
            lw=2.2,
            label=f"{modelo['nome']}  {_virgula(modelo['roc_auc'], 3)}",
        )
    ax.set_xlabel("Falso alarme", color=CREAM)
    ax.set_ylabel("Calotes capturados", color=CREAM)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(facecolor=PAINEL, edgecolor="#2a4a38", labelcolor=CREAM, fontsize=8, loc="lower right")
    return _salvar(fig, pasta, "roc_comparativo.png")


def faixas(proba: np.ndarray, faixas_info: list[dict], pasta: Path, nome_modelo: str) -> str:
    fig, eixos = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=140, gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor(BG)
    ax, ax2 = eixos
    _eixo(ax, f"Chance prevista · {nome_modelo}")
    ax.hist(proba, bins=40, color=AZUL, alpha=0.9)
    for info in faixas_info:
        ax.axvspan(info["de"], min(info["ate"], 1), color=FAIXA_COR[info["faixa"]], alpha=0.16)
    ax.axvline(0.08, color=VERDE, lw=1)
    ax.axvline(0.20, color=CORAL, lw=1)
    ax.set_xlabel("Probabilidade de inadimplência", color=CREAM)
    ax.set_ylabel("Clientes no teste", color=CREAM)
    _eixo(ax2, "Taxa observada em cada faixa")
    nomes = [info["faixa"] for info in faixas_info]
    taxas = [0 if info["taxa_observada"] is None else info["taxa_observada"] for info in faixas_info]
    ax2.bar(nomes, taxas, color=[FAIXA_COR[n] for n in nomes])
    for i, info in enumerate(faixas_info):
        texto = "—" if info["taxa_observada"] is None else _virgula(info["taxa_observada"] * 100, 1) + "%"
        ax2.text(i, taxas[i] + 0.01, texto, ha="center", color=CREAM, fontsize=8)
    ax2.set_ylim(0, max(taxas + [0.2]) * 1.25)
    ax2.set_ylabel("Inadimplência realizada", color=CREAM)
    return _salvar(fig, pasta, "faixas_alerta.png")


def blocos(importancia: dict, referencia: str, pasta: Path) -> str:
    fig, ax = _fig(10.4, 5.4)
    _eixo(ax, "Fatia da importância por bloco")
    slugs = [item["slug"] for item in importancia["ordem_modelos"]]
    nomes = {item["slug"]: item["nome"] for item in importancia["ordem_modelos"]}
    # Mesma ordem de blocos para os quatro: a do modelo de referência.
    ordem = [b["bloco"] for b in importancia["medido"][referencia]["blocos"]]
    rotulos = [b["nome"] for b in importancia["medido"][referencia]["blocos"]]
    y = np.arange(len(ordem))
    largura = 0.18
    for i, slug in enumerate(slugs):
        mapa = {b["bloco"]: b["fatia"] for b in importancia["medido"][slug]["blocos"]}
        vals = [mapa[b] for b in ordem]
        ax.barh(
            y + (i - (len(slugs) - 1) / 2) * largura,
            vals,
            largura,
            color=CORES[slug],
            label=nomes[slug],
        )
    ax.set_yticks(y, rotulos, color=CREAM, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fatia da queda de AUC ao embaralhar o bloco", color=CREAM)
    ax.legend(facecolor=PAINEL, edgecolor="#2a4a38", labelcolor=CREAM, fontsize=8)
    return _salvar(fig, pasta, "importancia_blocos.png")


def explicacao(itens: list[dict], pasta: Path) -> str:
    fig, ax = _fig(10.2, 5.8)
    _eixo(ax, "Logística: o que empurra o dossiê")
    mostra = list(reversed(itens[:12]))
    cores = [CORAL if item["efeito"] >= 0 else VERDE for item in mostra]
    ax.barh([item["rotulo"] for item in mostra], [item["efeito"] for item in mostra], color=cores)
    ax.axvline(0, color="#F4F1E8", lw=0.6)
    ax.set_xlabel("Peso. Coral aumenta a chance de calote; verde reduz.", color=CREAM)
    return _salvar(fig, pasta, "explicacao_variaveis.png")


def visao_modelo(modelo: dict, y: np.ndarray, proba: np.ndarray, pasta: Path) -> str:
    fig, eixos = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=140)
    fig.patch.set_facecolor(BG)
    ax, ax2 = eixos
    m = modelo["matriz"]
    grade = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    ax.imshow(grade, cmap="Greens", vmin=0, vmax=grade.max() or 1)
    ax.set_facecolor(PAINEL)
    ax.set_xticks([0, 1], ["Previu em dia", "Previu calote"], color=MUDO, fontsize=8)
    ax.set_yticks([0, 1], ["Estava em dia", "Estava em atraso"], color=MUDO, fontsize=8)
    ax.set_title("Matriz no corte 0,50", color=CREAM, loc="left", fontsize=11)
    for (i, j), valor in np.ndenumerate(grade):
        ax.text(j, i, f"{int(valor)}", ha="center", va="center", color=BG, fontsize=13, fontweight="bold")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#2a4a38")
    _eixo(ax2, "Chance prevista, por situação real")
    ax2.hist(proba[y == 0], bins=28, color=VERDE, alpha=0.75, label="Ficou em dia")
    ax2.hist(proba[y == 1], bins=28, color=CORAL, alpha=0.75, label="Ficou inadimplente")
    ax2.axvline(CORTE_MATRIZ, color=OURO, lw=1.2, label="Corte 0,50")
    ax2.set_xlabel("Probabilidade", color=CREAM)
    ax2.legend(facecolor=PAINEL, edgecolor="#2a4a38", labelcolor=CREAM, fontsize=8)
    fig.suptitle(
        f"{modelo['nome']}  ·  ROC AUC {_virgula(modelo['roc_auc'], 3)}",
        color=CREAM,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return _salvar(fig, pasta, f"visao_{modelo['slug']}.png")


def gerar_todos(
    modelos: list[dict],
    probas: dict[str, np.ndarray],
    y: np.ndarray,
    faixas_info: list[dict],
    importancia: dict,
    explicacao_itens: list[dict],
    referencia: str,
    pasta: Path,
) -> dict[str, str]:
    nome_ref = next(m["nome"] for m in modelos if m["slug"] == referencia)
    caminhos = {
        "metricas": metricas_barras(modelos, pasta),
        "matrizes": matrizes(modelos, pasta),
        "roc": roc(modelos, pasta),
        "faixas": faixas(probas[referencia], faixas_info, pasta, nome_ref),
        "blocos": blocos(importancia, referencia, pasta),
        "explicacao": explicacao(explicacao_itens, pasta),
    }
    for modelo in modelos:
        caminhos[f"visao_{modelo['slug']}"] = visao_modelo(modelo, y, probas[modelo["slug"]], pasta)
    return {chave: Path(valor).name for chave, valor in caminhos.items()}
