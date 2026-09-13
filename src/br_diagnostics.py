"""Textos curtos por método e ranking de impacto."""

from __future__ import annotations

METHOD_TIPS = {
    "Naive Bayes": (
        "O apostador de probabilidades. Rápido e bom com poucos clubes (45). "
        "Assume atributos independentes — evite juntar vitórias, empates e aproveitamento."
    ),
    "KNN": (
        "Me diga com quem andas. Classifica o clube pelos K vizinhos mais parecidos. "
        "Os números passam por StandardScaler para 'Jogos' não dominar 'Artilheiro'. "
        "Intuitivo, sem treino pesado."
    ),
    "Regressão Logística": (
        "O juiz de pesos. Coeficiente positivo empurra para SIM, negativo para NÃO. "
        "Dá para explicar a direção de cada fator pelo sinal do coeficiente."
    ),
    "Árvore de Decisão": (
        "O jogo de perguntas. Regras do tipo 'saldo > 100?'. Fácil de explicar. "
        "Sozinha pode decorar o treino — compare com Random Forest."
    ),
    "Random Forest": (
        "O conselho de árvores. Mais estável, mostra importância das variáveis, "
        "mas é menos transparente: acerta mais e explica menos."
    ),
    "Gradient Boosting": (
        "Corrige os erros das árvores anteriores. Costuma ter F1/AUC altos, "
        "mas o limiar muda bastante a contagem. Use o slider para o cabo de guerra precisão vs recall."
    ),
}


def _names(xs: list[str]) -> str:
    if not xs:
        return "nenhum neste recorte"
    return ", ".join(xs[:6]) + ("…" if len(xs) > 6 else "")


def method_narrative(result, real: int, target_key: str) -> str:
    return (
        f"**{result.modelo}** previu **{result.total_predito}** SIM (gabarito **{real}**, "
        f"margem {result.margem_pct:+.1f}%). "
        f"Acurácia {result.acuracia*100:.1f}% · precisão {result.precisao*100:.1f}% · "
        f"captura {result.captura_pct:.1f}% · F1 {result.f1*100:.1f}%. "
        f"TP {result.tp} · FP {result.fp} · FN {result.fn} · TN {result.tn}. "
        f"Acertos: {_names(result.exemplos_tp)}. "
        f"Alarmes: {_names(result.exemplos_fp)}. "
        f"Perdidos: {_names(result.exemplos_fn)}. "
        f"{METHOD_TIPS.get(result.modelo, '')}"
    )


def impact_rank(result) -> list[tuple[str, float]]:
    return sorted(result.importancias.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
