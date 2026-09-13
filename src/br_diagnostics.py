"""Diagnóstico e dicas do simulador do Brasileirão."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.br_data import BANK_MAP, FEATURE_LABELS, TARGET_LABELS

if TYPE_CHECKING:
    from src.br_modeling import EvaluationResult

METHOD_TIPS = {
    "Naive Bayes": (
        "O apostador de probabilidades. Rápido e bom com poucos clubes (45). "
        "Assume atributos independentes — evite juntar vitórias, empates e aproveitamento. "
        "No banco: ótimo para um primeiro teste de score."
    ),
    "KNN": (
        "Me diga com quem andas. Classifica o clube pelos K vizinhos mais parecidos. "
        "Os números passam por StandardScaler para 'Jogos' não dominar 'Artilheiro'. "
        "Intuitivo, sem treino pesado."
    ),
    "Regressão Logística": (
        "O juiz de pesos. Coeficiente positivo empurra para SIM, negativo para NÃO. "
        "É a rainha do crédito bancário: dá para explicar a direção de cada fator."
    ),
    "Árvore de Decisão": (
        "O jogo de perguntas. Regras do tipo 'saldo > 100?'. Fácil de explicar ao cliente "
        "e ao regulador. Sozinha pode decorar o treino — compare com Random Forest."
    ),
    "Random Forest": (
        "O conselho de árvores. Mais estável, mostra importância das variáveis, "
        "mas é menos transparente. No banco, acerta mais e explica menos."
    ),
    "Gradient Boosting": (
        "Corrige os erros das árvores anteriores. Costuma ter F1/AUC altos, "
        "mas o limiar muda bastante a contagem. Use o slider para o cabo de guerra precisão vs recall."
    ),
}


def _lab(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


def _auc(r) -> float:
    return r.roc_auc if r.roc_auc == r.roc_auc else -1.0


def _names(xs: list[str]) -> str:
    if not xs:
        return "nenhum neste recorte"
    return ", ".join(xs[:6]) + ("…" if len(xs) > 6 else "")


def small_sample_note(results, n_test: int, real: int, features: list[str], target_key: str) -> str:
    """Nota curta sobre o tamanho do teste — sem sermão de placar 100%."""
    return (
        f"Teste com **{n_test}** clubes e **{real}** positivos (de 45 na base). "
        f"Com amostra assim, uma linha muda bastante o F1. Use a coluna F1 CV para comparar métodos."
    )


def diagnostic_sections(results, real, features, n_test, threshold, target_key: str):
    if not results:
        return [("Nenhum modelo", "Marque ao menos um algoritmo.")]
    closest = min(results, key=lambda r: (abs(r.total_predito - real), -r.f1))
    best_f1 = max(results, key=lambda r: (r.f1, _auc(r)))
    best_prec = max(results, key=lambda r: (r.precisao, r.f1))
    best_rec = max(results, key=lambda r: (r.recall, r.f1))
    delta = closest.total_predito - real
    pct = (delta / real * 100.0) if real else 0.0
    alvo = TARGET_LABELS[target_key]
    banco = BANK_MAP[target_key]
    secs = []
    secs.append(("Amostra de teste", small_sample_note(results, n_test, real, features, target_key)))
    secs.append(
        (
            "Quem chegou mais perto da contagem real",
            f"Alvo **{alvo}** no teste ({n_test} clubes, gabarito **{real}** SIM, limiar {threshold:.2f}). "
            f"**{closest.modelo}** previu **{closest.total_predito}** (desvio {delta:+d}, {pct:+.1f}%). "
            f"No banco isso equivale a: {banco}.",
        )
    )
    if closest.total_predito < real:
        secs.append(
            (
                "O simulador está conservador",
                f"Deixou **{closest.fn}** caso(s) passar(em) (FN). "
                f"Exemplos: {_names(closest.exemplos_fn)}. "
                f"No crédito, FN = negar um bom cliente. **Baixe o limiar** para capturar mais.",
            )
        )
    elif closest.total_predito > real:
        secs.append(
            (
                "O simulador está gerando alarmes falsos",
                f"**{closest.fp}** FP. Exemplos: {_names(closest.exemplos_fp)}. "
                f"No crédito, FP = aprovar quem não paga (prejuízo). **Suba o limiar** ou tire features ruidosas.",
            )
        )
    else:
        secs.append(
            (
                "A contagem bateu — olhe *quem*",
                f"Predito = gabarito ({real}), mas pode haver FP e FN que se compensam. "
                f"Precisão {closest.precisao*100:.1f}% · captura {closest.captura_pct:.1f}%. "
                f"Acertos: {_names(closest.exemplos_tp)}.",
            )
        )
    if closest.acuracia >= 0.8 and closest.f1 < 0.55:
        secs.append(
            (
                "Acurácia alta, F1 baixo",
                f"Acurácia {closest.acuracia*100:.1f}% parece ótima, mas o F1 é {closest.f1*100:.1f}%. "
                f"Com poucos SIM, um modelo preguiçoso que chuta NÃO infla a acurácia. **Olhe o F1.**",
            )
        )
    secs.append(
        (
            "Contar certo × classificar certo",
            f"Melhor F1: **{best_f1.modelo}** ({best_f1.f1*100:.1f}%). "
            f"Mais preciso (alarme confiável): **{best_prec.modelo}** ({best_prec.precisao*100:.1f}%). "
            f"Maior captura: **{best_rec.modelo}** ({best_rec.captura_pct:.1f}%). "
            f"Se o FP for caro (crédito), priorize precisão. Se o FN for caro (fraude/venda perdida), priorize recall.",
        )
    )
    if closest.importancias:
        top = sorted(closest.importancias.items(), key=lambda kv: kv[1], reverse=True)[:3]
        secs.append(
            (
                "O que o método em destaque está olhando",
                "Maiores pesos: " + ", ".join(f"**{_lab(f)}**" for f, _ in top) + ".",
            )
        )
    secs.append(("Como este método pensa", METHOD_TIPS.get(closest.modelo, "")))
    secs.append(
        (
            "Próximo experimento",
            "Trave o alvo e varra o limiar (0,35 / 0,50 / 0,70). "
            "Compare Naive Bayes (poucos dados) com Regressão Logística (explicável) "
            "e Random Forest (estável).",
        )
    )
    secs.append(
        (
            "Como ler as métricas",
            "VP = acertou o SIM. VN = acertou o NÃO. FP = alarme falso. FN = deixou passar. "
            "Acurácia = (VP+VN)/total. Precisão = VP/(VP+FP). Recall = VP/(VP+FN). "
            "F1 = equilíbrio (métrica-estrela desta base desbalanceada). "
            "O limiar move precisão vs recall; o ROC AUC quase não muda.",
        )
    )
    return [(t, b) for t, b in secs if b]


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
        f"{METHOD_TIPS.get(result.modelo, '')} "
        f"Ponte CAIXA: {BANK_MAP[target_key]}."
    )
