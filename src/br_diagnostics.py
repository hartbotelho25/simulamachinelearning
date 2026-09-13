"""Diagnóstico e dicas do simulador do Brasileirão."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.br_data import BANK_MAP, CHALLENGE_TARGETS, FEATURE_LABELS, TARGET_LABELS

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
    """Explica o tamanho do teste e, nos alvos fáceis, o 100% enganoso."""
    perfect = [r.modelo for r in results if r.acuracia >= 0.999 or r.f1 >= 0.999]
    extra = ""
    if perfect:
        extra = f" Neste recorte, {', '.join(perfect)} bateu **100%** no teste — isso é frágil."
    base = (
        f"A base tem **45 clubes**. No split 70/30 o teste fica com **{n_test}** times e "
        f"**{real}** positivos."
    )
    if target_key == "campeao":
        return (
            f"{base} Os 9 campeões têm **772–894 jogos**; o modelo lê ‘clube grande’.{extra} "
            f"Troque para um alvo **Desafio** (caiu 2+ vezes, artilheiro ou multicampeão) "
            f"para os métodos deixarem de empatar em 100%."
        )
    if target_key == "rebaixado":
        return (
            f"{base} Só **5 clubes nunca caíram**. Chutar SIM já acerta ~89% "
            f"(armadilha da acurácia).{extra} No **Desafio · caiu duas vezes** a classe "
            f"fica ~24/45 e a acurácia cai para a casa dos 70%."
        )
    if target_key == "recorrente":
        return (
            f"{base} Quase metade da base é SIM (24/45). Esperado: acurácia ~70–80% e "
            f"F1 diferentes entre Naive Bayes, KNN e florestas. Se aparecer 100%, mude o split."
        )
    if target_key == "artilheiro":
        return (
            f"{base} 15 clubes tiveram artilheiro. Sem a coluna que define o alvo, "
            f"os F1 no teste costumam ficar ~60–89% — dá para eleger um vencedor."
        )
    if target_key == "multicampeao":
        return (
            f"{base} Só **6** multicampeões. Classe rara: um modelo que chuta NÃO infla "
            f"a acurácia e zera o F1. Compare F1 e F1 CV, não o percentual de acerto."
        )
    return f"{base}{extra} Olhe F1 + validação cruzada."


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
    if target_key in CHALLENGE_TARGETS:
        titulo_amostra = "Por que este alvo é o desafio"
    else:
        titulo_amostra = "Por que vários modelos chegam a 100%"
    secs.append((titulo_amostra, small_sample_note(results, n_test, real, features, target_key)))
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
                "Acurácia alta, F1 baixo — a armadilha do guia",
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
            "Se o placar estiver 100% para todo mundo, mude para um alvo **Desafio**. "
            "Trave o alvo e varra o limiar (0,35 / 0,50 / 0,70). "
            "Compare Naive Bayes (poucos dados) com Regressão Logística (explicável) "
            "e Random Forest (estável). Não ligue a coluna proibida do alvo.",
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
