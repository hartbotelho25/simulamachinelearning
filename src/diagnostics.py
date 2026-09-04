"""Parecer automático sobre o experimento de classificação."""

from __future__ import annotations

from src.modeling import EvaluationResult


def recommend(results: list[EvaluationResult], real_lendarios: int, features: list[str]) -> str:
    """Gera o diagnóstico textual dinâmico pedido no painel principal."""
    if not results:
        return (
            "Nenhum algoritmo foi selecionado. Marque ao menos um modelo na barra lateral "
            "para o sistema comparar as previsões."
        )

    closest = min(
        results,
        key=lambda r: (abs(r.total_predito - real_lendarios), -r.f1),
    )
    delta = closest.total_predito - real_lendarios
    pct = (delta / real_lendarios * 100.0) if real_lendarios else 0.0

    parts: list[str] = []
    parts.append(
        f"O modelo **{closest.modelo}** chegou mais perto da contagem real de lendários "
        f"no conjunto de teste: previu **{closest.total_predito}** frente a **{real_lendarios}** "
        f"no gabarito (desvio de {delta:+d}, **{pct:+.1f}%**)."
    )

    if closest.total_predito < real_lendarios:
        hint_total = ""
        if "base_total" not in features:
            hint_total = (
                " Inclua o atributo **base_total**: lendários concentram totais de combate "
                "mais altos e esse agregado costuma reduzir a subcontagem."
            )
        parts.append(
            f"O modelo está **conservador**: previu menos lendários do que o número real "
            f"e deixou **{closest.fn}** lendário(s) de fora (falsos negativos). "
            f"Sugestão: **reduza o limiar de decisão** para aceitar probabilidades menores.{hint_total}"
        )
    elif closest.total_predito > real_lendarios:
        noisy = [f for f in ("height_m", "weight_kg") if f in features]
        if noisy:
            extra = (
                " Há atributos físicos pouco relacionados ao status lendário "
                f"({', '.join(noisy)}). **Remova peso e altura** e, se ainda houver excesso, "
                "eleve o limiar."
            )
        else:
            extra = " Tente **elevar o limiar de probabilidade** para filtrar palpiteiros."
        parts.append(
            f"O modelo **superestimou** a contagem em {delta} Pokémon e gerou "
            f"**{closest.fp} falso(s) alarme(s)** (falsos positivos).{extra}"
        )
    else:
        parts.append(
            "A contagem predita **coincidiu exatamente** com o gabarito. Isso não garante "
            "que os Pokémon apontados sejam os lendários corretos — confira precisão, recall "
            "e a matriz de confusão do modelo em destaque."
        )

    if closest.total_predito > 0 and closest.erro_palpite_pct >= 25:
        noisy = [f for f in ("height_m", "weight_kg") if f in features]
        if noisy:
            ruido = (
                " Atributos como **peso e altura** costumam adicionar ruído; remova-os e "
                "prefira os stats de combate ou o `base_total`."
            )
        elif closest.total_predito < real_lendarios:
            ruido = (
                " Como a contagem ainda está abaixo do real, evite subir o limiar; "
                "troque a combinação de atributos (por exemplo incluindo `base_total`) "
                "para reduzir palpiteiros sem perder captura."
            )
        else:
            ruido = " Revise a combinação de atributos e considere um limiar mais alto."
        parts.append(
            f"Há muitos **falsos alarmes**: {closest.erro_palpite_pct:.1f}% dos Pokémon "
            f"classificados como lendários não são (FP = {closest.fp}).{ruido}"
        )

    best_f1 = max(results, key=lambda r: (r.f1, r.roc_auc if r.roc_auc == r.roc_auc else -1))
    if best_f1.modelo != closest.modelo:
        parts.append(
            f"Se o critério for qualidade da classificação (equilíbrio precisão/captura), "
            f"o melhor **F1-Score** ficou com **{best_f1.modelo}** "
            f"({best_f1.f1 * 100:.1f}%). Chegar perto da contagem real e acertar *quais* "
            "são os lendários nem sempre andam juntos."
        )

    return "\n\n".join(parts)
