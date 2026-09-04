"""Parecer automático, dicas por método e leitura das variáveis."""

from __future__ import annotations

from src.data import FEATURE_LABELS
from src.modeling import EvaluationResult

METHOD_TIPS: dict[str, str] = {
    "Naive Bayes": (
        "Assume que os atributos são independentes. Quando você junta stats "
        "individuais com o `base_total` (que já é a soma deles), a probabilidade "
        "pode inflar e gerar falsos alarmes. Funciona melhor com poucos atributos "
        "bem distintos, como o preset Agregado."
    ),
    "Regressão Logística": (
        "Cada variável ganha um peso: coeficiente positivo empurra para lendário, "
        "negativo empurra para comum. É o melhor método para *ler a direção* do "
        "efeito. Com limiar 0,50 tende a ser conservador nesta base desbalanceada — "
        "baixe o limiar se a captura estiver baixa."
    ),
    "Árvore de Decisão": (
        "Corta a base em regras simples (ex.: Total Base acima de um valor). "
        "Fácil de acertar a contagem, mas uma árvore só pode memorizar o treino. "
        "Compare sempre com Random Forest: se a árvore ganhar muito no teste e "
        "perder em ROC AUC, há overfitting."
    ),
    "Random Forest": (
        "Média de várias árvores: costuma ser o simulador mais estável para "
        "contar lendários. Olhe a importância das variáveis — se peso e altura "
        "aparecerem no topo, o modelo está usando ruído."
    ),
    "Gradient Boosting": (
        "Corrige os erros das árvores anteriores. Costuma ter ROC AUC alto, mas "
        "pode ficar conservador ou agressivo conforme o limiar. Use o slider para "
        "alinhar a *contagem* sem destruir a precisão."
    ),
}

FEATURE_TIPS: dict[str, str] = {
    "hp": "HP alto ajuda, mas há comuns tanky. Sozinho raramente separa lendário.",
    "attack": "Ataque elevado é comum em lendários ofensivos, porém também em megaevoluções e pseudo-lendários.",
    "defense": "Defesa isolada distingue pouco: muitos comuns de pedra/aço têm defesa alta.",
    "sp_attack": "Ataque especial costuma marcar lendários de ataque especial (Mewtwo, Lugia).",
    "sp_defense": "Defesa especial alta aparece em lendários, mas também em tanks não lendários.",
    "speed": "Velocidade ajuda a achar lendários rápidos, mas deixa de fora os lentos e fortes.",
    "height_m": "Altura não define o status. Há lendários pequenos e comuns enormes — costuma ser ruído.",
    "weight_kg": "Peso tem o mesmo problema da altura. Se o modelo o prioriza, prefira stats de combate.",
    "base_total": "Melhor resumo da força: lendários concentram totais altos. Excelente para acertar a *contagem*.",
}


def _label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


def _auc(r: EvaluationResult) -> float:
    return r.roc_auc if r.roc_auc == r.roc_auc else -1.0


def diagnostic_sections(
    results: list[EvaluationResult],
    real_lendarios: int,
    features: list[str],
    n_test: int,
    threshold: float,
) -> list[tuple[str, str]]:
    """Blocos de diagnóstico para o simulador (título, texto markdown)."""
    if not results:
        return [
            (
                "Nenhum modelo",
                "Marque ao menos um algoritmo na barra lateral para o sistema comparar as previsões.",
            )
        ]

    closest = min(results, key=lambda r: (abs(r.total_predito - real_lendarios), -r.f1))
    best_f1 = max(results, key=lambda r: (r.f1, _auc(r)))
    best_auc = max(results, key=lambda r: _auc(r))
    best_recall = max(results, key=lambda r: (r.recall, r.f1))
    safest = max(results, key=lambda r: (r.precisao, r.f1))

    delta = closest.total_predito - real_lendarios
    pct = (delta / real_lendarios * 100.0) if real_lendarios else 0.0
    sections: list[tuple[str, str]] = []

    sections.append(
        (
            "Quem chegou mais perto da contagem real",
            f"No teste com **{n_test}** Pokémon (gabarito: **{real_lendarios}** lendários, "
            f"limiar **{threshold:.2f}**), o método **{closest.modelo}** previu "
            f"**{closest.total_predito}** (desvio {delta:+d}, **{pct:+.1f}%**). "
            f"Isso mede *quantos* o simulador apontou, não necessariamente *quais*.",
        )
    )

    if closest.total_predito < real_lendarios:
        hint = (
            " Inclua **Total Base** no preset: lendários concentram totais altos e o agregado "
            "costuma reduzir a subcontagem."
            if "base_total" not in features
            else " Com o Total Base já no treino, o caminho mais direto é **baixar o limiar** "
            "(por exemplo 0,35–0,45) para capturar lendários com probabilidade mediana."
        )
        sections.append(
            (
                "O simulador está conservador",
                f"**{closest.modelo}** deixou **{closest.fn}** lendário(s) de fora (FN) e "
                f"capturou só **{closest.captura_pct:.1f}%** do gabarito.{hint} "
                f"Exemplos de lendários perdidos: {_fmt_names(closest.exemplos_fn)}.",
            )
        )
    elif closest.total_predito > real_lendarios:
        noisy = [f for f in ("height_m", "weight_kg") if f in features]
        extra = (
            f" Os atributos físicos ({', '.join(_label(f) for f in noisy)}) raramente definem "
            "lendário. Troque para o preset **Aula** ou **Agregado** e, se ainda houver excesso, "
            "**suba o limiar**."
            if noisy
            else " **Suba o limiar** para exigir mais certeza antes de classificar como lendário."
        )
        sections.append(
            (
                "O simulador está gerando falsos alarmes",
                f"**{closest.modelo}** apontou {delta} a mais e marcou **{closest.fp}** comum(ns) "
                f"como lendário (FP). {_fmt_names(closest.exemplos_fp)}.{extra}",
            )
        )
    else:
        sections.append(
            (
                "A contagem bateu — agora olhe *quem*",
                f"Previu **{real_lendarios}**, igual ao gabarito. Ainda assim pode ter "
                f"**{closest.fp}** falso(s) alarme(s) e **{closest.fn}** lendário(s) perdido(s) "
                f"que se compensam. Precisão **{closest.precisao * 100:.1f}%** · captura "
                f"**{closest.captura_pct:.1f}%**. Acertos: {_fmt_names(closest.exemplos_tp)}.",
            )
        )

    if closest.total_predito > 0 and closest.erro_palpite_pct >= 25:
        sections.append(
            (
                "Erro de palpite alto",
                f"**{closest.erro_palpite_pct:.1f}%** dos classificados como lendários não são "
                f"(FP = {closest.fp}). Dica: tire `height_m`/`weight_kg`, evite misturar os 6 stats "
                f"com `base_total` no Naive Bayes, ou suba o limiar se a contagem já estiver "
                f"acima do real.",
            )
        )

    if best_f1.modelo != closest.modelo or best_auc.modelo != closest.modelo:
        sections.append(
            (
                "Contar certo × classificar certo",
                f"Melhor **F1** (equilíbrio precisão/captura): **{best_f1.modelo}** "
                f"({best_f1.f1 * 100:.1f}%). Melhor **ROC AUC** (ranking das probabilidades): "
                f"**{best_auc.modelo}** ({_fmt_auc(best_auc)}). "
                f"Melhor captura: **{best_recall.modelo}** ({best_recall.captura_pct:.1f}%). "
                f"Mais cirúrgico (precisão): **{safest.modelo}** ({safest.precisao * 100:.1f}%). "
                f"Use o destaque da página para inspecionar o método que importa para o seu critério.",
            )
        )

    ranked = _top_features(closest)
    if ranked:
        top = ", ".join(f"**{_label(f)}**" for f, _ in ranked[:3])
        weak = [f for f in features if f in ("height_m", "weight_kg")]
        weak_txt = (
            f" {', '.join(_label(f) for f in weak)} está no treino e costuma ser ruído — "
            "teste o preset Aula ou Agregado para ver o impacto na tabela."
            if weak
            else ""
        )
        tip = FEATURE_TIPS.get(ranked[0][0], "")
        sections.append(
            (
                "O que o método em destaque está olhando",
                f"Em **{closest.modelo}**, as variáveis com mais peso foram {top}. {tip}{weak_txt}",
            )
        )

    sections.append(("Como este método pensa", METHOD_TIPS.get(closest.modelo, "")))

    next_step = _next_experiment(closest, real_lendarios, features, threshold)
    sections.append(("Próximo experimento sugerido", next_step))

    sections.append(
        (
            "Como ler as métricas deste simulador",
            "**Total predito** = quantos o método chamou de lendário no teste. "
            "**TP** = acertou o lendário (taxa de captura / recall). "
            "**FP** = falso alarme (% de erro de palpite). "
            "**FN** = lendário perdido. "
            "**Precisão** = dos que ele apontou, quantos eram lendários. "
            "**F1** = equilíbrio entre precisão e captura. "
            "**ROC AUC** = qualidade do ranking das *probabilidades*, independente do limiar. "
            "Mover o limiar muda TP/FP/FN e a contagem; o AUC quase não muda.",
        )
    )
    return [(t, b) for t, b in sections if b]


def method_narrative(result: EvaluationResult, real: int) -> str:
    """Relatório curto de um método específico."""
    gap = result.total_predito - real
    parts = [
        f"**{result.modelo}** previu **{result.total_predito}** lendários (gabarito **{real}**, "
        f"margem **{result.margem_pct:+.1f}%**).",
        f"Acurácia **{result.acuracia * 100:.1f}%** · precisão **{result.precisao * 100:.1f}%** · "
        f"captura **{result.captura_pct:.1f}%** · F1 **{result.f1 * 100:.1f}%** · "
        f"especificidade **{result.especificidade * 100:.1f}%** · ROC AUC **{_fmt_auc(result)}**.",
        f"Matriz: TP {result.tp} · FP {result.fp} · FN {result.fn} · TN {result.tn}.",
    ]
    if result.exemplos_tp:
        parts.append(f"Acertos reais: {_fmt_names(result.exemplos_tp)}.")
    if result.exemplos_fp:
        parts.append(f"Falsos alarmes: {_fmt_names(result.exemplos_fp)}.")
    if result.exemplos_fn:
        parts.append(f"Lendários perdidos: {_fmt_names(result.exemplos_fn)}.")
    if gap < 0:
        parts.append("Dica: baixe o limiar ou inclua Total Base para capturar mais.")
    elif gap > 0:
        parts.append("Dica: suba o limiar ou remova altura/peso para filtrar palpiteiros.")
    else:
        parts.append("A contagem fechou. Confira se TP está alto — coincidir o total não basta.")
    parts.append(METHOD_TIPS.get(result.modelo, ""))
    return " ".join(parts)


def recommend(results: list[EvaluationResult], real_lendarios: int, features: list[str]) -> str:
    """Texto único (compatível com usos antigos)."""
    sections = diagnostic_sections(results, real_lendarios, features, 0, 0.5)
    return "\n\n".join(f"**{title}.** {body}" for title, body in sections)


def _fmt_names(names: list[str]) -> str:
    if not names:
        return "nenhum neste recorte"
    shown = ", ".join(names[:5])
    extra = "" if len(names) <= 5 else "…"
    return shown + extra


def _fmt_auc(r: EvaluationResult) -> str:
    if r.roc_auc != r.roc_auc:
        return "—"
    return f"{r.roc_auc:.3f}"


def _top_features(result: EvaluationResult) -> list[tuple[str, float]]:
    if not result.importancias:
        return []
    return sorted(result.importancias.items(), key=lambda kv: kv[1], reverse=True)


def _next_experiment(
    closest: EvaluationResult,
    real: int,
    features: list[str],
    threshold: float,
) -> str:
    has_physical = any(f in features for f in ("height_m", "weight_kg"))
    has_total = "base_total" in features
    only_total = features == ["base_total"]
    combat = {"hp", "attack", "defense", "sp_attack", "sp_defense", "speed"}
    has_combat_and_total = has_total and any(f in combat for f in features)

    if closest.fn >= closest.fp and closest.total_predito < real:
        if threshold > 0.35:
            return (
                f"Baixe o limiar de {threshold:.2f} para cerca de 0,35 e veja se a captura sobe "
                "sem explodir os falsos alarmes."
            )
        if not has_total:
            return "Troque para o preset **Agregado** (só Total Base) e compare a margem do real."
        return "Mantenha o Total Base e ligue também Random Forest, que costuma equilibrar a contagem."
    if closest.fp > closest.fn:
        if has_physical:
            return "Vá para o preset **Aula** (sem altura/peso) e compare o erro de palpite."
        if has_combat_and_total and closest.modelo == "Naive Bayes":
            return (
                "No Naive Bayes, desligue o Completo: stats + Total Base são redundantes. "
                "Teste só Agregado ou só Aula."
            )
        if threshold < 0.65:
            return f"Suba o limiar de {threshold:.2f} para 0,65 e veja a precisão subir."
        return "Compare Random Forest e Gradient Boosting no mesmo preset: um deles costuma filtrar melhor os FP."
    if only_total:
        return (
            "O Agregado é forte para *contar*. Agora rode o preset **Aula** para ver se os 6 stats "
            "identificam *quais* lendários com F1 melhor."
        )
    return (
        "Trave o preset que você mais gostou e varra o limiar (0,30 / 0,50 / 0,70): "
        "é o jeito mais claro de ver o simulador de acerto reagir."
    )
