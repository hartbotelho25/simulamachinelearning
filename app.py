"""Portal Interativo de Machine Learning — previsão de Pokémon lendários."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.data import (
    ALL_FEATURES,
    FEATURE_LABELS,
    PRESET_CAPTIONS,
    PRESET_LABELS,
    PRESET_ORDER,
    PRESETS,
    load_pokemon_data,
)
from src.modeling import (
    MODEL_CATALOG,
    EvaluationResult,
    ablation_table,
    run_experiment,
    variable_profile,
)
from src.diagnostics import diagnostic_sections, method_narrative
from src.report_pdf import build_pdf_report
from src.ui import (
    inject_css,
    render_confusion,
    render_feature_chips,
    render_hero,
    render_kpis,
    render_treatment,
    results_table,
)

st.set_page_config(
    page_title="Portal ML · Hart Botelho",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, dict]:
    return load_pokemon_data()


@st.cache_data(show_spinner=False)
def cached_profile(features: tuple[str, ...]) -> pd.DataFrame:
    df, _ = get_data()
    return variable_profile(df, list(features))


@st.cache_data(show_spinner=False)
def cached_ablation(
    features: tuple[str, ...],
    model_name: str,
    train_pct: int,
    threshold: float,
) -> pd.DataFrame:
    df, _ = get_data()
    return ablation_table(df, list(features), model_name, train_pct, threshold)


@st.cache_data(show_spinner=False)
def cached_experiment(
    features: tuple[str, ...],
    models: tuple[str, ...],
    train_pct: int,
    threshold: float,
) -> tuple[list[EvaluationResult], int, int]:
    df, _ = get_data()
    return run_experiment(df, list(features), list(models), train_pct, threshold)


def _init_state() -> None:
    if "feature_preset" not in st.session_state:
        st.session_state.feature_preset = "aula"
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = PRESETS["aula"].copy()
    for feat in ALL_FEATURES:
        key = f"feat_{feat}"
        if key not in st.session_state:
            st.session_state[key] = feat in PRESETS["aula"]


def _sync_feature_checkboxes(features: list[str]) -> None:
    selected = set(features)
    for feat in ALL_FEATURES:
        st.session_state[f"feat_{feat}"] = feat in selected


def _on_preset_change() -> None:
    preset = st.session_state.feature_preset
    if preset == "personalizado":
        return
    chosen = PRESETS[preset].copy()
    st.session_state.selected_features = chosen
    _sync_feature_checkboxes(chosen)


def _collect_custom_features() -> list[str]:
    picked = [feat for feat in ALL_FEATURES if st.session_state.get(f"feat_{feat}")]
    st.session_state.selected_features = picked
    return picked


def render_sidebar(meta: dict) -> tuple[int, list[str], list[str], float]:
    with st.sidebar:
        st.markdown("**Desenvolvido por Hart Botelho**")
        st.markdown("### Experimento")
        st.caption(
            f"Base limpa: **{meta['rows_clean']}** Pokémon · "
            f"**{meta['legendaries']}** lendários · "
            f"{meta['dropped']} removidos por valores ausentes."
        )
        st.divider()

        st.markdown("**1. Treino / teste**")
        train_pct = st.slider(
            "Proporção treino / teste",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            help="A divisão usa stratify=y para preservar a proporção de lendários.",
        )
        st.caption(f"Treino **{train_pct}%** · Teste **{100 - train_pct}%** · `stratify=y` ativo.")

        st.divider()
        st.markdown("**2. Algoritmos**")
        selected_models: list[str] = []
        for name in MODEL_CATALOG:
            if st.checkbox(name, value=True, key=f"model_{name}"):
                selected_models.append(name)

        st.divider()
        st.markdown("**3. Atributos**")
        st.caption("Escolha um preset. As colunas usadas aparecem logo abaixo — sem um segundo menu.")

        st.radio(
            "Preset",
            options=PRESET_ORDER,
            format_func=lambda key: PRESET_LABELS[key],
            captions=[PRESET_CAPTIONS[key] for key in PRESET_ORDER],
            key="feature_preset",
            on_change=_on_preset_change,
            label_visibility="collapsed",
        )

        preset = st.session_state.feature_preset
        if preset == "personalizado":
            st.caption("Marque as colunas que entram no treino:")
            for feat in ALL_FEATURES:
                st.checkbox(FEATURE_LABELS[feat], key=f"feat_{feat}")
            selected_features = _collect_custom_features()
            if selected_features:
                render_feature_chips(
                    "Colunas no treino",
                    [FEATURE_LABELS[f] for f in selected_features],
                )
        else:
            selected_features = PRESETS[preset].copy()
            st.session_state.selected_features = selected_features
            render_feature_chips(
                f"Colunas do preset {PRESET_LABELS[preset]}",
                [FEATURE_LABELS[f] for f in selected_features],
            )

        st.divider()
        st.markdown("**4. Limiar de decisão**")
        threshold = st.slider(
            "Limiar de probabilidade",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Pokémon com probabilidade ≥ limiar são classificados como lendários.",
        )
        st.caption(f"Classifica como lendário se P(lendário) ≥ **{threshold:.2f}**.")

        st.divider()
        st.caption(
            "Alvo: `is_legendary`. Seed fixa (42) para experimentos reproduzíveis. "
            "Naive Bayes e Regressão Logística usam `StandardScaler`."
        )

    return train_pct, selected_models, selected_features, threshold


def _pick_highlight(results: list[EvaluationResult], real: int, choice: str) -> EvaluationResult:
    if choice == "Mais próximo da contagem real":
        return min(results, key=lambda r: (abs(r.total_predito - real), -r.f1))
    by_name = {r.modelo: r for r in results}
    return by_name.get(choice, results[0])


def main() -> None:
    _init_state()
    render_hero()

    try:
        _df, meta = get_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    train_pct, selected_models, selected_features, threshold = render_sidebar(meta)
    render_treatment(meta, train_pct)

    if not selected_features:
        st.warning("Marque ao menos uma coluna no preset Personalizado para treinar os modelos.")
        st.stop()
    if not selected_models:
        st.warning("Selecione ao menos um algoritmo na barra lateral.")
        st.stop()

    with st.spinner("Treinando modelos na amostra de teste…"):
        results, n_test, real = cached_experiment(
            tuple(selected_features),
            tuple(selected_models),
            train_pct,
            float(threshold),
        )

    highlight_options = ["Mais próximo da contagem real", *[r.modelo for r in results]]
    focus_choice = st.selectbox(
        "Método em destaque nos cartões, na matriz e no impacto das variáveis",
        options=highlight_options,
        index=0,
    )
    focus = _pick_highlight(results, real, focus_choice)

    st.markdown("#### Indicadores da amostra de teste")
    render_kpis(n_test, real, focus.total_predito, focus.margem_pct)
    st.caption(
        f"Destaque: **{focus.modelo}** · limiar {threshold:.2f} · "
        f"atributos: {', '.join(FEATURE_LABELS[f] for f in selected_features)}."
    )

    st.markdown("#### Comparativo de desempenho por método")
    table = results_table(results)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Precisão (%)": st.column_config.NumberColumn(format="%.1f"),
            "Recall / Taxa de Captura (%)": st.column_config.NumberColumn(format="%.1f"),
            "F1-Score (%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Cada linha é um método no mesmo split e no mesmo limiar. "
        "TP = acertos reais (+ % de captura). FP = falsos alarmes (+ % de erro de palpite). "
        "FN = lendários perdidos. A margem compara a contagem predita com o gabarito do teste."
    )

    left, right = st.columns((1.15, 1), gap="large")
    with left:
        st.markdown("#### Contagem predita versus gabarito")
        chart_df = pd.DataFrame(
            {
                "Predito": [r.total_predito for r in results],
                "Lendários reais": [real] * len(results),
            },
            index=[r.modelo for r in results],
        )
        st.bar_chart(chart_df, color=["#c084fc", "#E8B923"], height=280)
    with right:
        st.markdown(f"#### Matriz de confusão · {focus.modelo}")
        render_confusion(focus)
        st.caption("Valores no conjunto de teste, já com o limiar escolhido.")

    _render_variable_impact(focus, selected_features, train_pct, threshold, real)
    _render_method_reports(results, real)
    _render_diagnostics(results, real, selected_features, n_test, threshold)

    st.markdown("#### Relatório em PDF")
    st.caption("O arquivo reúne configuração, indicadores, comparativo, impacto das variáveis, cada método e o diagnóstico.")
    _render_pdf_download(
        meta,
        train_pct,
        threshold,
        selected_features,
        n_test,
        real,
        focus,
        results,
        key="pdf_main",
    )
    with st.sidebar:
        st.divider()
        st.markdown("**5. Exportar**")
        _render_pdf_download(
            meta,
            train_pct,
            threshold,
            selected_features,
            n_test,
            real,
            focus,
            results,
            key="pdf_sidebar",
        )


def _render_pdf_download(
    meta: dict,
    train_pct: int,
    threshold: float,
    features: list[str],
    n_test: int,
    real: int,
    focus: EvaluationResult,
    results: list[EvaluationResult],
    key: str,
) -> None:
    preset_key = st.session_state.get("feature_preset", "aula")
    preset_label = PRESET_LABELS.get(preset_key, str(preset_key))
    profile = cached_profile(tuple(features))
    ablation = None
    if len(features) >= 2:
        ablation = cached_ablation(tuple(features), focus.modelo, train_pct, float(threshold))
    try:
        pdf_bytes = build_pdf_report(
            meta=meta,
            train_pct=train_pct,
            threshold=threshold,
            features=features,
            preset_label=preset_label,
            n_test=n_test,
            real=real,
            focus=focus,
            results=results,
            profile=profile,
            ablation=ablation,
        )
    except Exception as exc:
        st.error(f"Não foi possível montar o PDF: {exc}")
        return

    st.download_button(
        label="Baixar relatório completo em PDF",
        data=pdf_bytes,
        file_name=f"relatorio_lendarios_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
        key=key,
        help="Gera um PDF com configuração, KPIs, comparativo, impacto das variáveis, relatório por método e diagnóstico.",
    )


def _render_variable_impact(
    focus: EvaluationResult,
    features: list[str],
    train_pct: int,
    threshold: float,
    real: int,
) -> None:
    st.markdown("#### Impacto de cada variável")
    st.caption(
        "Primeiro a diferença na base limpa (lendários vs comuns). "
        f"Depois o peso que **{focus.modelo}** atribuiu a cada coluna. "
        "Por fim, o que acontece com F1 e com a contagem se essa coluna for removida."
    )

    profile = cached_profile(tuple(features)).copy()
    profile["Variável"] = profile["atributo"].map(FEATURE_LABELS)
    profile["Média nos lendários"] = profile["media_lendarios"].round(2)
    profile["Média nos comuns"] = profile["media_comuns"].round(2)
    profile["Diferença (lendário − comum)"] = profile["diferenca"].round(2)
    st.dataframe(
        profile[
            [
                "Variável",
                "Média nos lendários",
                "Média nos comuns",
                "Diferença (lendário − comum)",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    diff_chart = profile.set_index("Variável")[["Diferença (lendário − comum)"]]
    st.bar_chart(diff_chart, color=["#E8B923"], height=220)
    st.caption(
        "Valores positivos: lendários têm média maior nessa variável. "
        "Altura e peso costumam ter diferença, mas isso não significa que sejam bons preditores."
    )

    if focus.importancias:
        st.markdown(f"**Peso no método {focus.modelo}**")
        st.caption(focus.origem_importancia or "Importância estimada no treino.")
        imp_rows = []
        for feat, score in sorted(
            focus.importancias.items(), key=lambda kv: kv[1], reverse=True
        ):
            row = {
                "Variável": FEATURE_LABELS.get(feat, feat),
                "Importância": round(score, 4),
            }
            if feat in focus.direcao:
                sinal = focus.direcao[feat]
                row["Direção"] = (
                    "↑ aumenta chance de lendário"
                    if sinal > 0
                    else "↓ diminui chance de lendário"
                    if sinal < 0
                    else "neutro"
                )
            imp_rows.append(row)
        imp_df = pd.DataFrame(imp_rows)
        left, right = st.columns((1.2, 1), gap="large")
        with left:
            st.dataframe(imp_df, width="stretch", hide_index=True)
        with right:
            chart = pd.DataFrame(
                {"Importância": [r["Importância"] for r in imp_rows]},
                index=[r["Variável"] for r in imp_rows],
            )
            st.bar_chart(chart, color=["#c084fc"], height=240)

    if len(features) >= 2:
        st.markdown(f"**Simulação: e se tirar a variável? ({focus.modelo})**")
        with st.spinner("Retreinando o método sem cada atributo…"):
            ab = cached_ablation(
                tuple(features), focus.modelo, train_pct, float(threshold)
            )
        if not ab.empty:
            show = ab.copy()
            show["Variável"] = show["atributo"].map(FEATURE_LABELS)
            show["F1 sem ela (%)"] = show["f1_sem"].round(1)
            show["Δ F1 (pp)"] = show["delta_f1"].map(lambda v: f"{v:+.1f}")
            show["Predito sem ela"] = show["predito_sem"].astype(int)
            show["Δ contagem"] = show["delta_predito"].map(lambda v: f"{v:+.0f}")
            show["Captura sem ela (%)"] = show["recall_sem"].round(1)
            st.dataframe(
                show[
                    [
                        "Variável",
                        "F1 sem ela (%)",
                        "Δ F1 (pp)",
                        "Predito sem ela",
                        "Δ contagem",
                        "Captura sem ela (%)",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Gabarito do teste: **{real}** lendários. "
                "Δ F1 negativo = a variável estava ajudando o método. "
                "Δ contagem mostra se removê-la deixa o simulador mais conservador ou mais agressivo."
            )
    else:
        st.caption("Com uma só variável não há simulação de remoção — experimente o preset Aula ou Completo.")


def _render_method_reports(results: list[EvaluationResult], real: int) -> None:
    st.markdown("#### Relatório de cada método")
    st.caption(
        "Abra a aba do algoritmo para ver as métricas daquele método, exemplos de acerto e erro, "
        "e uma dica de como ele toma a decisão."
    )
    tabs = st.tabs([r.modelo for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predito", result.total_predito, f"{result.margem_pct:+.1f}% vs real")
            k2.metric("F1-Score", f"{result.f1 * 100:.1f}%")
            k3.metric("Precisão", f"{result.precisao * 100:.1f}%")
            k4.metric("Captura", f"{result.captura_pct:.1f}%")
            m1, m2, m3 = st.columns(3)
            m1.metric("Acurácia", f"{result.acuracia * 100:.1f}%")
            m2.metric("Especificidade", f"{result.especificidade * 100:.1f}%")
            auc = "—" if result.roc_auc != result.roc_auc else f"{result.roc_auc:.3f}"
            m3.metric("ROC AUC", auc)
            render_confusion(result)
            st.markdown(
                f'<div class="diag">{_md_to_html(method_narrative(result, real))}</div>',
                unsafe_allow_html=True,
            )


def _render_diagnostics(
    results: list[EvaluationResult],
    real: int,
    features: list[str],
    n_test: int,
    threshold: float,
) -> None:
    st.markdown("#### Diagnóstico automático, dicas e próximo passo")
    for title, body in diagnostic_sections(results, real, features, n_test, threshold):
        st.markdown(
            f'<div class="diag"><div class="diag-title">{title}</div>{_md_to_html(body)}</div>',
            unsafe_allow_html=True,
        )
        st.write("")


def _md_to_html(text: str) -> str:
    """Converte negrito markdown simples em HTML para os cards de diagnóstico."""
    import re

    html = text.replace("\n\n", "<br><br>")
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = html.replace("`base_total`", "<code>base_total</code>")
    html = html.replace("`height_m`", "<code>height_m</code>")
    html = html.replace("`weight_kg`", "<code>weight_kg</code>")
    return html


if __name__ == "__main__":
    main()
