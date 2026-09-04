"""Portal Interativo de Machine Learning — previsão de Pokémon lendários."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import ALL_FEATURES, FEATURE_LABELS, PRESETS, TARGET, load_pokemon_data
from src.diagnostics import recommend
from src.modeling import MODEL_CATALOG, EvaluationResult, run_experiment
from src.ui import inject_css, render_confusion, render_hero, render_kpis, results_table

st.set_page_config(
    page_title="Portal ML · Pokémon Lendários",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, dict]:
    return load_pokemon_data()


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
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = PRESETS["aula"].copy()


def _apply_preset(key: str) -> None:
    st.session_state.selected_features = PRESETS[key].copy()


def render_sidebar(meta: dict) -> tuple[int, list[str], list[str], float]:
    with st.sidebar:
        st.markdown("### Controles do experimento")
        st.caption(
            f"Base limpa: **{meta['rows_clean']}** Pokémon · "
            f"**{meta['legendaries']}** lendários · "
            f"{meta['dropped']} removidos por valores ausentes."
        )
        st.divider()

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
        st.markdown("**Algoritmos**")
        selected_models: list[str] = []
        for name in MODEL_CATALOG:
            if st.checkbox(name, value=True, key=f"model_{name}"):
                selected_models.append(name)

        st.divider()
        st.markdown("**Atributos (features)**")
        c1, c2 = st.columns(2)
        c1.button("Preset Aula", width="stretch", on_click=_apply_preset, args=("aula",))
        c2.button("Preset Físicos", width="stretch", on_click=_apply_preset, args=("fisicos",))
        c3, c4 = st.columns(2)
        c3.button("Preset Agregado", width="stretch", on_click=_apply_preset, args=("agregado",))
        c4.button("Preset Completo", width="stretch", on_click=_apply_preset, args=("completo",))

        selected_features = st.multiselect(
            "Colunas usadas no treino",
            options=ALL_FEATURES,
            format_func=lambda k: FEATURE_LABELS[k],
            key="selected_features",
            help="Apenas registros sem valores ausentes entram no treino.",
        )

        st.divider()
        threshold = st.slider(
            "Limiar de decisão (threshold)",
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
        df, meta = get_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    train_pct, selected_models, selected_features, threshold = render_sidebar(meta)

    with st.expander("Amostra da base após o tratamento", expanded=False):
        preview_cols = ["pokedex_number", "name", "type1", "generation", *ALL_FEATURES, TARGET]
        preview_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(df[preview_cols].head(12), width="stretch", hide_index=True)
        st.caption(
            f"Arquivo original com {meta['rows_raw']} linhas. "
            f"`dropna()` nas colunas utilizadas removeu {meta['dropped']} Pokémon com dados vazios. "
            "Se `base_total` não existir no CSV, ele é calculado como "
            "`hp + attack + defense + sp_attack + sp_defense + speed`."
        )

    if not selected_features:
        st.warning("Selecione ao menos um atributo na barra lateral para treinar os modelos.")
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
        "Modelo em destaque nos cartões e na matriz de confusão",
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

    st.markdown("#### Comparativo de desempenho")
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
        "TP inclui a taxa de captura (recall). FP inclui o erro de palpite "
        "(falsos positivos entre os classificados como lendários). "
        "A margem do real compara a contagem predita com o gabarito do teste."
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

    st.markdown("#### Diagnóstico automático e recomendação")
    parecer = recommend(results, real, selected_features)
    st.markdown(f'<div class="diag">{_md_to_html(parecer)}</div>', unsafe_allow_html=True)


def _md_to_html(text: str) -> str:
    """Converte negrito markdown simples em HTML para o card de diagnóstico."""
    import re

    html = text.replace("\n\n", "<br><br>")
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = html.replace("`base_total`", "<code>base_total</code>")
    html = html.replace("`height_m`", "<code>height_m</code>")
    html = html.replace("`weight_kg`", "<code>weight_kg</code>")
    return html


if __name__ == "__main__":
    main()
