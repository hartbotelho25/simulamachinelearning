"""Simulador de Machine Learning — Brasileirão (piloto após o portal Pokémon)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.br_data import (
    ALL_FEATURES,
    FEATURE_LABELS,
    LEAKAGE_NOTE,
    PRESET_CAPTIONS,
    PRESET_LABELS,
    PRESET_ORDER,
    TARGET_HELP,
    TARGET_LABELS,
    TARGET_ORDER,
    TARGETS,
    allowed_features,
    load_brasileirao,
    preset_features,
)
from src.br_diagnostics import method_narrative
from src.br_modeling import (
    MODEL_CATALOG,
    impact_rank,
    run_experiment,
)
from src.br_report import build_pdf_report
from src.br_ui import (
    inject_css,
    render_chips,
    render_confusion,
    render_hero,
    render_kpis,
    render_treatment,
    results_table,
)

st.set_page_config(
    page_title="ML Brasileirão · Hart Botelho",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_data(show_spinner=False)
def get_data():
    return load_brasileirao()


@st.cache_data(show_spinner=False)
def cached_experiment(features, models, target_col, train_pct, threshold, k):
    df, _ = get_data()
    return run_experiment(df, list(features), target_col, list(models), train_pct, threshold, k)


def _init():
    if "br_preset" not in st.session_state or st.session_state.br_preset not in PRESET_ORDER:
        st.session_state.br_preset = "completo"
    if "br_target" not in st.session_state or st.session_state.br_target not in TARGET_ORDER:
        st.session_state.br_target = "recorrente"
    for feat in ALL_FEATURES:
        key = f"brfeat_{feat}"
        if key not in st.session_state:
            st.session_state[key] = feat in preset_features("completo", st.session_state.br_target)


def _on_preset():
    tgt = st.session_state.br_target
    preset = st.session_state.br_preset
    if preset == "personalizado":
        return
    chosen = set(preset_features(preset, tgt))
    for feat in ALL_FEATURES:
        st.session_state[f"brfeat_{feat}"] = feat in chosen


def _on_target():
    _on_preset()


def render_sidebar(meta):
    with st.sidebar:
        st.markdown("**Desenvolvido por Hart Botelho**")
        st.caption(f"{meta['rows_clean']} clubes · Série A 2003–2025")
        st.divider()
        st.markdown("**1. Pergunta (alvo)**")
        st.radio(
            "Alvo",
            options=TARGET_ORDER,
            format_func=lambda k: TARGET_LABELS[k],
            key="br_target",
            on_change=_on_target,
            label_visibility="collapsed",
        )
        st.caption(TARGET_HELP[st.session_state.br_target])
        st.markdown(
            f'<div class="warn">{LEAKAGE_NOTE[st.session_state.br_target]}</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**2. Treino / teste**")
        train_pct = st.slider("Proporção treino", 50, 90, 70, 5)
        st.caption(f"Treino {train_pct}% · teste {100-train_pct}% · stratify se possível. Base pequena: o teste tem poucos clubes.")
        st.divider()
        st.markdown("**3. Algoritmos**")
        models = []
        for name in MODEL_CATALOG:
            if st.checkbox(name, value=True, key=f"brm_{name}"):
                models.append(name)
        k_nn = st.slider("K do KNN (vizinhos)", 3, 7, 5, 2)

        st.divider()
        st.markdown("**4. Atributos**")
        st.radio(
            "Preset",
            options=PRESET_ORDER,
            format_func=lambda k: PRESET_LABELS[k],
            captions=[PRESET_CAPTIONS[k] for k in PRESET_ORDER],
            key="br_preset",
            on_change=_on_preset,
            label_visibility="collapsed",
        )
        allowed = allowed_features(st.session_state.br_target)
        if st.session_state.br_preset == "personalizado":
            for feat in allowed:
                st.checkbox(FEATURE_LABELS[feat], key=f"brfeat_{feat}")
            feats = [f for f in allowed if st.session_state.get(f"brfeat_{f}")]
        else:
            feats = preset_features(st.session_state.br_preset, st.session_state.br_target)
            if feats:
                render_chips("Atributos no treino", [FEATURE_LABELS[f] for f in feats])

        st.divider()
        st.markdown("**5. Limiar**")
        threshold = st.slider("Limiar de probabilidade", 0.10, 0.90, 0.50, 0.05)
        st.caption(f"SIM se P ≥ {threshold:.2f}. Baixar sobe recall; subir sobe precisão.")
        st.caption("Seed 42. NB, KNN e LR usam StandardScaler.")
    return train_pct, models, feats, threshold, k_nn


def _md(text: str) -> str:
    import re
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return html.replace("`", "")


def main():
    _init()
    render_hero()
    try:
        df, meta = get_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    train_pct, models, features, threshold, k_nn = render_sidebar(meta)
    tgt = st.session_state.br_target
    target_col = TARGETS[tgt]
    n_pos = int(df[target_col].sum())
    render_treatment(meta, TARGET_LABELS[tgt], n_pos)

    if not features:
        st.warning("Selecione ao menos um atributo permitido.")
        st.stop()
    if not models:
        st.warning("Selecione ao menos um algoritmo.")
        st.stop()

    with st.spinner("Treinando na amostra de teste…"):
        results, n_test, real = cached_experiment(
            tuple(features), tuple(models), target_col, train_pct, float(threshold), int(k_nn)
        )

    st.markdown("#### Indicadores da amostra de teste")
    render_kpis(n_test, real)
    st.caption(f"{TARGET_LABELS[tgt]} · limiar {threshold:.2f}")

    st.markdown("#### Comparativo por método")
    st.caption("Ordenado por ROC AUC, depois F1.")
    st.dataframe(results_table(results), width="stretch", hide_index=True)

    st.markdown("#### Relatório de cada método")
    tabs = st.tabs([r.modelo for r in results])
    for tab, r in zip(tabs, results):
        with tab:
            a, b, c, d = st.columns(4)
            a.metric("Predito", r.total_predito, f"{r.margem_pct:+.1f}%")
            b.metric("F1", f"{r.f1*100:.1f}%")
            c.metric("Precisão", f"{r.precisao*100:.1f}%")
            d.metric("Recall", f"{r.captura_pct:.1f}%")
            render_confusion(r)
            ranked = impact_rank(r)
            if ranked:
                mais, menos = ranked[0], ranked[-1]
                i1, i2 = st.columns(2)
                i1.metric("Mais impacto", FEATURE_LABELS.get(mais[0], mais[0]), f"{mais[1]:.3f}")
                i2.metric("Menos impacto", FEATURE_LABELS.get(menos[0], menos[0]), f"{menos[1]:.3f}")
                if r.origem_importancia:
                    st.caption(f"Neste modelo: {r.origem_importancia}.")
                imp = pd.DataFrame(
                    [
                        {"Variável": FEATURE_LABELS.get(f, f), "Importância": round(s, 4)}
                        for f, s in ranked
                    ]
                )
                st.dataframe(imp, width="stretch", hide_index=True)
            else:
                st.caption("Este método não devolveu pesos interpretáveis neste recorte.")
            st.markdown(f'<div class="diag">{_md(method_narrative(r, real, tgt))}</div>', unsafe_allow_html=True)

    st.markdown("#### Relatório em PDF")
    try:
        pdf_bytes = build_pdf_report(
            meta=meta,
            target_key=tgt,
            train_pct=train_pct,
            threshold=threshold,
            features=features,
            preset_label=PRESET_LABELS[st.session_state.br_preset],
            n_test=n_test,
            real=real,
            results=results,
        )
        st.download_button(
            "Baixar relatório completo em PDF",
            data=pdf_bytes,
            file_name=f"relatorio_brasileirao_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
    except Exception as exc:
        st.error(f"Não foi possível montar o PDF: {exc}")


if __name__ == "__main__":
    main()
