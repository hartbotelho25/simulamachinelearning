"""Simulador de Machine Learning — Brasileirão (piloto após o portal Pokémon)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.br_data import (
    ALL_FEATURES,
    FEATURE_LABELS,
    GUIDE_PDF,
    LEAKAGE_NOTE,
    PRESET_CAPTIONS,
    PRESET_LABELS,
    PRESET_ORDER,
    TARGET_HELP,
    TARGET_LABELS,
    TARGETS,
    load_brasileirao,
    preset_features,
)
from src.br_diagnostics import diagnostic_sections, method_narrative, small_sample_note
from src.br_modeling import (
    MODEL_CATALOG,
    EvaluationResult,
    ablation_table,
    crossval_f1,
    run_experiment,
    variable_profile,
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


@st.cache_data(show_spinner=False)
def cached_profile(features, target_col):
    df, _ = get_data()
    return variable_profile(df, list(features), target_col)


@st.cache_data(show_spinner=False)
def cached_ablation(features, target_col, model, train_pct, threshold, k):
    df, _ = get_data()
    return ablation_table(df, list(features), target_col, model, train_pct, threshold, k)


def _init():
    if "br_preset" not in st.session_state:
        st.session_state.br_preset = "aula"
    if "br_target" not in st.session_state:
        st.session_state.br_target = "campeao"
    for feat in ALL_FEATURES:
        key = f"brfeat_{feat}"
        if key not in st.session_state:
            st.session_state[key] = feat in preset_features("aula", "campeao")


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
            options=list(TARGET_LABELS),
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
        use_cv = st.checkbox("Mostrar F1 de validação cruzada (até 5 folds)", value=True)

        st.divider()
        st.markdown("**3. Algoritmos**")
        st.caption("Mesmos métodos do piloto Pokémon, mais o KNN do guia.")
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
        if st.session_state.br_preset == "personalizado":
            for feat in ALL_FEATURES:
                st.checkbox(FEATURE_LABELS[feat], key=f"brfeat_{feat}")
            feats = [f for f in ALL_FEATURES if st.session_state.get(f"brfeat_{f}")]
        else:
            feats = preset_features(st.session_state.br_preset, st.session_state.br_target)
            if feats:
                render_chips("Colunas neste preset (já sem vazamento)", [FEATURE_LABELS[f] for f in feats])

        st.divider()
        st.markdown("**5. Limiar**")
        threshold = st.slider("Limiar de probabilidade", 0.10, 0.90, 0.50, 0.05)
        st.caption(f"SIM se P ≥ {threshold:.2f}. Baixar sobe recall; subir sobe precisão.")
        st.caption("Seed 42. NB, KNN e LR usam StandardScaler.")
    return train_pct, models, feats, threshold, k_nn, use_cv


def _pick(results, real, choice):
    if choice == "Mais próximo da contagem real":
        return min(results, key=lambda r: (abs(r.total_predito - real), -r.f1))
    return {r.modelo: r for r in results}.get(choice, results[0])


def _md(text: str) -> str:
    import re
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return html.replace("`", "")


def _study():
    with st.expander("Caderno do guia — 5 exercícios com gabarito", expanded=False):
        st.markdown(
            """
**1. Classifique VP / VN / FP / FN** (alvo: já foi rebaixado?)  
1) Vitória — previu CAIU · real caiu → **VP**  
2) Palmeiras — previu NÃO · real não caiu → **VN**  
3) São Paulo — previu CAIU · real não caiu → **FP**  
4) Coritiba — previu NÃO · real caiu → **FN**  
5) Sport — previu CAIU · real caiu → **VP**

**2. Calcule as métricas** (VP=7, FN=2, FP=3, VN=33)  
Acurácia 40/45 = **88,9%** · Precisão 7/10 = **70%** · Recall 7/9 = **77,8%** · F1 **73,7%**.  
A acurácia está inflada pelos 33 “não campeão”. O F1 é mais honesto.

**3. Qual modelo?**  
Explicar ao cliente → **Árvore**. Poucos dados → **Naive Bayes**. Estável → **Random Forest**.  
Direção do saldo de gols → **Regressão Logística**.

**4. Qual métrica no banco?**  
Jamais aprovar mau pagador → **precisão**. Alcance máximo → **recall**. Avaliação geral → **F1**.

**5. V ou F?**  
90% de acurácia sempre ótimo? **F**. Usar Rebaixamentos para prever rebaixado? **F** (leakage).  
100% no treino e 55% no teste? **V** (overfitting).
            """
        )


def main():
    _init()
    render_hero()
    try:
        df, meta = get_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    train_pct, models, features, threshold, k_nn, use_cv = render_sidebar(meta)
    tgt = st.session_state.br_target
    target_col = TARGETS[tgt]
    n_pos = int(df[target_col].sum())
    render_treatment(meta, TARGET_LABELS[tgt], n_pos)
    if GUIDE_PDF.exists():
        st.download_button(
            "Baixar o guia de estudo (PDF)",
            data=GUIDE_PDF.read_bytes(),
            file_name="Guia-ML-Brasileirao-Premium.pdf",
            mime="application/pdf",
        )
    _study()

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

    if use_cv:
        for r in results:
            r.cv_f1 = crossval_f1(df, features, target_col, r.modelo, k_nn)

    choice = st.selectbox(
        "Método em destaque",
        ["Mais próximo da contagem real", *[r.modelo for r in results]],
    )
    focus = _pick(results, real, choice)

    st.markdown("#### Indicadores da amostra de teste")
    render_kpis(n_test, real, focus.total_predito, focus.margem_pct)
    st.caption(
        f"Destaque **{focus.modelo}** · {TARGET_LABELS[tgt]} · limiar {threshold:.2f} · "
        f"{', '.join(FEATURE_LABELS[f] for f in features)}"
    )
    st.markdown(
        f'<div class="warn">{_md(small_sample_note(results, n_test, real, features, tgt))}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Comparativo por método")
    table = results_table(results)
    if use_cv and any(r.cv_f1 is not None for r in results):
        table["F1 CV"] = [f"{r.cv_f1*100:.1f}%" if r.cv_f1 is not None else "—" for r in results]
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Olhe o F1, não só a acurácia. Com 45 clubes o teste é pequeno — a coluna F1 CV estabiliza a leitura.")

    c1, c2 = st.columns((1.15, 1), gap="large")
    with c1:
        st.markdown("#### Contagem predita vs gabarito")
        chart = pd.DataFrame(
            {"Predito": [r.total_predito for r in results], "SIM reais": [real] * len(results)},
            index=[r.modelo for r in results],
        )
        st.bar_chart(chart, color=["#c084fc", "#E8B923"], height=260)
    with c2:
        st.markdown(f"#### Matriz · {focus.modelo}")
        render_confusion(focus)

    st.markdown("#### Impacto de cada variável")
    prof = cached_profile(tuple(features), target_col).copy()
    prof["Variável"] = prof["atributo"].map(FEATURE_LABELS)
    showp = prof.rename(columns={"media_sim": "Média no SIM", "media_nao": "Média no NÃO", "diferenca": "Diferença"})
    st.dataframe(showp[["Variável", "Média no SIM", "Média no NÃO", "Diferença"]].round(2), width="stretch", hide_index=True)
    st.bar_chart(showp.set_index("Variável")[["Diferença"]], color=["#E8B923"], height=200)

    if focus.importancias:
        st.caption(f"Peso em {focus.modelo}: {focus.origem_importancia}")
        imp = pd.DataFrame(
            [
                {"Variável": FEATURE_LABELS.get(f, f), "Importância": round(s, 4)}
                for f, s in sorted(focus.importancias.items(), key=lambda kv: kv[1], reverse=True)
            ]
        )
        st.dataframe(imp, width="stretch", hide_index=True)

    if len(features) >= 2:
        with st.spinner("Simulando a remoção de cada variável…"):
            ab = cached_ablation(tuple(features), target_col, focus.modelo, train_pct, float(threshold), int(k_nn))
        if not ab.empty:
            ab2 = ab.copy()
            ab2["Variável"] = ab2["atributo"].map(FEATURE_LABELS)
            ab2["F1 sem ela"] = ab2["f1_sem"].round(1)
            ab2["Δ F1"] = ab2["delta_f1"].map(lambda v: f"{v:+.1f}")
            ab2["Predito sem ela"] = ab2["predito_sem"]
            st.dataframe(ab2[["Variável", "F1 sem ela", "Δ F1", "Predito sem ela"]], width="stretch", hide_index=True)
    else:
        ab = pd.DataFrame()

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
            st.markdown(f'<div class="diag">{_md(method_narrative(r, real, tgt))}</div>', unsafe_allow_html=True)

    st.markdown("#### Diagnóstico, dicas e ponte para a CAIXA")
    for title, body in diagnostic_sections(results, real, features, n_test, threshold, tgt):
        st.markdown(
            f'<div class="diag"><div class="diag-title">{title}</div>{_md(body)}</div>',
            unsafe_allow_html=True,
        )

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
            focus=focus,
            results=results,
            profile=prof,
            ablation=ab if len(features) >= 2 else None,
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
