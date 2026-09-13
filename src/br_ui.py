"""Visual do portal Brasileirão."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.br_data import FEATURE_LABELS
from src.br_modeling import EvaluationResult, impact_rank

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
    header[data-testid="stHeader"] { background: rgba(11,28,20,0.85); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #12261c 0%, #0b1812 100%); }
    .credit-top { color: #E8B923; font-size: 1.05rem; font-weight: 700; margin: 0 0 0.75rem 0; }
    .hero {
        background: radial-gradient(1000px 260px at 8% -20%, rgba(232,185,35,0.16), transparent),
                    linear-gradient(135deg, #163024 0%, #0b1812 55%, #1a2010 100%);
        border: 1px solid rgba(232,185,35,0.22);
        border-radius: 18px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    }
    .hero h1 { font-size: 1.8rem; margin: 0 0 0.3rem 0; color: #F4F1E8; }
    .hero p { color: #c9c3b4; margin: 0; line-height: 1.45; }
    .hero .badge {
        display: inline-block; background: rgba(232,185,35,0.15); color: #E8B923;
        border: 1px solid rgba(232,185,35,0.35); border-radius: 999px;
        padding: 0.15rem 0.7rem; font-size: 0.75rem; font-weight: 600;
        text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin: 0.4rem 0 1rem; }
    .mkpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.5rem; margin: 0.35rem 0 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: none; }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; height: 0 !important; }
    .stTabs [data-baseweb="tab"] {
        background: #1c3a2c; border-radius: 10px !important;
        border: 1px solid rgba(232,185,35,0.22) !important;
        padding: 0.45rem 0.85rem !important; color: #c9c3b4 !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #E8B923 !important; color: #0b1812 !important;
        font-weight: 800 !important;
        box-shadow: 0 0 0 2px rgba(232,185,35,0.28);
    }
    .cm-box { margin: 0.4rem 0 1rem; }
    .cm-axes { display: grid; grid-template-columns: 88px 1fr; gap: 0.35rem; align-items: stretch; }
    .cm-ylab { writing-mode: vertical-rl; transform: rotate(180deg);
        color: #a8a296; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
        display: flex; align-items: center; justify-content: center; }
    .cm-xlab { text-align: center; color: #a8a296; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.35rem; }
    .cm-grid { display: grid; grid-template-columns: 72px 1fr 1fr; gap: 0.45rem; }
    .cm-h { text-align: center; color: #a8a296; font-size: 0.7rem; text-transform: uppercase; padding: 0.2rem; }
    .cm-rh { display: flex; align-items: center; color: #a8a296; font-size: 0.7rem; text-transform: uppercase; }
    .cm { border-radius: 12px; padding: 0.75rem 0.85rem; background: #163024; border: 1px solid rgba(244,241,232,0.08); }
    .cm.ok { background: rgba(61,214,140,0.12); border-color: rgba(61,214,140,0.35); }
    .cm.bad { background: rgba(255,107,107,0.12); border-color: rgba(255,107,107,0.35); }
    .cm .k { font-size: 0.7rem; color: #c9c3b4; text-transform: uppercase; }
    .cm .v { font-size: 1.45rem; font-weight: 700; }
    .cm.ok .v { color: #3dd68c; }
    .cm.bad .v { color: #ff8a65; }
    .imp-help { color: #a8a296; font-size: 0.82rem; margin: 0.15rem 0 0.55rem; }
    .imp-row { display: grid; grid-template-columns: minmax(110px, 26%) 1fr 92px; gap: 0.55rem;
        align-items: center; margin: 0.28rem 0; }
    .imp-name { color: #F4F1E8; font-size: 0.86rem; }
    .imp-track { position: relative; height: 18px; background: #0f1f18; border-radius: 999px; overflow: hidden; }
    .imp-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(244,241,232,0.22); }
    .imp-bar { position: absolute; top: 3px; bottom: 3px; border-radius: 999px; }
    .imp-bar.pos { background: #3dd68c; }
    .imp-bar.neg { background: #c45c5c; }
    .imp-val { font-size: 0.82rem; font-weight: 600; text-align: right; }
    .imp-val.pos { color: #3dd68c; }
    .imp-val.neg { color: #ff8a65; }
    .kpi { background: #163024; border: 1px solid rgba(244,241,232,0.08); border-radius: 14px;
           padding: 0.85rem 1rem; border-top: 3px solid var(--accent, #E8B923); }
    .kpi .lbl { font-size: 0.76rem; color: #a8a296; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi .val { font-size: 1.65rem; font-weight: 700; color: #F4F1E8; margin-top: 0.15rem; }
    .kpi .sub { font-size: 0.8rem; color: #c9c3b4; }
    .diag { background: #163024; border-left: 4px solid #E8B923; border-radius: 12px;
            padding: 0.95rem 1.1rem; color: #F4F1E8; line-height: 1.55; margin-bottom: 0.6rem; }
    .diag-title { color: #E8B923; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; margin-bottom: 0.3rem; }
    .treat { background: #163024; border-radius: 14px; padding: 1rem 1.15rem; margin-bottom: 1rem;
             border: 1px solid rgba(244,241,232,0.08); }
    .treat h3 { margin: 0 0 0.35rem; color: #F4F1E8; font-size: 1.05rem; }
    .treat p { margin: 0 0 0.7rem; color: #c9c3b4; font-size: 0.92rem; }
    .treat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.55rem; }
    .treat .cell { background: #0f1f18; border-radius: 10px; padding: 0.6rem 0.7rem; }
    .treat .k { font-size: 0.7rem; color: #a8a296; text-transform: uppercase; }
    .treat .n { font-size: 1.3rem; font-weight: 700; color: #F4F1E8; }
    .treat .s { font-size: 0.78rem; color: #c9c3b4; }
    .preset-box { background: rgba(232,185,35,0.06); border: 1px solid rgba(232,185,35,0.22);
                  border-radius: 12px; padding: 0.65rem 0.75rem; margin-top: 0.4rem; }
    .preset-box .ttl { font-size: 0.7rem; color: #E8B923; text-transform: uppercase; font-weight: 600; margin-bottom: 0.35rem; }
    .chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
    .chip { background: #1c3a2c; border-radius: 999px; padding: 0.1rem 0.5rem; font-size: 0.76rem; color: #F4F1E8; }
    .warn { background: rgba(255,107,107,0.12); border-left: 3px solid #ff6b6b; border-radius: 8px;
            padding: 0.55rem 0.7rem; color: #F4F1E8; font-size: 0.85rem; margin: 0.4rem 0; }
    @media (max-width: 1100px) { .mkpi-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 900px) { .kpi-grid, .treat-grid { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 560px) {
        .kpi-grid, .treat-grid, .mkpi-grid { grid-template-columns: 1fr; }
        .imp-row { grid-template-columns: 1fr 70px; }
        .imp-name { grid-column: 1 / -1; }
    }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <p class="credit-top">Desenvolvido por Hart Botelho</p>
        <div class="hero">
            <div class="badge">Simulador de acerto · Brasileirão 2003–2025</div>
            <h1>Machine Learning &amp; Análise Preditiva</h1>
            <p>
                Simule cenários, ajuste limiares e compare a performance de
                múltiplos algoritmos de classificação em tempo real.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chips(title: str, labels: list[str]) -> None:
    chips = "".join(f'<span class="chip">{x}</span>' for x in labels)
    st.markdown(
        f'<div class="preset-box"><div class="ttl">{title}</div><div class="chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_treatment(meta: dict, target_label: str, n_pos: int) -> None:
    st.markdown(
        f"""
        <div class="treat">
            <h3>Como a base foi preparada</h3>
            <p>
                Série A 2003–2025, 9.165 jogos, <strong>{meta['rows_clean']}</strong> clubes.
                Sem valores vazios. Alvo atual: <strong>{target_label}</strong>
                ({n_pos} positivos). Colunas que entregam a resposta ficam bloqueadas (anti-leakage).
            </p>
            <div class="treat-grid">
                <div class="cell"><div class="k">Clubes</div><div class="n">{meta['rows_clean']}</div>
                <div class="s">Era dos pontos corridos</div></div>
                <div class="cell"><div class="k">Positivos neste alvo</div>
                <div class="n">{n_pos}</div>
                <div class="s">{target_label}</div></div>
                <div class="cell"><div class="k">Teste (70/30)</div><div class="n">~14</div>
                <div class="s">Poucas linhas — compare ROC AUC e F1</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:+.1f}%"


def render_kpis(n_test, real, predito=None, margem=None):
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi" style="--accent:#6ea8fe"><div class="lbl">Clubes no teste</div>
            <div class="val">{n_test}</div><div class="sub">Amostra avaliada</div></div>
            <div class="kpi" style="--accent:#E8B923"><div class="lbl">SIM reais no teste</div>
            <div class="val">{real}</div><div class="sub">Gabarito</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def results_table(results: list[EvaluationResult]) -> pd.DataFrame:
    ranked = sorted(
        results,
        key=lambda r: (
            r.roc_auc if r.roc_auc == r.roc_auc else -1.0,
            r.f1,
            r.acuracia,
        ),
        reverse=True,
    )
    rows = []
    for r in ranked:
        auc = None if math.isnan(r.roc_auc) else round(r.roc_auc, 3)
        rows.append(
            {
                "Modelo": r.modelo,
                "ROC AUC": auc,
                "F1-Score (%)": round(r.f1 * 100, 1),
                "Acurácia (%)": round(r.acuracia * 100, 1),
                "Precisão (%)": round(r.precisao * 100, 1),
                "Recall (%)": round(r.recall * 100, 1),
                "FP (alarme)": r.fp,
                "FN (perdidos)": r.fn,
            }
        )
    return pd.DataFrame(rows)


def _auc_txt(r: EvaluationResult) -> str:
    if r.roc_auc != r.roc_auc:
        return "—"
    return f"{r.roc_auc:.3f}"


def render_method_kpis(r: EvaluationResult) -> None:
    items = [
        ("Predito", str(r.total_predito), "SIM neste limiar", "#c084fc"),
        ("ROC AUC", _auc_txt(r), "Ordenação do comparativo", "#6ea8fe"),
        ("F1", f"{r.f1 * 100:.1f}%", "Equilíbrio precisão/recall", "#E8B923"),
        ("Acurácia", f"{r.acuracia * 100:.1f}%", "Acertos no teste", "#3dd68c"),
        ("Precisão", f"{r.precisao * 100:.1f}%", "Alarmes que acertam", "#E8B923"),
        ("Recall", f"{r.captura_pct:.1f}%", "SIM capturados", "#6ea8fe"),
    ]
    cards = "".join(
        f'<div class="kpi" style="--accent:{accent}"><div class="lbl">{lbl}</div>'
        f'<div class="val">{val}</div><div class="sub">{sub}</div></div>'
        for lbl, val, sub, accent in items
    )
    st.markdown(f'<div class="mkpi-grid">{cards}</div>', unsafe_allow_html=True)


def render_confusion(r: EvaluationResult) -> None:
    st.markdown(
        f"""
        <div class="cm-box">
            <div class="cm-xlab">Previsão</div>
            <div class="cm-axes">
                <div class="cm-ylab">Realidade</div>
                <div class="cm-grid">
                    <div></div>
                    <div class="cm-h">Previu NÃO</div>
                    <div class="cm-h">Previu SIM</div>
                    <div class="cm-rh">Era NÃO</div>
                    <div class="cm ok"><div class="k">Verdadeiro negativo</div><div class="v">{r.tn}</div></div>
                    <div class="cm bad"><div class="k">Falso positivo · alarme</div><div class="v">{r.fp}</div></div>
                    <div class="cm-rh">Era SIM</div>
                    <div class="cm bad"><div class="k">Falso negativo · omissão</div><div class="v">{r.fn}</div></div>
                    <div class="cm ok"><div class="k">Verdadeiro positivo</div><div class="v">{r.tp}</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_impact(r: EvaluationResult) -> None:
    ranked = impact_rank(r)
    if not ranked:
        st.caption("Este método não devolveu pesos interpretáveis neste recorte.")
        return
    origem = r.origem_importancia or "importância da variável neste modelo"
    as_pct = "queda de F1" in origem or "impureza" in origem or "único atributo" in origem
    rows = []
    for feat, mag in ranked:
        signed = r.direcao[feat] if feat in r.direcao else mag
        display = signed * 100.0 if as_pct else signed
        if as_pct:
            label = f"{display:+.2f}%".replace(".", ",")
        else:
            label = f"{display:+.2f}".replace(".", ",")
        rows.append((FEATURE_LABELS.get(feat, feat), display, label))
    peak = max((abs(v) for _, v, _ in rows), default=1.0) or 1.0
    html = [f'<p class="imp-help">Neste modelo: {origem}.</p>']
    for name, display, label in rows:
        width = max(4.0, abs(display) / peak * 50.0)
        kind = "neg" if display < 0 else "pos"
        if display < 0:
            bar = f'<span class="imp-bar neg" style="right:50%;width:{width:.1f}%"></span>'
        else:
            bar = f'<span class="imp-bar pos" style="left:50%;width:{width:.1f}%"></span>'
        html.append(
            f'<div class="imp-row"><div class="imp-name">{name}</div>'
            f'<div class="imp-track"><span class="imp-mid"></span>{bar}</div>'
            f'<div class="imp-val {kind}">{label}</div></div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)
