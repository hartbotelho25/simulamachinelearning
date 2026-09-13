"""Visual do portal Brasileirão."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.br_modeling import EvaluationResult

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
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 0.4rem 0 1rem; }
    .kpi { background: #163024; border: 1px solid rgba(244,241,232,0.08); border-radius: 14px;
           padding: 0.85rem 1rem; border-top: 3px solid var(--accent, #E8B923); }
    .kpi .lbl { font-size: 0.76rem; color: #a8a296; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi .val { font-size: 1.65rem; font-weight: 700; color: #F4F1E8; margin-top: 0.15rem; }
    .kpi .sub { font-size: 0.8rem; color: #c9c3b4; }
    .cm-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; }
    .cm { border-radius: 12px; padding: 0.8rem 1rem; background: #163024; border: 1px solid rgba(244,241,232,0.08); }
    .cm .k { font-size: 0.72rem; color: #a8a296; text-transform: uppercase; }
    .cm .v { font-size: 1.4rem; font-weight: 700; }
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
    @media (max-width: 900px) { .kpi-grid, .treat-grid { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 560px) { .kpi-grid, .treat-grid { grid-template-columns: 1fr; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <p class="credit-top">Desenvolvido por Hart Botelho</p>
        <div class="hero">
            <div class="badge">Simulador de acerto · Brasileirão 2003–2025 · CAIXA</div>
            <h1>Machine Learning na prática</h1>
            <p>
                Os alvos do <strong>guia</strong> (campeão / rebaixado) separam demais — vários
                modelos batem 100%. Os alvos <strong>desafio</strong> (caiu 2+ vezes, artilheiro,
                multicampeão) fazem os métodos discordarem. Mesmos tijolos: VP, VN, FP, FN, limiar, F1.
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
                <div class="cell"><div class="k">Guia · desafio</div>
                <div class="n">{meta['n_campeoes']} · {meta.get('n_recorrentes', 24)}</div>
                <div class="s">Campeões (fácil) · caiu 2+ (desafio)</div></div>
                <div class="cell"><div class="k">Teste pequeno</div><div class="n">~14</div>
                <div class="s">Com 70/30 o teste tem poucas linhas — use a validação cruzada</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:+.1f}%"


def render_kpis(n_test, real, predito, margem):
    tone = "#3dd68c" if abs(margem) < 15 else "#E8B923" if abs(margem) < 40 else "#ff6b6b"
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi" style="--accent:#6ea8fe"><div class="lbl">Clubes no teste</div>
            <div class="val">{n_test}</div><div class="sub">Amostra avaliada</div></div>
            <div class="kpi" style="--accent:#E8B923"><div class="lbl">SIM reais no teste</div>
            <div class="val">{real}</div><div class="sub">Gabarito</div></div>
            <div class="kpi" style="--accent:#c084fc"><div class="lbl">SIM preditos</div>
            <div class="val">{predito}</div><div class="sub">No limiar atual</div></div>
            <div class="kpi" style="--accent:{tone}"><div class="lbl">Desvio do real</div>
            <div class="val">{_fmt(margem)}</div><div class="sub">Contagem vs gabarito</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def results_table(results: list[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        auc = "—" if math.isnan(r.roc_auc) else f"{r.roc_auc:.3f}"
        rows.append(
            {
                "Modelo": r.modelo,
                "Total Predito": r.total_predito,
                "VP (captura)": f"{r.tp}  ({r.captura_pct:.1f}%)",
                "FP (alarme)": f"{r.fp}  ({r.erro_palpite_pct:.1f}%)",
                "FN (perdidos)": r.fn,
                "Acurácia (%)": round(r.acuracia * 100, 1),
                "Precisão (%)": round(r.precisao * 100, 1),
                "Recall (%)": round(r.recall * 100, 1),
                "F1-Score (%)": round(r.f1 * 100, 1),
                "ROC AUC": auc,
                "Margem (%)": _fmt(r.margem_pct),
            }
        )
    return pd.DataFrame(rows)


def render_confusion(r: EvaluationResult) -> None:
    st.markdown(
        f"""
        <div class="cm-wrap">
            <div class="cm"><div class="k">Verdadeiros negativos</div><div class="v" style="color:#6ea8fe">{r.tn}</div></div>
            <div class="cm"><div class="k">Falsos positivos · alarme</div><div class="v" style="color:#ff6b6b">{r.fp}</div></div>
            <div class="cm"><div class="k">Falsos negativos · passou</div><div class="v" style="color:#E8B923">{r.fn}</div></div>
            <div class="cm"><div class="k">Verdadeiros positivos · acerto</div><div class="v" style="color:#3dd68c">{r.tp}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
