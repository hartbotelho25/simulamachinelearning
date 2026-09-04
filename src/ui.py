"""Componentes visuais do portal Streamlit."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.modeling import EvaluationResult

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
    header[data-testid="stHeader"] { background: rgba(11,16,32,0.8); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #12182c 0%, #0b1020 100%); }
    .hero {
        background: radial-gradient(1200px 280px at 10% -20%, rgba(232,185,35,0.18), transparent),
                    linear-gradient(135deg, #161d33 0%, #0b1020 55%, #1a1230 100%);
        border: 1px solid rgba(232,185,35,0.22);
        border-radius: 18px;
        padding: 1.35rem 1.6rem 1.2rem 1.6rem;
        margin-bottom: 1.1rem;
    }
    .hero h1 {
        font-size: 1.85rem;
        margin: 0 0 0.25rem 0;
        color: #F4F1E8;
        letter-spacing: -0.02em;
    }
    .hero p { color: #c9c3b4; margin: 0; font-size: 0.98rem; line-height: 1.45; }
    .hero .badge {
        display: inline-block;
        background: rgba(232,185,35,0.15);
        color: #E8B923;
        border: 1px solid rgba(232,185,35,0.35);
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 0.4rem 0 1rem 0; }
    .kpi {
        background: #161d33;
        border: 1px solid rgba(244,241,232,0.08);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.85rem 1rem;
        border-top: 3px solid var(--accent, #E8B923);
    }
    .kpi .lbl { font-size: 0.78rem; color: #a8a296; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi .val { font-size: 1.7rem; font-weight: 700; color: #F4F1E8; margin-top: 0.2rem; line-height: 1.15; }
    .kpi .sub { font-size: 0.82rem; color: #c9c3b4; margin-top: 0.2rem; }
    .cm-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
    .cm {
        border-radius: 12px; padding: 0.85rem 1rem;
        border: 1px solid rgba(244,241,232,0.08);
        background: #161d33;
    }
    .cm .k { font-size: 0.75rem; color: #a8a296; text-transform: uppercase; letter-spacing: 0.04em; }
    .cm .v { font-size: 1.45rem; font-weight: 700; margin-top: 0.15rem; }
    .diag {
        background: #161d33;
        border-left: 4px solid #E8B923;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        color: #F4F1E8;
        line-height: 1.55;
    }
    @media (max-width: 900px) {
        .kpi-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
        .kpi-grid { grid-template-columns: 1fr; }
    }
    [data-testid="stSidebarNav"] { display: none; }
    .preset-box {
        background: rgba(232,185,35,0.06);
        border: 1px solid rgba(232,185,35,0.22);
        border-radius: 12px;
        padding: 0.7rem 0.8rem 0.75rem 0.8rem;
        margin: 0.35rem 0 0.5rem 0;
    }
    .preset-box .ttl {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #E8B923;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
    .chip {
        background: #1c2442;
        border: 1px solid rgba(244,241,232,0.12);
        color: #F4F1E8;
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        font-size: 0.78rem;
    }
    .treat {
        background: #161d33;
        border: 1px solid rgba(244,241,232,0.08);
        border-radius: 14px;
        padding: 1rem 1.15rem 0.9rem 1.15rem;
        margin-bottom: 1rem;
    }
    .treat h3 { margin: 0 0 0.35rem 0; font-size: 1.05rem; color: #F4F1E8; }
    .treat p { margin: 0 0 0.75rem 0; color: #c9c3b4; font-size: 0.92rem; line-height: 1.45; }
    .treat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }
    .treat .cell {
        background: #12182c;
        border-radius: 10px;
        padding: 0.65rem 0.75rem;
    }
    .treat .cell .k { font-size: 0.72rem; color: #a8a296; text-transform: uppercase; letter-spacing: 0.04em; }
    .treat .cell .n { font-size: 1.35rem; font-weight: 700; color: #F4F1E8; margin-top: 0.1rem; }
    .treat .cell .s { font-size: 0.8rem; color: #c9c3b4; margin-top: 0.15rem; }
    .diag-title { color: #E8B923; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; margin-bottom: 0.35rem; }
    @media (max-width: 800px) {
        .treat-grid { grid-template-columns: 1fr; }
    }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_feature_chips(title: str, labels: list[str]) -> None:
    chips = "".join(f'<span class="chip">{label}</span>' for label in labels)
    st.markdown(
        f'<div class="preset-box"><div class="ttl">{title}</div>'
        f'<div class="chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Simulador de acerto · is_legendary</div>
            <h1>Portal Interativo de Machine Learning</h1>
            <p>
                Simule a previsão de <strong>quantos Pokémon lendários</strong> existem na amostra
                de teste. Troque método, variáveis e limiar para ver o impacto em cada métrica —
                contagem, captura, falsos alarmes e o peso de cada atributo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt_signed_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:+.1f}%"


def render_kpis(
    n_test: int,
    real: int,
    predito: int,
    margem_pct: float,
) -> None:
    desvio = _fmt_signed_pct(margem_pct)
    tone = "#3dd68c" if abs(margem_pct) < 10 else "#E8B923" if abs(margem_pct) < 25 else "#ff6b6b"
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi" style="--accent:#6ea8fe">
                <div class="lbl">Total de Pokémon no teste</div>
                <div class="val">{n_test}</div>
                <div class="sub">Registros da amostra avaliada</div>
            </div>
            <div class="kpi" style="--accent:#E8B923">
                <div class="lbl">Lendários reais no teste</div>
                <div class="val">{real}</div>
                <div class="sub">Gabarito da amostra</div>
            </div>
            <div class="kpi" style="--accent:#c084fc">
                <div class="lbl">Total predito pelo modelo</div>
                <div class="val">{predito}</div>
                <div class="sub">Contagem estimada no limiar</div>
            </div>
            <div class="kpi" style="--accent:{tone}">
                <div class="lbl">Desvio relativo do real</div>
                <div class="val">{desvio}</div>
                <div class="sub">Predito versus gabarito</div>
            </div>
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
                "Verdadeiros Positivos (TP)": f"{r.tp}  ({r.captura_pct:.1f}% captura)",
                "Falsos Positivos (FP)": f"{r.fp}  ({r.erro_palpite_pct:.1f}% erro de palpite)",
                "Falsos Negativos (FN)": f"{r.fn}  lendários perdidos",
                "Precisão (%)": round(r.precisao * 100, 1),
                "Recall / Taxa de Captura (%)": round(r.recall * 100, 1),
                "F1-Score (%)": round(r.f1 * 100, 1),
                "ROC AUC": auc,
                "Margem do Real (%)": _fmt_signed_pct(r.margem_pct),
            }
        )
    return pd.DataFrame(rows)


def render_treatment(meta: dict, train_pct: int) -> None:
    """Explica a limpeza da base — não é uma amostra aleatória de 12 linhas."""
    na = meta.get("na_detail") or {}
    if na:
        motivo = ", ".join(
            f"{k} ({v} vazios)" for k, v in na.items()
        )
        motivo_txt = f"Motivo: valores ausentes em {motivo}."
    else:
        motivo_txt = "Nenhuma linha tinha valor ausente nas colunas usadas."
    teste_pct = 100 - train_pct
    st.markdown(
        f"""
        <div class="treat">
            <h3>Como a base foi preparada</h3>
            <p>
                O CSV original tem <strong>{meta['rows_raw']}</strong> Pokémon.
                Antes de treinar, o portal remove qualquer registro com dado vazio
                nas colunas de modelagem (<code>dropna</code>). {motivo_txt}
                O que resta é a <strong>base limpa</strong> — não uma amostra de 12 nomes.
                O treino e o teste são sorteados a partir desses
                <strong>{meta['rows_clean']}</strong> registros ({train_pct}% / {teste_pct}%, com <code>stratify</code>).
            </p>
            <div class="treat-grid">
                <div class="cell">
                    <div class="k">CSV original</div>
                    <div class="n">{meta['rows_raw']}</div>
                    <div class="s">Pokémon no arquivo</div>
                </div>
                <div class="cell">
                    <div class="k">Removidos na limpeza</div>
                    <div class="n">{meta['dropped']}</div>
                    <div class="s">Linhas com NaN</div>
                </div>
                <div class="cell">
                    <div class="k">Base limpa · lendários</div>
                    <div class="n">{meta['rows_clean']} · {meta['legendaries']}</div>
                    <div class="s">{meta.get('comuns', meta['rows_clean'] - meta['legendaries'])} comuns seguem no treino/teste</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confusion(result: EvaluationResult) -> None:
    st.markdown(
        f"""
        <div class="cm-wrap">
            <div class="cm"><div class="k">Verdadeiros negativos</div><div class="v" style="color:#6ea8fe">{result.tn}</div></div>
            <div class="cm"><div class="k">Falsos positivos · alarmes</div><div class="v" style="color:#ff6b6b">{result.fp}</div></div>
            <div class="cm"><div class="k">Falsos negativos · perdidos</div><div class="v" style="color:#E8B923">{result.fn}</div></div>
            <div class="cm"><div class="k">Verdadeiros positivos · acertos</div><div class="v" style="color:#3dd68c">{result.tp}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
