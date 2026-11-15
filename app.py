"""Menu: Simula Brasileirão, o estudo de inadimplência e o simulador."""

from __future__ import annotations

import streamlit as st

from src.br_ui import inject_css, render_chips

st.set_page_config(
    page_title="Simulador de Machine Learning · Hart Botelho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MENU_CSS = """
<style>
    .entry {
        background: radial-gradient(420px 160px at 0% 0%, rgba(232,185,35,0.12), transparent),
                    linear-gradient(180deg, #163024 0%, #10261c 100%);
        border-radius: 18px;
        padding: 1.15rem 1.2rem 1rem;
        min-height: 248px;
        border: 1px solid rgba(244,241,232,0.08);
    }
    .entry.br { border-top: 3px solid #3dd68c; }
    .entry.ml { border-top: 3px solid #E8B923; }
    .entry.sim { border-top: 3px solid #ff8a65; }
    .entry h2 { color: #F4F1E8; font-size: 1.28rem; margin: 0.45rem 0 0.4rem; }
    .entry p { color: #c9c3b4; margin: 0; line-height: 1.45; font-size: 0.95rem; }
    .entry ul { margin: 0.75rem 0 0; padding-left: 1.1rem; color: #F4F1E8; }
    .entry li { margin: 0.22rem 0; font-size: 0.9rem; }
    .pill {
        display: inline-block; border-radius: 999px; padding: 0.12rem 0.65rem;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .pill.br { background: rgba(61,214,140,0.14); color: #3dd68c; border: 1px solid rgba(61,214,140,0.45); }
    .pill.ml { background: rgba(232,185,35,0.12); color: #E8B923; border: 1px solid rgba(232,185,35,0.45); }
    .pill.sim { background: rgba(255,138,101,0.12); color: #ff8a65; border: 1px solid rgba(255,138,101,0.45); }
</style>
"""

_DESTINOS = {"home", "br", "ml", "simulador", "credito"}


def _estudo() -> str:
    bruto = st.query_params.get("estudo", "home")
    if isinstance(bruto, (list, tuple)):
        bruto = bruto[0] if bruto else "home"
    bruto = str(bruto)
    return bruto if bruto in _DESTINOS else "home"


def _ir(destino: str) -> None:
    st.query_params["estudo"] = destino
    st.rerun()


def _cartao(classe: str, selo: str, titulo: str, texto: str, itens: list[str]) -> None:
    lis = "".join(f"<li>{item}</li>" for item in itens)
    st.markdown(
        f"""
        <div class="entry {classe}">
            <span class="pill {classe}">{selo}</span>
            <h2>{titulo}</h2>
            <p>{texto}</p>
            <ul>{lis}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    with st.sidebar:
        st.markdown("**Desenvolvido por Hart Botelho**")
        st.caption("Três entradas no mesmo portal.")
        st.divider()
        st.markdown("**Simula Brasileirão**")
        st.caption("Série A 2003–2025, com treino ao vivo e PDF.")
        st.markdown("**Machine Learning Aplicado à Inadimplência**")
        st.caption("Estudo dos quatro modelos nas duas respostas.")
        st.markdown("**Simulador de Inadimplência**")
        st.caption("Vinte e três chaves e as duas chances na mesma tela.")

    st.markdown('<p class="credit-top">Desenvolvido por Hart Botelho</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Menu inicial</div>
            <h1>Por onde você entra?</h1>
            <p>
                Três caminhos, o mesmo visual. Simula Brasileirão abre a Série A.
                Machine Learning Aplicado à Inadimplência abre o estudo.
                O Simulador de Inadimplência abre as duas respostas na hora.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    serie, estudo, simulador = st.columns(3, gap="medium")
    with serie:
        _cartao(
            "br",
            "Série A",
            "Simula Brasileirão",
            "Série A 2003–2025. A tela seguinte traz o hero, a barra, o treino e o PDF deste estudo.",
            [
                "Alvo, atributos, algoritmos e limiar",
                "Comparativo e PDF com a base real",
                "Volta ao menu quando quiser",
            ],
        )
        if st.button("Entrar em Simula Brasileirão", type="primary", width="stretch", key="go_br"):
            _ir("br")
    with estudo:
        _cartao(
            "ml",
            "Estudo",
            "Machine Learning Aplicado à Inadimplência",
            "Relatório dos quatro modelos nas duas respostas, na base sintética de 50 mil clientes.",
            [
                "Mesa de gatilhos preventivos",
                "Pesos em 100%, do maior para o menor",
                "Quando os sinais se juntam",
            ],
        )
        if st.button("Entrar no estudo", type="primary", width="stretch", key="go_ml"):
            _ir("ml")
    with simulador:
        _cartao(
            "sim",
            "Ao vivo",
            "Simulador de Inadimplência",
            "Vinte e três chaves Sim ou Não e as duas chances na mesma tela.",
            [
                "Inadimplência e atraso acima de 90 dias",
                "Quatro modelos lado a lado",
                "Volta ao estudo quando quiser",
            ],
        )
        if st.button("Entrar no Simulador de Inadimplência", type="primary", width="stretch", key="go_simulador"):
            _ir("simulador")

    render_chips(
        "Três entradas",
        ["Simula Brasileirão", "Estudo de inadimplência", "Simulador ao vivo"],
    )


def render_br() -> None:
    from src import br_app

    with st.sidebar:
        if st.button("Voltar ao menu", key="voltar_br", width="stretch"):
            _ir("home")
    br_app.main()


def render_ml() -> None:
    from src.credito.relato import render_estudo

    inject_css()
    render_estudo(_ir)


def render_simulador_pagina() -> None:
    from src.credito.relato import render_simulador

    inject_css()
    render_simulador(_ir)


def main() -> None:
    destino = _estudo()
    if destino == "br":
        render_br()
        return
    if destino in {"ml", "credito"}:
        render_ml()
        return
    if destino == "simulador":
        render_simulador_pagina()
        return
    inject_css()
    st.markdown(MENU_CSS, unsafe_allow_html=True)
    render_home()


main()
