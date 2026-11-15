"""Menu do simulador: Simula Brasileirão e Simulador de Inadimplência."""

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
    .entry.credito { border-top: 3px solid #E8B923; }
    .entry h2 { color: #F4F1E8; font-size: 1.35rem; margin: 0.45rem 0 0.4rem; }
    .entry p { color: #c9c3b4; margin: 0; line-height: 1.45; font-size: 0.95rem; }
    .entry ul { margin: 0.75rem 0 0; padding-left: 1.1rem; color: #F4F1E8; }
    .entry li { margin: 0.22rem 0; font-size: 0.9rem; }
    .pill {
        display: inline-block; border-radius: 999px; padding: 0.12rem 0.65rem;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .pill.br { background: rgba(61,214,140,0.14); color: #3dd68c; border: 1px solid rgba(61,214,140,0.45); }
    .pill.credito { background: rgba(232,185,35,0.12); color: #E8B923; border: 1px solid rgba(232,185,35,0.45); }
</style>
"""

_DESTINOS = {"home", "br", "credito"}


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
        st.caption("Dois estudos no mesmo portal.")
        st.divider()
        st.markdown("**Simula Brasileirão**")
        st.caption("Série A 2003–2025, com treino ao vivo e PDF.")
        st.markdown("**Simulador de Inadimplência**")
        st.caption("50.000 clientes, duas respostas e gatilhos preventivos. Sem renda nem saldo.")

    st.markdown('<p class="credit-top">Desenvolvido por Hart Botelho</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Menu inicial</div>
            <h1>Por onde você entra?</h1>
            <p>
                Dois caminhos, o mesmo visual. Simula Brasileirão abre a Série A.
                Simulador de Inadimplência responde se o cliente fica inadimplente
                e se passa de 90 dias, só com gatilhos preventivos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    esquerda, direita = st.columns(2, gap="large")
    with esquerda:
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
    with direita:
        _cartao(
            "credito",
            "Base sintética",
            "Simulador de Inadimplência",
            "Duas respostas na mesma base sintética de 50.000 clientes: inadimplência e atraso acima de 90 dias.",
            [
                "Quatro modelos, entre eles uma rede neural",
                "Só gatilhos preventivos, em Sim ou Não",
                "Simulador aberto na página, nas duas respostas",
            ],
        )
        if st.button("Entrar no Simulador de Inadimplência", type="primary", width="stretch", key="go_credito"):
            _ir("credito")

    render_chips(
        "Simulador de Inadimplência",
        ["Base sintética", "50.000 clientes", "Duas respostas", "Gatilhos preventivos", "Simulador"],
    )


def render_br() -> None:
    from src import br_app

    with st.sidebar:
        if st.button("Voltar ao menu", key="voltar_br", width="stretch"):
            _ir("home")
    br_app.main()


def render_credito() -> None:
    from src.credito.relato import render_relato

    inject_css()
    render_relato(_ir)


def main() -> None:
    destino = _estudo()
    if destino == "br":
        render_br()
        return
    if destino == "credito":
        render_credito()
        return
    inject_css()
    st.markdown(MENU_CSS, unsafe_allow_html=True)
    render_home()


main()
