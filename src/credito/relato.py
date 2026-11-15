"""Simulador de Inadimplência: duas respostas e gatilhos preventivos.

A mesa explicativa fica recolhida. O simulador permanece aberto na página.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.credito.chaves import FILHOS, GATILHOS, PARES, PRESETS, coerente, grupos, respostas_vazias, vetor
from src.credito.politica import CORTE_MATRIZ, nome_faixa
from src.credito.simulador import (
    amostra_teste,
    assinatura_modelos,
    carregar_modelos,
    chances,
    figura,
    ler_relatorio,
    pronto,
    resolver_modelos,
    resolver_teste,
)

COR_FAIXA = {"verde": "#3dd68c", "amarelo": "#E8B923", "vermelho": "#ff8a65"}
NOME_FAIXA = {"verde": "Verde", "amarelo": "Amarelo", "vermelho": "Vermelho"}
ORDEM_ALVOS = ("inadimplente", "inadimplente_90d")

CSS = """
<style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    section.main .block-container { max-width: 1360px !important; padding-top: 0.8rem; }
    .sintetico {
        border: 1px solid rgba(232,185,35,0.4);
        background: rgba(232,185,35,0.08);
        color: #F4F1E8; border-radius: 12px;
        padding: 0.7rem 0.9rem; margin: 0.2rem 0 0.8rem; line-height: 1.45;
    }
    .sintetico strong { color: #E8B923; }
    .acc {
        background: #163024; border-radius: 14px; padding: 0.75rem 0.8rem 0.65rem;
        border-top: 3px solid #2a4a38; min-height: 168px;
    }
    .acc.melhor { border-top-color: #E8B923; }
    .acc .nome { color: #c9c3b4; font-size: 0.82rem; }
    .acc .grande { color: #F4F1E8; font-size: 2rem; font-weight: 700; line-height: 1.1; margin: 0.2rem 0; }
    .acc .miudo { color: #c9c3b4; font-size: 0.78rem; line-height: 1.45; }
    .selo { color: #E8B923; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .rank-nome { color: #F4F1E8; font-size: 0.84rem; }
    .rank-barra { height: 6px; border-radius: 99px; background: #0f1f18; margin: 0.12rem 0 0.45rem; }
    .rank-barra span { display: block; height: 100%; border-radius: 99px; background: #E8B923; }
    .combo {
        background: #163024; border-radius: 14px; padding: 0.8rem 0.9rem; margin-bottom: 0.65rem;
        border-left: 4px solid #ff8a65;
    }
    .combo h3 { margin: 0 0 0.35rem; color: #F4F1E8; font-size: 1rem; }
    .combo p { margin: 0.15rem 0; color: #c9c3b4; font-size: 0.88rem; }
    .combo strong { color: #F4F1E8; }
    .dupla { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin-bottom: 0.6rem; }
    .veredito {
        background: #163024; border-radius: 16px; padding: 0.85rem 1rem 0.75rem;
        border: 1px solid rgba(244,241,232,0.08);
    }
    .veredito .pergunta { color: #c9c3b4; font-size: 0.92rem; margin: 0; }
    .veredito .grande { font-size: 2.4rem; font-weight: 700; line-height: 1.05; margin: 0.2rem 0 0; }
    .veredito .chamada { font-size: 1.2rem; font-weight: 700; margin: 0.15rem 0 0; }
    .sim-grupo { color: #E8B923; font-weight: 700; font-size: 0.92rem; margin: 0.35rem 0 0.15rem; }
    @media (max-width: 900px) { .dupla { grid-template-columns: 1fr; } }
</style>
"""


def _pct(valor: float | None) -> str:
    if valor is None:
        return "—"
    return f"{100 * valor:.1f}%".replace(".", ",")


def _num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _mil(valor: float) -> str:
    return f"{int(round(valor)):,}".replace(",", ".")


def _pp(valor: float) -> str:
    return f"{valor:.1f}%".replace(".", ",")


@st.cache_resource(show_spinner=False)
def _modelos_gravados(assinatura: str):
    if not assinatura:
        raise RuntimeError("assinatura vazia")
    return carregar_modelos()


@st.cache_data(show_spinner=False)
def _teste(assinatura: str) -> pd.DataFrame:
    if not assinatura:
        raise RuntimeError("assinatura vazia")
    return amostra_teste()


def _modelos_atuais() -> dict:
    assinatura = assinatura_modelos()
    return resolver_modelos(
        lambda: _modelos_gravados(assinatura),
        _modelos_gravados.clear,
        carregar_modelos,
    )


def _amostra_atual() -> pd.DataFrame:
    assinatura = assinatura_modelos()
    return resolver_teste(
        lambda: _teste(assinatura),
        _teste.clear,
        amostra_teste,
    )


def _chave(geracao: int, ident: str) -> str:
    return f"g{geracao}_{ident}"


def _nova_geracao(respostas: dict[str, bool], caso: dict | None) -> None:
    geracao = int(st.session_state.get("sim_geracao", 0)) + 1
    st.session_state["sim_geracao"] = geracao
    st.session_state["sim_respostas"] = coerente(respostas)
    st.session_state["sim_caso"] = caso


def _sincronizar(geracao: int) -> None:
    for filho, pai in PARES:
        chave_filho = _chave(geracao, filho)
        chave_pai = _chave(geracao, pai)
        if st.session_state.get(chave_filho) and chave_pai in st.session_state:
            st.session_state[chave_pai] = True
    for pai, filhos in FILHOS.items():
        chave_pai = _chave(geracao, pai)
        if chave_pai in st.session_state and not st.session_state[chave_pai]:
            for filho in filhos:
                chave_filho = _chave(geracao, filho)
                if chave_filho in st.session_state:
                    st.session_state[chave_filho] = False


def _cb_filho(geracao: int, filho: str, pai: str):
    def _fn() -> None:
        if st.session_state[_chave(geracao, filho)]:
            st.session_state[_chave(geracao, pai)] = True

    return _fn


def _cb_pai(geracao: int, pai: str, filhos: list[str]):
    def _fn() -> None:
        if not st.session_state[_chave(geracao, pai)]:
            for filho in filhos:
                st.session_state[_chave(geracao, filho)] = False

    return _fn


def _respostas(geracao: int) -> dict[str, bool]:
    return {item["id"]: bool(st.session_state[_chave(geracao, item["id"])]) for item in GATILHOS}


def _cabecalho(relatorio: dict, ir) -> None:
    if st.button("← Voltar ao menu inicial", key="credito_voltar"):
        ir("home")
    st.title("Simulador de Inadimplência")
    base = relatorio["base"]
    st.markdown(
        '<div class="sintetico"><strong>Base sintética de 50 mil clientes.</strong> '
        "O estudo responde duas perguntas com os mesmos quatro modelos: "
        "vai ficar inadimplente, e vai passar de 90 dias. "
        "Quem passa de 90 dias já está inadimplente. "
        "Não entra renda declarada, saldo, fluxo de caixa nem atraso que já se instalou. "
        "A mesa olha só gatilhos preventivos."
        "</div>",
        unsafe_allow_html=True,
    )
    kpis = [
        ("Clientes", _mil(base["n"])),
        ("Inadimplência", _pct(base["taxa_inadimplente"])),
        ("Acima de 90 dias", _pct(base["taxa_90d"])),
        ("Gatilhos preventivos", _mil(base["n_gatilhos"])),
    ]
    cols = st.columns(4)
    for col, (nome, valor) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi"><div class="label">{nome}</div><div class="value">{valor}</div></div>',
            unsafe_allow_html=True,
        )


def _mesa() -> None:
    with st.expander("O que a mesa avalia", expanded=False):
        st.caption(
            "Lista dos gatilhos preventivos. Se uma reincidência fica em Sim, o gatilho base também fica."
        )
        linhas = []
        for nome, itens in grupos():
            linhas.append(f"**{nome}**")
            linhas.extend(f"- {item['texto']}" for item in itens)
            linhas.append("")
        st.markdown("\n".join(linhas))


def _acuracia(pacote: dict) -> None:
    st.write(
        f"No teste, o acerto usa corte de {_pct(CORTE_MATRIZ)}. "
        "A classe em dia é a maioria, então a acurácia sozinha não conta a história."
    )
    melhor = max(pacote["modelos"], key=lambda modelo: modelo["matriz"]["acuracia"])["slug"]
    cols = st.columns(4)
    for col, modelo in zip(cols, pacote["modelos"]):
        corte = modelo["matriz"]
        classe = "acc melhor" if modelo["slug"] == melhor else "acc"
        selo = '<div class="selo">Maior acurácia</div>' if modelo["slug"] == melhor else ""
        col.markdown(
            f'<div class="{classe}">{selo}'
            f'<div class="nome">{modelo["nome"]}</div>'
            f'<div class="grande">{_pct(corte["acuracia"])}</div>'
            f'<div class="miudo">AUC {_num(modelo["roc_auc"])}<br>'
            f'Precisão {_pct(corte["precisao"])} · Recall {_pct(corte["recall"])}<br>'
            f'F1 {_pct(corte["f1"])}</div></div>',
            unsafe_allow_html=True,
        )
    st.image(str(figura(f"matrizes_{pacote['id']}.png")), width="stretch")


def _ranking(pacote: dict) -> None:
    st.markdown("**O que cada modelo aponta, em percentual dentro de 100%**")
    st.caption("Do maior para o menor. A lista inteira de cada modelo soma 100%.")
    nomes = [modelo["slug"] for modelo in pacote["modelos"]]
    rotulo = {modelo["slug"]: modelo["nome"] for modelo in pacote["modelos"]}
    for linha in (nomes[:2], nomes[2:]):
        cols = st.columns(2)
        for col, slug in zip(cols, linha):
            with col:
                st.markdown(f"**{rotulo[slug]}**")
                ranking = sorted(pacote["importancia"]["itens"][slug], key=lambda item: item["pct"], reverse=True)
                maior = ranking[0]["pct"] if ranking else 1
                for item in ranking:
                    largura = 100.0 * item["pct"] / maior if maior else 0
                    st.markdown(
                        f'<div class="rank-nome">{item["rotulo"]} '
                        f'<strong style="color:#E8B923">{_pp(item["pct"])}</strong></div>'
                        f'<div class="rank-barra"><span style="width:{largura:.1f}%"></span></div>',
                        unsafe_allow_html=True,
                    )


def _metricas(relatorio: dict) -> None:
    st.subheader("Os quatro modelos, nas duas respostas")
    abas = st.tabs(["Inadimplência geral", "Acima de 90 dias"])
    for aba, slug in zip(abas, ORDEM_ALVOS):
        with aba:
            _acuracia(relatorio["alvos"][slug])
            _ranking(relatorio["alvos"][slug])


def _combinacoes(relatorio: dict) -> None:
    st.subheader("Quando os sinais se juntam")
    base_inad = relatorio["base"]["taxa_teste_inad"]
    base_90 = relatorio["base"]["taxa_teste_90"]
    st.write(
        f"Medido no teste. A carteira está em {_pct(base_inad)} de inadimplência "
        f"e {_pct(base_90)} acima de 90 dias. A barra mostra o salto observado."
    )
    for combo in relatorio["combinacoes"]:
        if not combo.get("critica"):
            continue
        st.markdown(
            f'<div class="combo"><h3>{combo["titulo"]}</h3>'
            f'<p>{" · ".join(combo["sinais"])}</p>'
            f'<p><strong>{_mil(combo["n"])} clientes</strong> no teste. '
            f'Inadimplência {_pct(combo["taxa_inad"])} '
            f'(o modelo apontou {_pct(combo["prev_inad"])}). '
            f'Acima de 90 dias {_pct(combo["taxa_90"])} '
            f'(o modelo apontou {_pct(combo["prev_90"])}).</p></div>',
            unsafe_allow_html=True,
        )
    caminho = figura("combinacoes.png")
    if caminho.exists():
        st.image(str(caminho), width="stretch")


def _cartao(pergunta: str, prob: float) -> str:
    faixa, acao = nome_faixa(prob)
    chamada = "SIM" if prob >= CORTE_MATRIZ else "NÃO"
    return (
        f'<article class="veredito">'
        f'<p class="pergunta">{pergunta}</p>'
        f'<p class="grande" style="color:{COR_FAIXA[faixa]}">{_pct(prob)}</p>'
        f'<p class="chamada" style="color:{COR_FAIXA[faixa]}">{chamada}</p>'
        f'<p class="chamada" style="color:{COR_FAIXA[faixa]}">{NOME_FAIXA[faixa]} · {acao}</p>'
        f"</article>"
    )


def _placar(relatorio: dict, linha: dict, modelos: dict, caso: dict | None) -> None:
    slug = st.selectbox(
        "Modelo em destaque",
        [modelo["slug"] for modelo in relatorio["alvos"]["inadimplente"]["modelos"]],
        format_func=lambda item: next(
            modelo["nome"] for modelo in relatorio["alvos"]["inadimplente"]["modelos"] if modelo["slug"] == item
        ),
        key="sim_modelo",
    )
    p_inad, p_90 = chances(modelos, slug, linha)
    st.markdown(
        '<div class="dupla">'
        + _cartao("1. Vai ficar inadimplente?", p_inad)
        + _cartao("2. Vai passar de 90 dias?", p_90)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Sim e Não usam o corte de {_pct(CORTE_MATRIZ)}, o mesmo da matriz. A faixa de alerta acende antes.")
    cols = st.columns(4)
    for col, modelo in zip(cols, relatorio["alvos"]["inadimplente"]["modelos"]):
        pi, pg = chances(modelos, modelo["slug"], linha)
        si = "Sim" if pi >= CORTE_MATRIZ else "Não"
        sg = "Sim" if pg >= CORTE_MATRIZ else "Não"
        col.metric(modelo["nome"], f"{_pct(pi)} · {si}")
        col.caption(f"90 dias {_pct(pg)} · {sg}")
    if caso is None:
        return
    real_inad = "Sim" if caso["inad"] else "Não"
    real_90 = "Sim" if caso["d90"] else "Não"
    acertou_inad = "acertou" if (p_inad >= CORTE_MATRIZ) == bool(caso["inad"]) else "errou"
    acertou_90 = "acertou" if (p_90 >= CORTE_MATRIZ) == bool(caso["d90"]) else "errou"
    st.caption(
        f"Cliente {caso['id']} do teste: inadimplente {real_inad}, acima de 90 dias {real_90}. "
        f"Este modelo {acertou_inad} na primeira resposta e {acertou_90} na segunda."
    )


@st.fragment
def _simulador(relatorio: dict) -> None:
    st.subheader("Simulador de Inadimplência")
    st.write("Cada chave é Sim ou Não. As duas respostas mudam na hora.")
    if "sim_geracao" not in st.session_state:
        _nova_geracao(respostas_vazias(), None)
    geracao = int(st.session_state["sim_geracao"])
    guardadas = coerente(st.session_state.get("sim_respostas") or respostas_vazias())
    for item in GATILHOS:
        chave = _chave(geracao, item["id"])
        if chave not in st.session_state:
            st.session_state[chave] = bool(guardadas[item["id"]])
    _sincronizar(geracao)

    modelos = _modelos_atuais()
    linha = vetor(_respostas(geracao))
    _placar(relatorio, linha, modelos, st.session_state.get("sim_caso"))

    botoes = st.columns(4)
    acoes = list(PRESETS) + [{"id": "sortear", "texto": "Sortear cliente do teste", "ligar": []}]
    clicados = []
    for coluna, preset in zip(botoes, acoes):
        with coluna:
            clicados.append(st.button(preset["texto"], width="stretch", key=f"preset_{preset['id']}"))
    if clicados[0]:
        _nova_geracao(respostas_vazias(), None)
        st.rerun(scope="fragment")
    if clicados[1]:
        _nova_geracao({item: True for item in PRESETS[1]["ligar"]}, None)
        st.rerun(scope="fragment")
    if clicados[2]:
        _nova_geracao({item: True for item in PRESETS[2]["ligar"]}, None)
        st.rerun(scope="fragment")
    if clicados[3]:
        teste = _amostra_atual()
        row = teste.iloc[int(np.random.default_rng().integers(0, len(teste)))]
        _nova_geracao(
            {item["id"]: bool(int(row[item["id"]])) for item in GATILHOS},
            {
                "id": str(row["id_cliente"]),
                "inad": int(row["inadimplente"]),
                "d90": int(row["inadimplente_90d"]),
            },
        )
        st.rerun(scope="fragment")

    colunas = st.columns(3)
    blocos = grupos()
    distribuicao = [
        blocos[0:1] + blocos[4:5],
        blocos[1:2],
        blocos[2:4],
    ]
    for indice, fatia in enumerate(distribuicao):
        with colunas[indice]:
            for nome, itens in fatia:
                st.markdown(f'<p class="sim-grupo">{nome}</p>', unsafe_allow_html=True)
                for item in itens:
                    callback = None
                    if item["pai"]:
                        callback = _cb_filho(geracao, item["id"], item["pai"])
                    elif item["id"] in FILHOS:
                        callback = _cb_pai(geracao, item["id"], FILHOS[item["id"]])
                    st.toggle(
                        item["texto"],
                        key=_chave(geracao, item["id"]),
                        on_change=callback,
                    )


def render_relato(ir) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if not pronto():
        st.error("O dossiê ainda não foi gerado. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    relatorio = ler_relatorio()
    alvos = relatorio.get("alvos") or {}
    itens = alvos.get("inadimplente", {}).get("importancia", {}).get("itens", {})
    if "chaves" not in relatorio or not itens.get("logistica") or "pct" not in itens["logistica"][0]:
        st.error("O dossiê está incompleto. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    _cabecalho(relatorio, ir)
    _mesa()
    _metricas(relatorio)
    _combinacoes(relatorio)
    _simulador(relatorio)
