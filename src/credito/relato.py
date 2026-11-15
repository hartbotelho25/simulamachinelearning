"""Estudo de inadimplência e a tela do simulador, em páginas separadas.

A mesa explicativa fica recolhida. O simulador ocupa a própria página, sem rolagem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.credito.chaves import (
    FILHOS,
    GATILHOS,
    PARES,
    PRESETS,
    coerente,
    grupos,
    respostas_vazias,
    vetor,
)
from src.credito.politica import CORTE_MATRIZ, nome_faixa
from src.credito.simulador import (
    ModelosIncompativeis,
    amostra_teste,
    assinatura_modelos,
    carregar_modelos,
    chances,
    ler_relatorio,
    pesos_do_slug,
    pronto,
    resolver_modelos,
    resolver_teste,
    treinar_em_memoria,
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
    .faixas-comite { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.65rem; margin: 0.15rem 0 0.85rem; }
    .faixas-comite article { border-radius: 12px; padding: 0.7rem 0.8rem; }
    .faixas-comite strong { display: block; color: #F4F1E8; font-size: 0.95rem; }
    .faixas-comite p { margin: 0.28rem 0 0; color: #F4F1E8; font-size: 0.86rem; line-height: 1.35; }
    .faixas-comite .verde { background: #146c43; }
    .faixas-comite .amarelo { background: #8a6a12; }
    .faixas-comite .vermelho { background: #8c3b2c; }
    [data-testid="stExpander"] {
        border: 1px solid #E8B923 !important;
        border-left: 6px solid #E8B923 !important;
        border-radius: 14px !important;
        background: rgba(232, 185, 35, 0.12) !important;
        margin: 0.35rem 0 1rem !important;
    }
    [data-testid="stExpander"]:hover {
        background: rgba(232, 185, 35, 0.22) !important;
        box-shadow: 0 0 0 1px #ffe08a;
    }
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #F4F1E8 !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
    }
    [data-testid="stExpander"] summary svg { fill: #E8B923 !important; }
    .matrizes { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.65rem; margin: 0.35rem 0 0.75rem; }
    .matriz { background: #10261c; border-radius: 12px; padding: 0.55rem 0.55rem 0.6rem; border: 1px solid rgba(244,241,232,0.08); }
    .matriz h3 { margin: 0 0 0.4rem; color: #F4F1E8; font-size: 0.92rem; font-weight: 700; }
    .grade { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; }
    .cel { border-radius: 8px; padding: 0.42rem 0.45rem 0.4rem; min-height: 86px; }
    .cel span, .cel em { display: block; color: #F4F1E8; font-size: 0.72rem; line-height: 1.25; font-style: normal; }
    .cel strong { display: block; color: #F4F1E8; font-size: 1.2rem; line-height: 1.15; margin: 0.12rem 0; }
    .cel.vn { background: #146c43; }
    .cel.fp { background: #8a6a12; }
    .cel.fn { background: #8c3b2c; }
    .cel.vp { background: #1d4f8a; }
    .lendas, .insights { display: grid; gap: 0.65rem; margin: 0 0 0.85rem; }
    .lendas { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .insights { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .lenda, .insight {
        background: #163024; border-radius: 12px; padding: 0.7rem 0.75rem 0.75rem;
        border-top: 4px solid #3dd68c;
    }
    .lenda.vp, .insight.vp { border-top-color: #6ea8fe; }
    .lenda.fp, .insight.fp { border-top-color: #E8B923; }
    .lenda.fn, .insight.fn { border-top-color: #ff8a65; }
    .lenda strong, .insight h3 { display: block; color: #F4F1E8; font-size: 0.92rem; margin: 0; line-height: 1.3; }
    .lenda p, .insight p { color: #F4F1E8; font-size: 0.84rem; line-height: 1.4; margin: 0.28rem 0 0; }
    .insight .num { color: #E8B923; font-size: 1.8rem; font-weight: 700; line-height: 1; }
    .insight h3 { margin-top: 0.3rem; }
    @media (max-width: 1100px) {
        .matrizes, .lendas { grid-template-columns: 1fr 1fr; }
        .insights { grid-template-columns: 1fr; }
    }
    @media (max-width: 700px) {
        .matrizes, .lendas { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) { .dupla { grid-template-columns: 1fr; } }
</style>
"""

ROTULO_MESA = (
    "📋 O que a mesa avalia — 23 gatilhos preventivos · Clique para abrir a lista completa"
)


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
    try:
        return carregar_modelos()
    except (ModelosIncompativeis, AttributeError):
        return treinar_em_memoria()


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
    voltar, abrir = st.columns(2)
    with voltar:
        if st.button("← Voltar ao menu inicial", key="credito_voltar", width="stretch"):
            ir("home")
    with abrir:
        if st.button("Abrir o Simulador de Inadimplência", key="ir_simulador", type="primary", width="stretch"):
            ir("simulador")
    st.title("Machine Learning Aplicado à Inadimplência")
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
    with st.expander(ROTULO_MESA, expanded=False):
        st.caption("Os 23 gatilhos preventivos. Cada um entra no modelo como Sim ou Não.")
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
    st.markdown(_matriz_html(pacote), unsafe_allow_html=True)


def _ranking(pacote: dict) -> None:
    blocos = (pacote.get("importancia") or {}).get("blocos") or {}
    if blocos:
        st.markdown("**O que cada bloco pesa, em percentual dentro de 100%**")
        st.caption("Do maior para o menor. Os cinco blocos de cada modelo somam 100%.")
        nomes = [modelo["slug"] for modelo in pacote["modelos"]]
        rotulo = {modelo["slug"]: modelo["nome"] for modelo in pacote["modelos"]}
        cols = st.columns(4)
        for col, slug in zip(cols, nomes):
            with col:
                st.markdown(f"**{rotulo[slug]}**")
                ranking = blocos.get(slug) or []
                maior = ranking[0]["pct"] if ranking else 1
                for item in ranking:
                    largura = 100.0 * item["pct"] / maior if maior else 0
                    st.markdown(
                        f'<div class="rank-nome">{item["nome"]} '
                        f'<strong style="color:#E8B923">{_pp(item["pct"])}</strong></div>'
                        f'<div class="rank-barra"><span style="width:{largura:.1f}%"></span></div>',
                        unsafe_allow_html=True,
                    )
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


_QUADS = (
    ("vn", "tn", "Verdadeiro negativo", "Acerto — bom pagador"),
    ("fp", "fp", "Falso positivo", "Alarme falso"),
    ("fn", "fn", "Falso negativo", "Risco que escapou"),
    ("vp", "tp", "Verdadeiro positivo", "Inadimplente capturado"),
)


def _matriz_html(pacote: dict) -> str:
    blocos = []
    for modelo in pacote["modelos"]:
        corte = modelo["matriz"]
        celulas = "".join(
            f'<div class="cel {classe}"><span>{nome}</span>'
            f"<strong>{_mil(corte[chave])}</strong><em>{curto}</em></div>"
            for classe, chave, nome, curto in _QUADS
        )
        blocos.append(f'<article class="matriz"><h3>{modelo["nome"]}</h3><div class="grade">{celulas}</div></article>')
    lendas = """
    <div class="lendas">
      <article class="lenda vn"><strong>Verdadeiro Negativo (Acerto — Bom pagador)</strong>
        <p>Ficou em dia e o modelo aprovou.</p></article>
      <article class="lenda vp"><strong>Verdadeiro Positivo (Acerto — Inadimplente capturado)</strong>
        <p>Atrasou e o modelo alertou antes.</p></article>
      <article class="lenda fp"><strong>Falso Positivo (Alarme falso)</strong>
        <p>Ficou em dia, mas o modelo marcou risco.</p></article>
      <article class="lenda fn"><strong>Falso Negativo (Risco que escapou)</strong>
        <p>Ficou inadimplente, mas passou sem alerta do modelo.</p></article>
    </div>
    """
    return f'<div class="matrizes">{"".join(blocos)}</div>{lendas}'


_RAIZ = Path(__file__).resolve().parents[2]
_REPO = "https://github.com/hartbotelho25/simulamachinelearning"
_MODELOS_CODIGO = (
    ("regressao_logistica", "Regressão logística"),
    ("random_forest", "Random Forest"),
    ("gradient_boosting", "Gradient Boosting (XGBoost)"),
    ("rede_neural", "Rede neural MLP"),
)


def _ler_arquivo(relativo: str) -> str:
    return (_RAIZ / relativo).read_text(encoding="utf-8")


def _codigo() -> None:
    st.subheader("O código")
    st.caption(
        "Cada modelo é um script completo, recolhido. "
        "Baixe o .py para o VS Code ou o .ipynb para o Google Colab. "
        "Os dois leem a base direto do GitHub, separam treino e teste, treinam "
        "e imprimem AUC, acurácia e a matriz de confusão. Não gravam por cima do modelo do site."
    )
    for slug, titulo in _MODELOS_CODIGO:
        with st.expander(titulo, expanded=False):
            py = _RAIZ / "notebooks" / f"{slug}.py"
            ipynb = _RAIZ / "notebooks" / f"{slug}.ipynb"
            st.code(py.read_text(encoding="utf-8"), language="python")
            coluna_py, coluna_nb = st.columns(2)
            coluna_py.download_button(
                "Baixar .py",
                data=py.read_bytes(),
                file_name=py.name,
                mime="text/x-python",
                key=f"dl_py_{slug}",
            )
            coluna_nb.download_button(
                "Baixar .ipynb",
                data=ipynb.read_bytes(),
                file_name=ipynb.name,
                mime="application/x-ipynb+json",
                key=f"dl_ipynb_{slug}",
            )
    st.markdown(f"[Repositório]({_REPO})")


def _veredito(pergunta: str, prob: float) -> str:
    faixa, postura = nome_faixa(prob)
    chamada = "SIM" if prob >= CORTE_MATRIZ else "NÃO"
    return (
        f'<article class="veredito">'
        f'<p class="pergunta">{pergunta}</p>'
        f'<p class="grande" style="color:{COR_FAIXA[faixa]}">{_pct(prob)}</p>'
        f'<p class="chamada" style="color:{COR_FAIXA[faixa]}">{chamada} · {NOME_FAIXA[faixa]} · {postura}</p>'
        f"</article>"
    )


def _quadro(relatorio: dict, linha: dict, modelos: dict, caso: dict | None) -> None:
    slug = st.session_state.get("sim_modelo", "logistica")
    p_inad, p_90 = chances(modelos, slug, linha)
    linhas = []
    for modelo in relatorio["alvos"]["inadimplente"]["modelos"]:
        pi, pg = chances(modelos, modelo["slug"], linha)
        si = "Sim" if pi >= CORTE_MATRIZ else "Não"
        sg = "Sim" if pg >= CORTE_MATRIZ else "Não"
        destaque = " class='destaque'" if modelo["slug"] == slug else ""
        linhas.append(
            f"<tr{destaque}><td>{modelo['nome']}</td>"
            f"<td>{_pct(pi)} · {si}</td><td>{_pct(pg)} · {sg}</td></tr>"
        )
    ligados = sum(int(valor) for valor in linha.values())
    pesos = pesos_do_slug(modelos, slug)
    ativos = [item for item in GATILHOS if linha.get(item["id"])]
    ativos.sort(key=lambda item: pesos.get(item["id"], 0.0), reverse=True)
    puxam = ""
    if ativos:
        puxam = "<p class='puxam'>" + " · ".join(item["rotulo"] for item in ativos[:3]) + "</p>"
    nota = ""
    if caso is not None:
        real_inad = "Sim" if caso["inad"] else "Não"
        real_90 = "Sim" if caso["d90"] else "Não"
        nota = f"<p class='nota'>Cliente {caso['id']}: inadimplente {real_inad}, acima de 90 dias {real_90}.</p>"
    st.markdown(
        '<div class="painel-respostas"><p class="olho">Predição do modelo</p>'
        + '<div class="dupla">'
        + _veredito("Vai ficar inadimplente?", p_inad)
        + _veredito("Vai passar de 90 dias?", p_90)
        + "</div>"
        + "<table class='quadro'><thead><tr><th>Modelo</th><th>Inadimplência</th><th>90 dias</th></tr></thead><tbody>"
        + "".join(linhas)
        + "</tbody></table>"
        + f"<p class='ligados'>{ligados} gatilhos em Sim</p>"
        + puxam
        + nota
        + "</div>",
        unsafe_allow_html=True,
    )


def _sortear() -> None:
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


@st.fragment
def _simulador(relatorio: dict, ir) -> None:
    if "sim_geracao" not in st.session_state:
        _nova_geracao(respostas_vazias(), None)
    geracao = int(st.session_state["sim_geracao"])
    guardadas = coerente(st.session_state.get("sim_respostas") or respostas_vazias())
    for item in GATILHOS:
        chave = _chave(geracao, item["id"])
        if chave not in st.session_state:
            st.session_state[chave] = bool(guardadas[item["id"]])
    _sincronizar(geracao)

    titulo, menu, estudo, modelo, aperto, grave, sortear, limpar = st.columns(
        [2.15, 0.95, 0.95, 1.55, 0.72, 0.68, 0.78, 1.25],
        gap="small",
    )
    titulo.markdown('<p class="cockpit-titulo">Simulador de Inadimplência</p>', unsafe_allow_html=True)
    if menu.button("Menu inicial", width="stretch", key="sim_menu"):
        ir("home")
    if estudo.button("Estudo de ML", width="stretch", key="sim_estudo"):
        ir("ml")
    modelo.selectbox(
        "Modelo principal",
        [item["slug"] for item in relatorio["alvos"]["inadimplente"]["modelos"]],
        format_func=lambda item: next(
            modelo_item["nome"]
            for modelo_item in relatorio["alvos"]["inadimplente"]["modelos"]
            if modelo_item["slug"] == item
        ),
        key="sim_modelo",
        label_visibility="collapsed",
    )
    if aperto.button("Aperto", width="stretch", key="preset_aperto"):
        _nova_geracao({item: True for item in PRESETS[1]["ligar"]}, None)
        st.rerun(scope="fragment")
    if grave.button("Grave", width="stretch", key="preset_grave"):
        _nova_geracao({item: True for item in PRESETS[2]["ligar"]}, None)
        st.rerun(scope="fragment")
    if sortear.button("Sortear", width="stretch", key="preset_sortear"):
        _sortear()
        st.rerun(scope="fragment")
    if limpar.button("Limpar tudo (Não)", width="stretch", key="preset_limpar"):
        _nova_geracao(respostas_vazias(), None)
        st.rerun(scope="fragment")

    modelos = _modelos_atuais()
    linha = vetor(_respostas(geracao))
    gatilhos, respostas = st.columns([1.75, 1], gap="small")
    with gatilhos:
        colunas = st.columns(3, gap="small")
        blocos = grupos()
        distribuicao = [blocos[0:1] + blocos[4:5], blocos[1:2], blocos[2:4]]
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
                        st.toggle(item["texto"], key=_chave(geracao, item["id"]), on_change=callback)
    with respostas:
        _quadro(relatorio, linha, modelos, st.session_state.get("sim_caso"))


def _dossie() -> dict:
    if not pronto():
        st.error("O dossiê ainda não foi gerado. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    relatorio = ler_relatorio()
    alvos = relatorio.get("alvos") or {}
    itens = alvos.get("inadimplente", {}).get("importancia", {}).get("itens", {})
    if "chaves" not in relatorio or not itens.get("logistica") or "pct" not in itens["logistica"][0]:
        st.error("O dossiê está incompleto. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    return relatorio


def render_estudo(ir) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    relatorio = _dossie()
    _cabecalho(relatorio, ir)
    _mesa()
    _metricas(relatorio)
    _codigo()


COCKPIT_CSS = """
<style>
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    footer,
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    [data-testid="stMainBlockContainer"],
    section.main .block-container {
        max-width: 100% !important;
        padding-top: 0.35rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }
    [data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.15rem !important; }
    [data-testid="stElementContainer"] { margin-bottom: 0 !important; }
    .cockpit-titulo {
        color: #F4F1E8; font-size: 1.05rem; font-weight: 700;
        margin: 0.35rem 0 0; line-height: 1.1; white-space: nowrap;
    }
    div[data-testid="stButton"] button {
        min-height: 2rem !important;
        padding: 0.15rem 0.35rem !important;
        font-size: 0.78rem !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] { min-height: 2rem; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div { min-height: 2rem; padding-top: 0; padding-bottom: 0; }
    .sim-grupo {
        color: #E8B923 !important; font-weight: 700 !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.02em; text-transform: uppercase;
        margin: 0.2rem 0 0.05rem !important; line-height: 1.15 !important;
        white-space: normal;
    }
    [data-testid="stMarkdownContainer"]:has(.sim-grupo) {
        margin: 0 0 0.12rem !important;
    }
    [data-testid="stCheckbox"] { height: auto !important; }
    [data-testid="stCheckbox"] label { height: auto !important; align-items: flex-start !important; }
    [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
    [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.74rem !important;
        line-height: 1.15 !important;
        margin: 0 !important;
        height: auto !important;
        max-height: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.painel-respostas) > div[data-testid="stColumn"]:last-child {
        position: sticky; top: 0.75rem; align-self: start;
    }
    .dupla { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; margin: 0 0 0.35rem; }
    .veredito {
        background: #163024; border-radius: 10px; padding: 0.4rem 0.5rem 0.35rem;
        border: 1px solid rgba(244,241,232,0.08);
    }
    .veredito .pergunta { color: #c9c3b4; font-size: 0.75rem; margin: 0; line-height: 1.15; }
    .veredito .grande { font-size: 1.55rem; font-weight: 700; line-height: 1; margin: 0.12rem 0 0; }
    .veredito .chamada { font-size: 0.78rem; font-weight: 700; margin: 0.12rem 0 0; line-height: 1.15; }
    .quadro { width: 100%; border-collapse: collapse; margin: 0.1rem 0 0.2rem; }
    .quadro th, .quadro td {
        text-align: left; color: #F4F1E8; font-size: 0.72rem;
        padding: 0.16rem 0.2rem; border-bottom: 1px solid rgba(244,241,232,0.08);
        line-height: 1.15;
    }
    .quadro th { color: #c9c3b4; font-weight: 600; }
    .quadro tr.destaque td { color: #E8B923; }
    .olho {
        margin: 0 0 0.2rem; color: #c9c3b4; font-size: 0.68rem;
        font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .ligados { color: #F4F1E8; font-size: 0.78rem; font-weight: 700; margin: 0.15rem 0 0; }
    .puxam { color: #c9c3b4; font-size: 0.68rem; line-height: 1.2; margin: 0.08rem 0 0; }
    .nota { color: #c9c3b4; font-size: 0.7rem; margin: 0.1rem 0 0; line-height: 1.2; }
</style>
"""


def render_simulador(ir) -> None:
    st.markdown(COCKPIT_CSS, unsafe_allow_html=True)
    relatorio = _dossie()
    _simulador(relatorio, ir)
