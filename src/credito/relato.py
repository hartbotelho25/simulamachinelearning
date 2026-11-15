"""Simulador de Inadimplência: relato estático e simulador na mesma página.

O placar muda com os controles. Não há gaveta, aba nem formulário congelado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.credito.chaves import PERGUNTAS, aplicar, respostas_da_linha, respostas_em_dia
from src.credito.politica import CORTE_MATRIZ, nome_faixa
from src.credito.simulador import (
    amostra_teste,
    assinatura_modelos,
    carregar_modelos,
    chance,
    figura,
    ler_relatorio,
    pronto,
    resolver_modelos,
    resolver_teste,
)

MESA = [
    {
        "titulo": "Histórico de atrasos no banco",
        "alerta": "Acende com atraso de 30 a 60 dias, atraso na 1ª parcela ou renegociação ativa.",
        "blocos": ["regua"],
    },
    {
        "titulo": "Uso do cartão de crédito",
        "alerta": "Acende com pagamento no mínimo, atraso, saque no crédito ou fatura paga no cheque especial.",
        "blocos": ["cartao"],
    },
    {
        "titulo": "Conta corrente e cheque especial",
        "alerta": "Acende com cheque devolvido, uso contínuo do especial ou saldo e fluxo fracos.",
        "blocos": ["cheque", "fluxo"],
    },
    {
        "titulo": "Empréstimos e peso das parcelas",
        "alerta": "Acende quando as parcelas comem fatia alta das entradas ou o contrato é novo ou atrasado.",
        "blocos": ["endividamento"],
    },
    {
        "titulo": "Restrições no mercado",
        "alerta": "Acende com protesto em cartório ou inadimplência em outra instituição.",
        "blocos": ["bureau"],
    },
    {
        "titulo": "Perfil de relacionamento",
        "alerta": "Acende em cliente recente ou sem imóvel próprio.",
        "blocos": ["cadastral"],
    },
]
COR_FAIXA = {"verde": "#3dd68c", "amarelo": "#E8B923", "vermelho": "#ff8a65"}
NOME_FAIXA = {"verde": "Verde", "amarelo": "Amarelo", "vermelho": "Vermelho"}

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
    .grade-grupos { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin: 0.3rem 0 0.6rem; }
    .grupo {
        background: #163024; border-radius: 14px; padding: 0.75rem 0.9rem 0.55rem;
        border-top: 3px solid #2a4a38;
    }
    .grupo.primeiro { border-top-color: #E8B923; }
    .grupo h3 { margin: 0.15rem 0 0.4rem; color: #F4F1E8; font-size: 1.02rem; }
    .grupo ul { margin: 0; padding: 0; }
    .grupo .alerta { margin: 0; color: #c9c3b4; font-size: 0.9rem; line-height: 1.4; }
    .olhe { color: #E8B923; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .tec { color: #8d887c; font-size: 0.74rem; margin-left: 0.3rem; }
    .ponto { color: #E8B923; margin-right: 0.25rem; }
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
    .placar {
        background: #163024; border-radius: 16px; padding: 0.9rem 1rem 0.75rem;
        border: 1px solid rgba(244,241,232,0.08); margin-bottom: 0.6rem;
    }
    .placar .grande { font-size: 2.6rem; font-weight: 700; line-height: 1; }
    .placar .parecer { font-size: 1.25rem; font-weight: 700; margin-top: 0.15rem; }
    .sim-grupo { color: #E8B923; font-weight: 700; font-size: 0.92rem; margin: 0.2rem 0 0.15rem; }
    @media (max-width: 900px) { .grade-grupos { grid-template-columns: 1fr; } }
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


@st.cache_resource(show_spinner=False)
def _modelos_gravados(assinatura: str):
    # `assinatura` entra na chave. Sem argumento o processo entrega o joblib antigo.
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



def _reais(valor: float) -> str:
    texto = f"{valor:,.0f}".replace(",", ".")
    return f"R$ {texto}"


def _chave(geracao: int, ident: str) -> str:
    return f"g{geracao}_{ident}"


def _respostas(geracao: int) -> dict[str, bool]:
    return {pergunta["id"]: bool(st.session_state[_chave(geracao, pergunta["id"])]) for pergunta in PERGUNTAS}


def _nova_geracao(respostas: dict[str, bool], caso: dict | None) -> None:
    geracao = int(st.session_state.get("sim_geracao", 0)) + 1
    st.session_state["sim_geracao"] = geracao
    st.session_state["sim_respostas"] = respostas
    st.session_state["sim_caso"] = caso


def _sins_que_puxam(relatorio: dict, slug: str, respostas: dict[str, bool]) -> list[str]:
    ranking = {item["variavel"]: item["pct"] for item in relatorio["importancia"]["itens"][slug]}
    sims = relatorio["chaves"]["sim"]
    acesos = []
    for pergunta in PERGUNTAS:
        ident = pergunta["id"]
        if ident == "imovel":
            if respostas.get(ident, True):
                continue
            colunas = ["possui_imovel_proprio"]
        elif not respostas.get(ident):
            continue
        else:
            colunas = list(sims[ident])
        peso = max((ranking.get(coluna, 0.0) for coluna in colunas), default=0.0)
        acesos.append((peso, pergunta["texto"]))
    acesos.sort(key=lambda par: par[0], reverse=True)
    return [texto for _, texto in acesos[:6]]


def _cabecalho(relatorio: dict, ir) -> None:
    if st.button("← Voltar ao menu inicial", key="credito_voltar"):
        ir("home")
    st.title("Simulador de Inadimplência")
    st.markdown(
        '<div class="sintetico"><strong>Base sintética.</strong> '
        "Não é carteira de banco. O modelo não usa renda declarada: o banco quase nunca "
        "vê a renda verdadeira, porque o cliente pode omitir, ter outra conta ou receber fora do banco."
        "</div>",
        unsafe_allow_html=True,
    )
    base = relatorio["base"]
    kpis = [
        ("Clientes", _mil(base["n"])),
        ("Inadimplência em 90 dias", _pct(base["taxa"])),
        ("Fluxo mediano de entradas", _reais(base["fluxo_mediano"])),
        ("Maior acurácia", _pct(max(m["matriz"]["acuracia"] for m in relatorio["modelos"]))),
    ]
    cols = st.columns(4)
    for col, (nome, valor) in zip(cols, kpis):
        col.markdown(f'<div class="kpi"><div class="label">{nome}</div><div class="value">{valor}</div></div>', unsafe_allow_html=True)


def _grupos(relatorio: dict) -> None:
    st.subheader("O que a mesa avalia")
    st.write("A mesa olha o comportamento na conta. Cada cartão diz, numa linha, o que acende alerta.")
    fatia = {
        item["bloco"]: item["fatia"]
        for item in relatorio["importancia"]["por_modelo"][relatorio["referencia"]]
    }
    ordem = sorted(MESA, key=lambda card: sum(fatia.get(bloco, 0) for bloco in card["blocos"]), reverse=True)
    primeiro = sum(fatia.get(bloco, 0) for bloco in ordem[0]["blocos"])
    partes = []
    for card in ordem:
        peso = sum(fatia.get(bloco, 0) for bloco in card["blocos"])
        classe = "grupo primeiro" if peso == primeiro else "grupo"
        selo = '<p class="olhe">Olhar primeiro</p>' if peso == primeiro else ""
        partes.append(
            f'<article class="{classe}">{selo}<h3>{card["titulo"]}</h3>'
            f'<p class="alerta">{card["alerta"]}</p></article>'
        )
    st.markdown(f'<div class="grade-grupos">{"".join(partes)}</div>', unsafe_allow_html=True)


def _acuracia(relatorio: dict) -> None:
    st.subheader("Acurácia dos quatro modelos")
    st.write(
        f"No teste de {_mil(relatorio['base']['n_teste'])} clientes, o acerto é quem o modelo "
        f"classifica certo com corte em {_pct(CORTE_MATRIZ)}. "
        "Marcar todo mundo como em dia já acertaria a maioria, porque o calote é minoria."
    )
    melhor = max(relatorio["modelos"], key=lambda modelo: modelo["matriz"]["acuracia"])["slug"]
    cols = st.columns(4)
    for col, modelo in zip(cols, relatorio["modelos"]):
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
    st.image(str(figura("matrizes_confusao.png")), width="stretch")


def _itens_por_modelo(relatorio: dict) -> None:
    st.subheader("O que cada modelo aponta para o calote")
    st.write(
        "Cada barra é a fatia do peso que puxa a inadimplência. "
        "Em cada modelo as fatias de todos os fatores somam 100%, do maior para o menor."
    )
    nomes = [modelo["slug"] for modelo in relatorio["modelos"]]
    rotulo = {modelo["slug"]: modelo["nome"] for modelo in relatorio["modelos"]}
    for linha in (nomes[:2], nomes[2:]):
        cols = st.columns(2)
        for col, slug in zip(cols, linha):
            with col:
                st.markdown(f"**{rotulo[slug]}**")
                ranking = sorted(relatorio["importancia"]["itens"][slug], key=lambda item: item["pct"], reverse=True)
                top = ranking[:8]
                soma = sum(item["pct"] for item in top)
                maior = top[0]["pct"] if top else 1
                for item in top:
                    largura = 100.0 * item["pct"] / maior if maior else 0
                    texto = f"{item['pct']:.1f}%".replace(".", ",")
                    st.markdown(
                        f'<div class="rank-nome">{item["rotulo"]} <strong style="color:#E8B923">{texto}</strong></div>'
                        f'<div class="rank-barra"><span style="width:{largura:.1f}%"></span></div>',
                        unsafe_allow_html=True,
                    )
                soma_txt = f"{soma:.1f}".replace(".", ",")
                st.caption(
                    f"Estes 8, do maior para o menor, somam {soma_txt}% do peso. O restante fecha os 100%."
                )


def _combinacoes(relatorio: dict) -> None:
    st.subheader("Quando os sinais se juntam")
    criticas = [item for item in relatorio["combinacoes"] if item.get("critica")]
    base = criticas[0]["taxa_base"] if criticas else 0.0
    st.write(
        "Medido no teste, não no treino. A barra coral é o que aconteceu. "
        "A dourada é o que a regressão logística previu para as mesmas pessoas."
    )
    for combo in criticas:
        st.markdown(
            f'<div class="combo"><h3>{combo["titulo"]}</h3>'
            f'<p>{" · ".join(combo["sinais"])}</p>'
            f'<p><strong>{_mil(combo["n"])} clientes</strong> no teste. '
            f'A base do teste está em {_pct(base)}. '
            f'Neste recorte, {_pct(combo["taxa_observada"])} ficaram inadimplentes '
            f'e o modelo apontou {_pct(combo["prevista_referencia"])}.</p></div>',
            unsafe_allow_html=True,
        )
    caminho = figura("combinacoes.png")
    if caminho.exists():
        st.image(str(caminho), width="stretch")


def _placar(relatorio: dict, linha: dict, respostas: dict[str, bool], modelos: dict, caso: dict | None) -> None:
    slug = st.selectbox(
        "Modelo do parecer",
        [modelo["slug"] for modelo in relatorio["modelos"]],
        format_func=lambda item: next(modelo["nome"] for modelo in relatorio["modelos"] if modelo["slug"] == item),
        key="sim_modelo",
    )
    prob = chance(modelos[slug], linha)
    faixa, acao = nome_faixa(prob)
    probs = [(modelo["slug"], chance(modelos[modelo["slug"]], linha)) for modelo in relatorio["modelos"]]
    st.markdown(
        f'<div class="placar"><div class="grande" style="color:{COR_FAIXA[faixa]}">{_pct(prob)}</div>'
        f'<div class="parecer" style="color:{COR_FAIXA[faixa]}">{NOME_FAIXA[faixa]} · {acao}</div>'
        f'<p style="color:#c9c3b4;margin:0.35rem 0 0">Chance de inadimplência em 90 dias, na hora.</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for col, (nome, valor) in zip(cols, probs):
        nome_modelo = next(modelo["nome"] for modelo in relatorio["modelos"] if modelo["slug"] == nome)
        col.metric(nome_modelo, _pct(valor))
    puxam = _sins_que_puxam(relatorio, slug, respostas)
    if puxam:
        st.markdown("**O que mais puxou o risco, entre os Sim marcados**")
        for texto in puxam:
            st.markdown(f"- {texto}")
    else:
        st.caption("Nenhum Sim de risco marcado. O perfil está em dia.")
    if caso is None:
        return
    real = "inadimplente" if caso["y"] else "em dia"
    previsto = "inadimplente" if prob >= CORTE_MATRIZ else "em dia"
    acertou = "acertou" if (prob >= CORTE_MATRIZ) == bool(caso["y"]) else "errou"
    st.caption(
        f"Cliente {caso['id']} do teste estava {real}. "
        f"Com corte em {_pct(CORTE_MATRIZ)}, este modelo {acertou}: previu {previsto}."
    )


@st.fragment
def _simulador(relatorio: dict) -> None:
    st.subheader("Simulador")
    st.write("Cada chave é Sim ou Não. A chance muda na hora, sem abrir gaveta.")
    chaves = relatorio["chaves"]
    if "sim_geracao" not in st.session_state:
        _nova_geracao(respostas_em_dia(), None)
    geracao = int(st.session_state["sim_geracao"])
    guardadas = st.session_state.get("sim_respostas") or respostas_em_dia()
    for pergunta in PERGUNTAS:
        chave = _chave(geracao, pergunta["id"])
        if chave not in st.session_state:
            st.session_state[chave] = bool(guardadas[pergunta["id"]])

    modelos = _modelos_atuais()
    respostas = _respostas(geracao)
    linha = aplicar(chaves["em_dia"], chaves["sim"], respostas)
    _placar(relatorio, linha, respostas, modelos, st.session_state.get("sim_caso"))

    acao1, acao2, acao3 = st.columns(3)
    with acao1:
        limpar = st.button("Limpar tudo (perfil em dia)", width="stretch")
    with acao2:
        sortear = st.button("Sortear cliente do teste", width="stretch")
    with acao3:
        atualizar = st.button("Atualizar diagnóstico", width="stretch")
    if limpar:
        _nova_geracao(respostas_em_dia(), None)
        st.rerun(scope="fragment")
    if sortear:
        teste = _amostra_atual()
        row = teste.iloc[int(np.random.default_rng().integers(0, len(teste)))]
        _nova_geracao(
            respostas_da_linha(row),
            {"id": str(row["id_cliente"]), "y": int(row["inadimplente_90d"])},
        )
        st.rerun(scope="fragment")
    if atualizar:
        st.rerun(scope="fragment")

    grupos: dict[str, list[dict]] = {}
    for pergunta in PERGUNTAS:
        grupos.setdefault(pergunta["grupo"], []).append(pergunta)
    colunas = st.columns(3)
    for indice, (grupo, perguntas) in enumerate(grupos.items()):
        with colunas[indice % 3]:
            st.markdown(f'<p class="sim-grupo">{grupo}</p>', unsafe_allow_html=True)
            for pergunta in perguntas:
                st.toggle(
                    pergunta["texto"],
                    key=_chave(geracao, pergunta["id"]),
                )


def render_relato(ir) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if not pronto():
        st.error("O dossiê ainda não foi gerado. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    relatorio = ler_relatorio()
    itens = relatorio.get("importancia", {}).get("itens", {})
    referencia = relatorio.get("referencia")
    if "chaves" not in relatorio or not itens.get(referencia) or "pct" not in itens[referencia][0]:
        st.error("O dossiê está incompleto. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    _cabecalho(relatorio, ir)
    _grupos(relatorio)
    _acuracia(relatorio)
    _itens_por_modelo(relatorio)
    _combinacoes(relatorio)
    _simulador(relatorio)
