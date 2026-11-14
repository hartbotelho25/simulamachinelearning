"""Comitê de Crédito: relato estático e simulador na mesma página.

O placar muda com os controles. Não há gaveta, aba nem formulário congelado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.credito.dataset import BLOCO_NOME, BLOCOS, CATEGORICAS, FEATURES, LIMITES, ROTULOS
from src.credito.politica import CORTE_MATRIZ, nome_faixa
from src.credito.simulador import (
    amostra_teste,
    carregar_modelos,
    chance,
    figura,
    ler_perfil,
    ler_relatorio,
    pronto,
)

ROTULO_CAT = {
    "clt": "CLT",
    "servidor": "Servidor",
    "autonomo": "Autônomo",
    "empresario": "Empresário",
    "aposentado": "Aposentado",
    "informal": "Informal",
    "fundamental": "Fundamental",
    "medio": "Médio",
    "superior": "Superior",
    "pos": "Pós-graduação",
    "integral": "Integral",
    "parcial": "Parcial",
    "minimo": "Mínimo",
}
FLAGS = {
    "possui_imovel_proprio",
    "flag_pagamento_cartao_com_cheque_especial",
    "flag_historico_fpd",
    "possui_renegociacao_ativa",
}
SLIDERS = {
    "renda_declarada",
    "comprometimento_renda_efetivo",
    "score_bureau_atual",
    "utilizacao_cheque_especial_atual",
    "dias_utilizando_cheque_especial_12m",
}
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
    .grupo li { margin: 0.16rem 0; color: #F4F1E8; font-size: 0.88rem; list-style: none; }
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
def _modelos_gravados():
    return carregar_modelos()


@st.cache_data(show_spinner=False)
def _teste() -> pd.DataFrame:
    return amostra_teste()


def _valor_tipico(perfil: dict, coluna: str):
    info = perfil[coluna]
    if info["tipo"] == "categoria":
        return info["moda"]
    return info["mediana"]


def _widget(coluna: str, bruto, perfil: dict):
    if coluna in CATEGORICAS:
        opcoes = perfil[coluna]["opcoes"]
        texto = str(bruto)
        return texto if texto in opcoes else opcoes[0]
    if coluna in FLAGS:
        return bool(int(float(bruto)))
    baixo, alto, passo = LIMITES[coluna]
    numero = min(max(float(bruto), float(baixo)), float(alto))
    degraus = round((numero - float(baixo)) / float(passo))
    ajustado = min(max(float(baixo) + degraus * float(passo), float(baixo)), float(alto))
    if isinstance(baixo, int) and isinstance(passo, int):
        return int(round(ajustado))
    return round(ajustado, 4)


def _para_modelo(coluna: str, valor):
    if coluna in FLAGS:
        return int(bool(valor))
    if coluna in CATEGORICAS:
        return str(valor)
    return valor


def _chave(geracao: int, coluna: str) -> str:
    return f"g{geracao}_{coluna}"


def _linha_atual(geracao: int) -> dict:
    linha = {}
    for coluna in FEATURES:
        linha[coluna] = _para_modelo(coluna, st.session_state[_chave(geracao, coluna)])
    return linha


def _nova_geracao(perfil: dict, bruto: dict, caso: dict | None) -> None:
    # Chave nova a cada perfil. O fragmento não reinscreve o valor antigo do controle.
    geracao = int(st.session_state.get("sim_geracao", 0)) + 1
    st.session_state["sim_geracao"] = geracao
    st.session_state["sim_base"] = {coluna: _widget(coluna, bruto[coluna], perfil) for coluna in FEATURES}
    st.session_state["sim_caso"] = caso


def _igual(atual: dict, original: dict) -> bool:
    for coluna in FEATURES:
        if coluna in CATEGORICAS:
            if str(atual[coluna]) != str(original[coluna]):
                return False
        elif abs(float(atual[coluna]) - float(original[coluna])) > 1e-4:
            return False
    return True


def _acionados(relatorio: dict, slug: str, linha: dict, perfil: dict) -> list[dict]:
    sinais = relatorio["importancia"]["sinais"]
    acesos = []
    for item in relatorio["importancia"]["itens"][slug]:
        coluna = item["variavel"]
        sinal = sinais[coluna]
        valor = linha[coluna]
        if sinal["sentido"] == "categoria":
            ligado = bool(sinal.get("acima_da_base")) and str(valor) == sinal["nivel"]
        else:
            mediana = float(perfil[coluna]["mediana"])
            numero = float(valor)
            if coluna in FLAGS or mediana <= 0:
                ligado = numero >= 1 if sinal["sentido"] == "sobe" else numero < 1
            elif sinal["sentido"] == "sobe":
                ligado = numero >= mediana * 1.25
            else:
                ligado = numero <= mediana * 0.80
        if ligado:
            acesos.append(item)
    return acesos[:5]


def _cabecalho(relatorio: dict, ir) -> None:
    if st.button("← Voltar ao menu inicial", key="voltar_credito"):
        ir("home")
    base = relatorio["base"]
    acuracias = [m["matriz"]["acuracia"] for m in relatorio["modelos"]]
    st.markdown('<p class="credit-top">Desenvolvido por Hart Botelho</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Base sintética corporativa · 50.000 clientes · 38 colunas</div>
            <h1>Comitê de Crédito</h1>
            <p>O relato já está calculado. Embaixo, o simulador muda a chance na hora.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sintetico">
            <strong>Não é carteira de banco.</strong>
            50.000 clientes gerados, 38 colunas, processo aberto.
            Os números são do teste desta base.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi" style="--accent:#E8B923"><div class="lbl">Clientes</div>
            <div class="val">{_mil(base['n'])}</div><div class="sub">Livro sintético</div></div>
            <div class="kpi" style="--accent:#ff8a65"><div class="lbl">Inadimplência</div>
            <div class="val">{_pct(base['taxa'])}</div><div class="sub">{_mil(base['positivos'])} em 90 dias</div></div>
            <div class="kpi" style="--accent:#6ea8fe"><div class="lbl">Teste</div>
            <div class="val">{_mil(base['n_teste'])}</div><div class="sub">{_pct(base['taxa_teste'])} inadimplentes</div></div>
            <div class="kpi" style="--accent:#3dd68c"><div class="lbl">Maior acurácia</div>
            <div class="val">{_pct(max(acuracias))}</div><div class="sub">Corte 0,50 no teste</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _grupos(relatorio: dict) -> None:
    blocos = relatorio["importancia"]["por_modelo"][relatorio["referencia"]]
    ordem = [item["bloco"] for item in blocos]
    fatia = {item["bloco"]: item["fatia"] for item in blocos}
    primeiros = set(ordem[:3])
    topo = {item["variavel"] for item in relatorio["importancia"]["itens"][relatorio["referencia"]][:8]}
    nomes = [BLOCO_NOME[bloco] for bloco in ordem[:3]]
    st.markdown("### O que a mesa avalia")
    st.markdown(
        f"O arquivo tem 38 colunas: o identificador, **36 itens** e o alvo de 90 dias. "
        f"Olhe primeiro **{nomes[0]}**, **{nomes[1]}** e **{nomes[2]}**."
    )
    cards = []
    for bloco in ordem:
        classe = "grupo primeiro" if bloco in primeiros else "grupo"
        selo = '<div class="olhe">Observar primeiro</div>' if bloco in primeiros else ""
        itens = []
        for coluna in BLOCOS[bloco]:
            marca = '<span class="ponto">●</span>' if coluna in topo else ""
            itens.append(
                f"<li>{marca}{ROTULOS[coluna]}<span class=\"tec\">{coluna}</span></li>"
            )
        cards.append(
            f'<div class="{classe}">{selo}'
            f"<h3>{BLOCO_NOME[bloco]}</h3>"
            f'<div class="tec">{_pct(fatia[bloco])} do sinal no modelo de referência</div>'
            f"<ul>{''.join(itens)}</ul></div>"
        )
    st.markdown(f'<div class="grade-grupos">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption("O ponto dourado marca os oito itens que mais pesaram no modelo de referência. O identificador não entra no treino.")


def _acuracia(relatorio: dict) -> None:
    st.markdown("### Acerto dos quatro modelos")
    chute = 1.0 - relatorio["base"]["taxa_teste"]
    melhor = max(relatorio["modelos"], key=lambda m: m["matriz"]["acuracia"])
    st.markdown(
        f"Chutar sempre em dia acertaria **{_pct(chute)}** deste teste. "
        f"A acurácia acima disso é o que o modelo acrescenta. "
        f"O recall é a parte do calote que ele pega no corte 0,50."
    )
    colunas = st.columns(4)
    for coluna, modelo in zip(colunas, relatorio["modelos"]):
        m = modelo["matriz"]
        classe = "acc melhor" if modelo["slug"] == melhor["slug"] else "acc"
        selo = '<div class="selo">Maior acerto</div>' if modelo["slug"] == melhor["slug"] else '<div class="selo">&nbsp;</div>'
        with coluna:
            st.markdown(
                f"""
                <div class="{classe}">
                    {selo}
                    <div class="nome">{modelo['nome']}</div>
                    <div class="grande">{_pct(m['acuracia'])}</div>
                    <div class="miudo">Acurácia no teste<br>
                    ROC AUC {_num(modelo['roc_auc'])}<br>
                    Precisão {_pct(m['precisao'])} · Recall {_pct(m['recall'])}<br>
                    F1 {_pct(m['f1'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.image(str(figura(relatorio["figuras"]["matrizes"])), width="stretch")
    st.caption("Matriz no corte 0,50. Verde mais claro é a célula com mais clientes. Acima, quem ficou em dia ou em atraso; ao lado, o que o modelo previu.")


def _itens_por_modelo(relatorio: dict) -> None:
    st.markdown("### O que cada modelo aponta para o calote")
    st.markdown("Embaralhar o item no teste derruba a curva. A barra é essa queda, dentro de cada modelo.")
    colunas = st.columns(4)
    for coluna, modelo in zip(colunas, relatorio["modelos"]):
        ranking = relatorio["importancia"]["itens"][modelo["slug"]][:8]
        teto = ranking[0]["queda_auc"] if ranking else 1.0
        linhas = []
        for item in ranking:
            largura = 8 + 92 * (item["queda_auc"] / teto if teto else 0)
            linhas.append(
                f'<div class="rank-nome">{item["rotulo"]}</div>'
                f'<div class="rank-barra"><span style="width:{largura:.0f}%"></span></div>'
            )
        with coluna:
            st.markdown(f"**{modelo['nome']}**")
            st.markdown("".join(linhas), unsafe_allow_html=True)


def _combinacoes(relatorio: dict) -> None:
    todas = relatorio.get("combinacoes") or []
    criticas = [item for item in todas if item.get("critica")] or todas
    st.markdown("### Combinações que travam a aprovação")
    if not criticas:
        st.caption("Nenhuma combinação ficou com amostra suficiente neste teste.")
        return
    base = criticas[0]["taxa_base"]
    st.markdown(
        f"Medidas no teste, não no texto. A base deste teste é **{_pct(base)}**. "
        "A prevista é a chance média do modelo de referência nos mesmos clientes."
    )
    esquerda, direita = st.columns(2)
    for i, item in enumerate(criticas):
        sinais = " · ".join(item["sinais"])
        bloco = f"""
        <div class="combo">
            <h3>{item['titulo']}</h3>
            <p>{sinais}</p>
            <p>{_mil(item['n'])} clientes no teste</p>
            <p>Base <strong>{_pct(item['taxa_base'])}</strong>
            · observada <strong>{_pct(item['taxa_observada'])}</strong>
            · prevista <strong>{_pct(item['prevista_referencia'])}</strong></p>
        </div>
        """
        (esquerda if i % 2 == 0 else direita).markdown(bloco, unsafe_allow_html=True)
    if "combinacoes" in relatorio.get("figuras", {}):
        st.image(str(figura(relatorio["figuras"]["combinacoes"])), width="stretch")
        st.caption("Cada combinação: base do teste, taxa que aconteceu e chance média prevista.")


def _controle(coluna: str, perfil: dict, geracao: int) -> None:
    rotulo = ROTULOS[coluna]
    chave = _chave(geracao, coluna)
    if coluna in CATEGORICAS:
        st.selectbox(
            rotulo,
            perfil[coluna]["opcoes"],
            key=chave,
            format_func=lambda valor: ROTULO_CAT.get(valor, valor),
            help=coluna,
        )
        return
    if coluna in FLAGS:
        st.toggle(rotulo, key=chave, help=coluna)
        return
    baixo, alto, passo = LIMITES[coluna]
    if coluna in SLIDERS:
        st.slider(rotulo, baixo, alto, step=passo, key=chave, help=coluna)
        return
    st.number_input(rotulo, min_value=baixo, max_value=alto, step=passo, key=chave, help=coluna)


def _placar(relatorio: dict, linha: dict, perfil: dict, modelos: dict) -> None:
    nomes = {item["slug"]: item["nome"] for item in relatorio["modelos"]}
    slug = st.selectbox(
        "Parecer pelo modelo",
        list(nomes),
        format_func=lambda s: nomes[s],
        key="sim_modelo",
    )
    probs = {item: chance(modelos[item], linha) for item in nomes}
    faixa, acao = nome_faixa(probs[slug])
    cor = COR_FAIXA[faixa]
    pesos = _acionados(relatorio, slug, linha, perfil)
    if pesos:
        lista = "".join(f"<li>{item['rotulo']}</li>" for item in pesos)
        pesos_html = f"<p>Pesaram para o calote neste cliente:</p><ul>{lista}</ul>"
    else:
        pesos_html = "<p>Nenhum item está do lado do calote em relação ao perfil base.</p>"
    caso = st.session_state.get("sim_caso")
    nota = ""
    if caso:
        real = "inadimplente" if int(caso["y"]) == 1 else "em dia"
        if _igual(linha, caso["linha"]):
            previsto = int(probs[slug] >= CORTE_MATRIZ)
            leitura = "acertou" if previsto == int(caso["y"]) else "errou"
            nota = (
                f"<p>Cliente {caso['id']} do teste. Desfecho real: <strong>{real}</strong>. "
                f"No corte 0,50 este modelo <strong>{leitura}</strong>.</p>"
            )
        else:
            nota = (
                f"<p>Partiu do cliente {caso['id']} (desfecho real: {real}). "
                "Os itens já foram editados. O placar é desta edição.</p>"
            )
    mini = []
    for item, prob in probs.items():
        destaque = "font-weight:700;color:#F4F1E8" if item == slug else "color:#c9c3b4"
        mini.append(f'<div style="{destaque}">{nomes[item]}<br><span style="font-size:1.15rem">{_pct(prob)}</span></div>')
    st.markdown(
        f"""
        <div class="placar">
            <div class="nome" style="color:#c9c3b4;font-size:0.82rem">{nomes[slug]} · chance de inadimplência</div>
            <div class="grande" style="color:{cor}">{_pct(probs[slug])}</div>
            <div class="parecer" style="color:{cor}">{NOME_FAIXA[faixa]} · {acao}</div>
            <div class="miudo" style="color:#c9c3b4;margin-top:0.45rem">Verde até 8% aprova. Amarelo até 20% exige garantia. Acima, recusa.</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.4rem;margin-top:0.7rem">{''.join(mini)}</div>
            {pesos_html}
            {nota}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment
def _simulador(relatorio: dict) -> None:
    perfil = ler_perfil()
    if "sim_base" not in st.session_state:
        st.session_state["sim_base"] = {
            coluna: _widget(coluna, _valor_tipico(perfil, coluna), perfil) for coluna in FEATURES
        }
        st.session_state["sim_geracao"] = 0
        st.session_state["sim_caso"] = None
    geracao = int(st.session_state["sim_geracao"])
    for coluna in FEATURES:
        chave = _chave(geracao, coluna)
        if chave not in st.session_state:
            st.session_state[chave] = st.session_state["sim_base"][coluna]
    if "sim_modelo" not in st.session_state:
        st.session_state["sim_modelo"] = relatorio["referencia"]

    st.markdown("### Simulador do comitê")
    st.markdown("Tudo aberto. A chance, a faixa e o parecer mudam quando um item muda.")

    try:
        modelos = _modelos_gravados()
    except Exception:
        st.warning(
            "Os modelos gravados não abriram nesta instalação. O relato acima continua. "
            "requirements.txt fixa scikit-learn e xgboost na versão em que eles foram salvos."
        )
        return

    linha = _linha_atual(geracao)
    _placar(relatorio, linha, perfil, modelos)

    b1, b2, b3 = st.columns(3)
    with b1:
        calcular = st.button("Calcular / Atualizar diagnóstico", type="primary", width="stretch", key="sim_calcular")
    with b2:
        limpar = st.button("Limpar para perfil base", width="stretch", key="sim_limpar")
    with b3:
        sortear = st.button("Sortear cliente real do teste", width="stretch", key="sim_sortear")
    if calcular:
        st.caption("Diagnóstico atualizado com os itens que estão na tela.")
    if limpar:
        tipico = {coluna: _valor_tipico(perfil, coluna) for coluna in FEATURES}
        _nova_geracao(perfil, tipico, None)
        st.rerun(scope="fragment")
    if sortear:
        teste = _teste()
        st.session_state["sim_sorteio"] = int(st.session_state.get("sim_sorteio", 0)) + 1
        indice = int(np.random.default_rng(st.session_state["sim_sorteio"]).integers(0, len(teste)))
        row = teste.iloc[indice]
        bruto = {coluna: row[coluna] for coluna in FEATURES}
        caso = {
            "id": str(row["id_cliente"]),
            "y": int(row["inadimplente_90d"]),
            "linha": {coluna: _para_modelo(coluna, _widget(coluna, bruto[coluna], perfil)) for coluna in FEATURES},
        }
        _nova_geracao(perfil, bruto, caso)
        st.rerun(scope="fragment")

    blocos = relatorio["importancia"]["por_modelo"][relatorio["referencia"]]
    ordem = [item["bloco"] for item in blocos]
    primeiros = set(ordem[:3])
    colunas = st.columns(4)
    for i, bloco in enumerate(ordem):
        with colunas[i % 4]:
            selo = '<p class="olhe">Olhar primeiro</p>' if bloco in primeiros else ""
            st.markdown(f'<p class="sim-grupo">{BLOCO_NOME[bloco]}</p>{selo}', unsafe_allow_html=True)
            for coluna in BLOCOS[bloco]:
                _controle(coluna, perfil, geracao)


def render_relato(ir) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if not pronto():
        st.error("O dossiê ainda não foi gerado. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    relatorio = ler_relatorio()
    if "itens" not in relatorio.get("importancia", {}) or "combinacoes" not in relatorio:
        st.error("O dossiê está incompleto. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    _cabecalho(relatorio, ir)
    _grupos(relatorio)
    _acuracia(relatorio)
    _itens_por_modelo(relatorio)
    _combinacoes(relatorio)
    _simulador(relatorio)
